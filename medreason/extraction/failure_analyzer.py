"""Failure analyzer — extract corrective rules from agent mistakes.

The training loop in Phase 5 only learned from cases the agent got
*right*. Every wrong case was discarded after the "agent correct" gate.
The consequence: the hardest cases — the ones memory is supposed to
help with — never produced a rule, because there was no successful
trace to mine. That's the chicken-and-egg identified in the Phase 6
v0.2 failure analysis (SESSION_BRIEF.md §6, issue #2).

This module is Phase 51's corrective extractor. When the agent reaches
the *wrong* determination on a training case, we still have the ground
truth available (it's the benchmark split, after all). We feed the
case, the agent's wrong determination, and the correct outcome into a
failure-analyzer LLM with a prompt that asks: "what atomic policy rule
would have led a reviewer to the correct answer?" The LLM emits
candidates in the SAME JSON schema as rule_proposer.txt, so the same
validation / citation / PII / action-length pipeline is reused.

Invariants that distinguish this from rule_proposer:

1. **No agent reasoning chain is passed in.** Per plan risk #14, the
   same way the rule proposer never sees agent reasoning, this module
   also never sees it. The failure analyzer reasons through the policy
   independently using only (case inputs, agent determination label,
   correct determination label). This keeps the rules policy-level, not
   "patch the specific model's specific mistake."

2. **The ground-truth outcome IS revealed to the analyzer.** This is a
   deliberate asymmetry with the critic. The critic must not see the
   ground truth (it is an independent re-derivation). The failure
   analyzer must see the ground truth (otherwise it can't target the
   correction). The analyzer's rules must still cite the policy — a
   citation existence check is enforced downstream.

3. **Same ProposalResult + rejection reasons as propose_rules.** So
   the training loop's accounting stays uniform.

4. **Does NOT write to any store.** Caller owns persistence (same as
   propose_rules).
"""

from __future__ import annotations

from typing import Optional
from uuid import uuid4

from ..llm.base import LLMClient
from ..ontology.case import BenchmarkCase, Outcome
from ..prompts import load_prompt
from ..runners._prompting import ResponseParseError, parse_json_response
from .rule_proposer import (
    PatientIdentifierError,
    PolicyCitationError,
    ProposalResult,
    _build_rule_from_candidate,
)


FAILURE_ANALYZER_PROMPT_FILE = "failure_analyzer.txt"


def analyze_failure(
    case: BenchmarkCase,
    *,
    agent_determination: Outcome,
    ground_truth_outcome: Outcome,
    llm_client: LLMClient,
    policy,  # LCDPolicy or None (multi-policy mode)
    supporting_case_ids: list[str],
    proposer_run_id: Optional[str] = None,
    max_tokens: int = 2048,
    seed: int = 0,
) -> ProposalResult:
    """Ask the failure-analyzer LLM for corrective rules on a wrong case.

    Input contract:

    - `case` is the benchmark case whose determination the agent got
      wrong. Its clinical_notes and policy_excerpt are rendered into
      the user message. No agent reasoning is included.
    - `agent_determination` is the wrong determination the agent
      reached. Rendered as a structural label only.
    - `ground_truth_outcome` is the correct determination from the
      benchmark's frozen ground truth.
    - `policy` is the structured LCDPolicy used for citation validation.
      Pass None to enter multi-policy wildcard mode (same as
      propose_rules) — each case's own policy_excerpt is used as the
      source text and citation validation accepts any well-formed
      non-empty citation.
    - `supporting_case_ids` seeds the Evidence.supporting_case_ids for
      each promoted rule. Must contain only train+dev ids — the
      LeakGuard at store.put() time will reject test ids anyway, but
      the caller is the earliest layer with context to know which
      split the case came from.

    No-op case: if `agent_determination == ground_truth_outcome`, the
    case was actually correct and should have gone through the normal
    propose_rules path. This function raises ValueError in that case.
    """
    if agent_determination == ground_truth_outcome:
        raise ValueError(
            "analyze_failure called on a correct case "
            f"(agent={agent_determination.value}, gt={ground_truth_outcome.value}). "
            "Use propose_rules for correct cases and analyze_failure only for wrong ones."
        )

    system_prompt = load_prompt(FAILURE_ANALYZER_PROMPT_FILE)

    # Build the user message. We intentionally do NOT include any agent
    # reasoning chain — the analyzer reasons through the policy fresh.
    # We do include:
    #   - task config
    #   - clinical notes
    #   - policy excerpt (structured or raw text)
    #   - agent determination (label only)
    #   - ground truth outcome (label only)
    parts: list[str] = [
        "## Case task config",
        f"Payer: {case.task_config.payer.value}",
        f"CPT: {case.task_config.cpt_code}",
        f"ICD-10: {', '.join(case.task_config.icd10_codes) or '(none)'}",
        f"Facility: {case.task_config.facility_type.value}",
    ]
    if case.task_config.modifiers:
        parts.append(f"Modifiers: {', '.join(case.task_config.modifiers)}")

    parts.extend([
        "",
        "## Clinical notes",
        (case.clinical_notes or "").strip() or "(no notes provided)",
        "",
        "## Policy excerpt",
    ])

    if policy is None:
        # Multi-policy fixture: use the case's own excerpt text.
        parts.append((case.policy_excerpt or "").strip()
                     or "(no policy excerpt provided)")
    else:
        # Structured policy: render its indications/limitations lists.
        for crit in getattr(policy, "indications", []):
            parts.append(f"§{crit.criterion_id} ({crit.tag}): {crit.text}")
        for lim in getattr(policy, "limitations", []):
            parts.append(f"§{lim.limitation_id} ({lim.tag}): {lim.text}")

    parts.extend([
        "",
        "## Prior reviewer's (WRONG) determination",
        agent_determination.value,
        "",
        "## CORRECT determination (verified ground truth)",
        ground_truth_outcome.value,
        "",
        "What atomic policy rule(s) would have led a reviewer to the "
        "correct determination? Emit rules that target this error and "
        "generalize to future cases with the same structural pattern.",
    ])

    user_msg = "\n".join(parts)

    try:
        response = llm_client.complete(
            system=system_prompt,
            user=user_msg,
            max_tokens=max_tokens,
            seed=seed,
        )
    except Exception as e:
        return ProposalResult(
            candidates=[],
            rejected=[({}, f"llm_error: {type(e).__name__}: {e}")],
            raw_response_text="",
        )

    raw = response.text or ""
    try:
        data = parse_json_response(raw)
    except ResponseParseError as e:
        return ProposalResult(
            candidates=[],
            rejected=[({}, f"parse_error: {e}")],
            raw_response_text=raw,
        )

    rules_data = data.get("rules", [])
    if not isinstance(rules_data, list):
        return ProposalResult(
            candidates=[],
            rejected=[(data, "top-level 'rules' is not a list")],
            raw_response_text=raw,
        )

    proposer_model = llm_client.model_version
    run_id = proposer_run_id or f"failure_{uuid4().hex[:8]}"

    result = ProposalResult(raw_response_text=raw)

    for cand in rules_data:
        if not isinstance(cand, dict):
            result.rejected.append(({}, "candidate is not an object"))
            continue
        try:
            rule = _build_rule_from_candidate(
                cand,
                policy=policy,
                supporting_case_ids=supporting_case_ids,
                proposer_model=proposer_model,
                proposer_run_id=run_id,
            )
        except (
            PolicyCitationError,
            PatientIdentifierError,
            ValueError,
        ) as e:
            result.rejected.append((cand, f"{type(e).__name__}: {e}"))
            continue
        result.candidates.append(rule)

    return result
