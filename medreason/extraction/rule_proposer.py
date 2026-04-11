"""Rule proposer — turns a verified critic trace into candidate ReasoningRules.

Critical invariants enforced here:

1. **The proposer never sees agent reasoning.** Its input is the
   critic-verified trace + the policy excerpt + task_config. The risk
   #14 in the plan is that a refactor might accidentally pipe
   agent_result.reasoning_chain into the proposer context, letting the
   agent's (possibly bad) reasoning influence future rules. A unit test
   asserts this input is never present.

2. **Citations are validated against the actual policy.** The plan
   risk #2 is that LLMs hallucinate citations like "CMS LCD L99999
   §X.Y.Z". Every candidate rule is checked: its
   source_policy_citation must reference a section id that exists in
   the provided `LCDPolicy`. Unvalidated rules are rejected before
   construction.

3. **Patient identifiers trigger rejection.** A regex pass filters out
   rules whose action or rationale contains names, dates, ages, or
   other potentially identifying strings.

4. **Action length ≤ 25 words.** Structural guard against
   multi-criterion bundling.

5. **Test-set case_ids can never appear in supporting_case_ids.** This
   is enforced downstream by the LeakGuard when the rule is put() into
   the RuleStore, but callers of propose_rules() control the
   supporting_case_ids list directly — make sure you pass only
   train+dev ids.

`propose_rules()` returns a ProposalResult with two lists: the
successfully constructed `candidates` and the `rejected` entries paired
with their reason. Nothing is written to any store here — that's the
caller's job after the generalization gate runs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Optional
from uuid import uuid4

from ..llm.base import LLMClient
from ..ontology.case import Payer
from ..ontology.codes import CPTFamily, ICD10Chapter
from ..ontology.rule import (
    ReasoningRule,
    RuleEvidence,
    RuleStatus,
    RuleTrigger,
)
from ..ontology.trace import ReasoningTrace
from ..prompts import load_prompt
from ..runners._prompting import ResponseParseError, parse_json_response


PROPOSER_PROMPT_FILE = "rule_proposer.txt"

# Maximum word count for a rule's action string. Enforced here at
# proposer validation time. The injector displays the action verbatim;
# the limit exists to keep rules atomic (bundling multiple criteria
# into a long sentence is the fail mode we're guarding against).
ACTION_MAX_WORDS = 25


# ── Enum parsers that accept either the enum value or the member name ──────


def _parse_cpt_family(raw: str) -> Optional[CPTFamily]:
    """Accept either the snake_case value ("imaging_mri") or the upper
    member name ("IMAGING_MRI"). Returns None on no match."""
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    try:
        return CPTFamily(s)
    except ValueError:
        pass
    try:
        return CPTFamily[s.upper()]
    except KeyError:
        return None


def _parse_icd10_chapter(raw: str) -> Optional[ICD10Chapter]:
    """Accept either the range value ("M00-M99"), the member name
    ("musculoskeletal" / "MUSCULOSKELETAL"), or an empty-string pass.

    The LLM prompt uses semantic names because the range strings are
    unnatural. This parser is the bridge between that UX and the
    underlying enum whose canonical values are the chapter ranges.
    """
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    try:
        return ICD10Chapter(s)  # tries the range value
    except ValueError:
        pass
    try:
        return ICD10Chapter[s.upper()]  # tries the member name
    except KeyError:
        return None


def _parse_payer(raw: str) -> Optional[Payer]:
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    try:
        return Payer(s)
    except ValueError:
        pass
    try:
        return Payer[s.upper()]
    except KeyError:
        return None


# ── Error / rejection types ─────────────────────────────────────────────────


class PolicyCitationError(ValueError):
    """Citation references a section that does not exist in the policy."""


class PatientIdentifierError(ValueError):
    """Rule text contains patient-identifying strings."""


ProposalRejection = tuple[dict, str]  # (raw_candidate_dict, reason)


@dataclass
class ProposalResult:
    candidates: list[ReasoningRule] = field(default_factory=list)
    rejected: list[ProposalRejection] = field(default_factory=list)
    raw_response_text: str = ""

    @property
    def n_candidates(self) -> int:
        return len(self.candidates)

    @property
    def n_rejected(self) -> int:
        return len(self.rejected)


# ── Validation predicates ───────────────────────────────────────────────────


# Forbidden string patterns: ages, date-like strings, and a small
# proper-noun blacklist. This is a filter, not a perfect PII scanner —
# the real CMS LCD sources we'll ingest in Phase 6 are already
# de-identified. Phase 5 just needs to catch the obvious cases the
# proposer might regurgitate from the trace.
_FORBIDDEN_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"\b\d{1,3}\s*y/o\b", re.IGNORECASE),        # "58 y/o"
    re.compile(r"\b\d{1,3}\s*years?\s*old\b", re.IGNORECASE),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),                    # ISO date
    re.compile(r"\bpatient\s+[A-Z][a-z]+\b"),                # "patient John"
    re.compile(r"\bMr\.?\s+[A-Z]", re.IGNORECASE),
    re.compile(r"\bMrs\.?\s+[A-Z]", re.IGNORECASE),
    re.compile(r"\bDr\.?\s+[A-Z][a-z]+"),
)


def _has_patient_identifier(text: str) -> Optional[str]:
    """Return the offending match string, or None if clean."""
    for pattern in _FORBIDDEN_PATTERNS:
        m = pattern.search(text)
        if m:
            return m.group(0)
    return None


def _citation_exists_in_policy(citation: str, policy) -> bool:
    """Return True iff `citation` references a section id present in
    `policy.indications` or `policy.limitations`.

    Accepts both "CMS LCD L34522 §C.1" and "L34522 §C.1" style.
    """
    if not citation:
        return False
    # Extract the section id fragment after §
    idx = citation.find("§")
    if idx == -1:
        # Tolerate the ASCII fallback '§' missing — look for a
        # capital-letter dotted id like "C.1" or "L.2" at the end.
        m = re.search(r"\b([CL])\.\d+\b", citation)
        if not m:
            return False
        section_id = m.group(0)
    else:
        tail = citation[idx + 1:].strip()
        m = re.match(r"^([A-Z]\.\d+)", tail)
        if not m:
            return False
        section_id = m.group(1)

    valid_ids = set()
    for crit in getattr(policy, "indications", []):
        valid_ids.add(crit.criterion_id)
    for lim in getattr(policy, "limitations", []):
        valid_ids.add(lim.limitation_id)
    return section_id in valid_ids


def _count_words(text: str) -> int:
    return len(re.findall(r"\S+", text))


# ── Candidate construction ──────────────────────────────────────────────────


def _build_rule_from_candidate(
    cand: dict,
    *,
    policy,
    supporting_case_ids: list[str],
    proposer_model: str,
    proposer_run_id: str,
) -> ReasoningRule:
    """Validate a single candidate dict and build a ReasoningRule.

    Raises ValueError (or a subclass) with a specific reason on any
    validation failure. Callers catch the exception and add
    (candidate, reason) to ProposalResult.rejected.
    """
    trigger_data = cand.get("trigger") or {}
    if not isinstance(trigger_data, dict):
        raise ValueError("trigger is not an object")

    # Enum parsing — accepts either value or member name, drops unknowns
    cpt_fams: list[CPTFamily] = []
    for fam_raw in trigger_data.get("cpt_families") or []:
        parsed = _parse_cpt_family(str(fam_raw))
        if parsed is not None:
            cpt_fams.append(parsed)
    icd_chaps: list[ICD10Chapter] = []
    for ch_raw in trigger_data.get("icd10_chapters") or []:
        parsed_ch = _parse_icd10_chapter(str(ch_raw))
        if parsed_ch is not None:
            icd_chaps.append(parsed_ch)
    payers: list[Payer] = []
    for p_raw in trigger_data.get("payers") or []:
        parsed_p = _parse_payer(str(p_raw))
        if parsed_p is not None:
            payers.append(parsed_p)

    semantic_predicate = str(trigger_data.get("semantic_predicate", "")).strip()

    trigger = RuleTrigger(
        cpt_families=cpt_fams,
        icd10_chapters=icd_chaps,
        payers=payers,
        semantic_predicate=semantic_predicate,
    )

    action = str(cand.get("action", "")).strip()
    if not action:
        raise ValueError("action is empty")
    if _count_words(action) > ACTION_MAX_WORDS:
        raise ValueError(
            f"action exceeds {ACTION_MAX_WORDS}-word limit "
            f"({_count_words(action)} words)"
        )

    rationale = str(cand.get("rationale", "")).strip()
    polarity_raw = str(cand.get("polarity", "requires_check")).strip()
    if polarity_raw not in ("supports_approval", "supports_denial", "requires_check"):
        polarity_raw = "requires_check"
    polarity: Literal[
        "supports_approval", "supports_denial", "requires_check"
    ] = polarity_raw  # type: ignore[assignment]

    citation = str(cand.get("source_policy_citation", "")).strip()
    if not citation:
        raise PolicyCitationError("missing source_policy_citation")
    if not _citation_exists_in_policy(citation, policy):
        raise PolicyCitationError(
            f"citation {citation!r} not found in policy"
        )

    # Patient-identifier scan across all user-visible rule text
    for field_name, text in (
        ("action", action),
        ("rationale", rationale),
        ("semantic_predicate", semantic_predicate),
    ):
        offender = _has_patient_identifier(text)
        if offender is not None:
            raise PatientIdentifierError(
                f"{field_name} contains identifier-like string {offender!r}"
            )

    evidence = RuleEvidence(
        supporting_case_ids=list(supporting_case_ids),
        source_policy_citation=citation,
        source_policy_url=getattr(policy, "url", None),
        extracted_from_trace_ids=[],  # filled by caller if available
        proposer_model=proposer_model,
        proposer_run_id=proposer_run_id,
    )

    return ReasoningRule(
        status=RuleStatus.CANDIDATE,
        trigger=trigger,
        action=action,
        rationale=rationale,
        polarity=polarity,
        evidence=evidence,
    )


# ── Public API ──────────────────────────────────────────────────────────────


def propose_rules(
    trace: ReasoningTrace,
    policy,
    llm_client: LLMClient,
    *,
    supporting_case_ids: list[str],
    proposer_run_id: Optional[str] = None,
    max_tokens: int = 2048,
    seed: int = 0,
) -> ProposalResult:
    """Ask the proposer LLM for candidate rules and validate them.

    Input contract: `trace` must be a CRITIC-VERIFIED trace. The plan's
    risk #14 warns against accidentally passing the agent's raw reasoning
    here — do NOT construct `trace` from an AgentResult's reasoning_chain
    without running it through the critic first.

    `policy` is the LCDPolicy (or anything with .indications/.limitations
    lists of objects with .criterion_id/.limitation_id). Citation
    validation walks these lists.

    `supporting_case_ids` must contain only train+dev case_ids. The
    LeakGuard will reject any rule referencing a test case_id at
    store.put() time, but this is the earliest layer where we can guard
    the invariant.
    """
    system_prompt = load_prompt(PROPOSER_PROMPT_FILE)

    # User message — structured, no agent reasoning included.
    user_parts = [
        "## Verified reasoning trace (critic-derived)",
        f"Outcome: {trace.outcome.value}",
        f"Source: {trace.source}",
        "",
        "Reasoning steps:",
    ]
    for step in trace.reasoning_steps:
        user_parts.append(
            f"  {step.step_number}. [{step.decision_branch}] "
            f"{step.action}  — evidence: {step.evidence_cited}"
        )
    user_parts.extend([
        "",
        "## Task config",
        f"Payer: {trace.task_config.payer.value}",
        f"CPT: {trace.task_config.cpt_code}",
        f"ICD-10: {', '.join(trace.task_config.icd10_codes)}",
        f"Facility: {trace.task_config.facility_type.value}",
        "",
        "## Policy excerpt",
    ])
    for crit in getattr(policy, "indications", []):
        user_parts.append(f"§{crit.criterion_id} ({crit.tag}): {crit.text}")
    for lim in getattr(policy, "limitations", []):
        user_parts.append(f"§{lim.limitation_id} ({lim.tag}): {lim.text}")

    user_msg = "\n".join(user_parts)

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
    run_id = proposer_run_id or f"propose_{uuid4().hex[:8]}"

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
