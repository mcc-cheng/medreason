"""Cross-agent metacognitive analyzer.

The Phase 51 failure_analyzer (medreason/extraction/failure_analyzer.py)
operates on a SINGLE wrong case: it asks "what rule would have flipped
this case?" The cross-agent analyzer is the swarm-scale equivalent: it
looks at a SwarmReport plus ground-truth labels (in the retro benchmark)
or reviewer labels (in production), identifies SYSTEMATIC reasoning
failures across multiple targets, and proposes corrective rules.

Concrete example:
  - Swarm runs on 30 historical MAPK targets.
  - 8 of those 30 had paralog-compensation bypass in real trials.
  - The swarm assigned bypass_risk_score < 0.4 to 7 of those 8.
  - That's a systematic miss: the agents are not penalising paralog
    redundancy enough.
  - The cross-agent analyzer extracts a candidate UNIVERSAL-layer rule:
    "When paralog_count >= 2 for a kinase target, raise bypass_risk by
    at least 0.3" — citation: aggregate of the 8 missed cases.
  - The rule then goes through rule_abstractor + generalization_gate
    before LayerRouter writes it to UniversalLayer.

This is the *moat* mechanism. The user's pitch says "metacognitive
memory stores where reasoning predictably goes wrong" — this is the
multi-agent realization of that pattern.

Skeleton scope: data types + the orchestration shape (group-by-error +
propose call), with the actual LLM-driven rule generation deferred to
the existing failure_analyzer wired against a swarm-summary prompt.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
from uuid import uuid4

from ..extraction.rule_proposer import (
    ACTION_MAX_WORDS,
    _count_words,
    _has_patient_identifier,
)
from ..llm.base import LLMClient
from ..ontology.rule import (
    ReasoningRule,
    RuleEvidence,
    RuleStatus,
    RuleTrigger,
)
from ..runners._prompting import ResponseParseError, parse_json_response
from .case import BypassMechanism, GroundTruthOutcome, TargetValidationCase
from .cross_agent_prompts import (
    REPAIR_SUFFIX,
    SYSTEM_PROMPT_CROSS_AGENT,
    build_proposer_prompt,
)
from .layers import Layer, LayerPolicy, LayerPolicyViolation
from .swarm import SwarmReport, TargetMemo


# ── Error classification ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class SystematicError:
    """A category of error the swarm makes across multiple cases.

    Attributes
    ----------
    error_kind
        Free-text tag, e.g. "missed_paralog_bypass", "over_credited_genetics".
    affected_case_ids
        case_ids the analyzer grouped under this error.
    severity
        Crude prevalence-weighted severity: |affected| / |total_eligible|.
        Eligible = cases for which this error category is *applicable*
        (e.g., the denominator for "missed_paralog_bypass" is cases that
        actually had paralog-bypass ground truth).
    """

    error_kind: str
    affected_case_ids: tuple[str, ...]
    severity: float
    description: str = ""


# ── Orchestrator ──────────────────────────────────────────────────────────────


@dataclass
class CrossAgentAnalysis:
    systematic_errors: list[SystematicError] = field(default_factory=list)
    candidate_rules: list[ReasoningRule] = field(default_factory=list)
    cost_usd: float = 0.0


# Severity threshold below which we don't bother proposing a rule. The
# Phase 51 lesson is that low-prevalence errors generate noise; only
# patterns the swarm misses at meaningful rates earn a universal rule.
_DEFAULT_SEVERITY_FLOOR = 0.35


def detect_systematic_errors(
    report: SwarmReport,
    cases: list[TargetValidationCase],
    *,
    bypass_risk_threshold: float = 0.5,
    severity_floor: float = _DEFAULT_SEVERITY_FLOOR,
) -> list[SystematicError]:
    """Compare swarm memos against retrospective ground truth and bucket
    errors by category.

    Categories detected in v0.1:
    - missed_bypass_<mechanism>: ground truth shows mechanism M, swarm's
      bypass_risk_score for those cases is below threshold.
    - false_alarm_bypass: ground truth shows no bypass, swarm flagged
      bypass with score above threshold.
    - missed_efficacy: ground truth is phase2_efficacy_yes but swarm
      ranked the target outside the top half.
    - missed_failure: ground truth is phase2_efficacy_no but swarm
      ranked the target inside the top quarter.
    """
    memos_by_id = {m.case_id: m for m in report.memos}
    errors: list[SystematicError] = []

    # ── missed_bypass_<M> ───────────────────────────────────────────────
    by_mechanism: dict[BypassMechanism, list[str]] = defaultdict(list)
    eligible_by_mechanism: dict[BypassMechanism, int] = defaultdict(int)
    for case in cases:
        if case.ground_truth_bypass in (
            BypassMechanism.UNKNOWN,
            BypassMechanism.NO_BYPASS_KNOWN,
        ):
            continue
        eligible_by_mechanism[case.ground_truth_bypass] += 1
        memo = memos_by_id.get(case.case_id)
        if memo is None:
            continue
        if memo.bypass_risk_score < bypass_risk_threshold:
            by_mechanism[case.ground_truth_bypass].append(case.case_id)

    for mech, affected in by_mechanism.items():
        denom = eligible_by_mechanism[mech] or 1
        severity = len(affected) / denom
        if severity >= severity_floor:
            errors.append(
                SystematicError(
                    error_kind=f"missed_bypass_{mech.value}",
                    affected_case_ids=tuple(affected),
                    severity=severity,
                    description=(
                        f"Swarm under-scored bypass risk on "
                        f"{len(affected)}/{denom} cases with ground-truth "
                        f"bypass mechanism {mech.value}"
                    ),
                )
            )

    # ── missed_efficacy ────────────────────────────────────────────────
    success_cases = [
        c
        for c in cases
        if c.ground_truth_outcome
        in (GroundTruthOutcome.PHASE2_EFFICACY_YES, GroundTruthOutcome.APPROVED_LATER)
    ]
    if success_cases:
        half = max(1, len(report.ranking) // 2)
        top_half_ids = set(_ids_for_ranks(report, range(0, half)))
        missed = [
            c.case_id
            for c in success_cases
            if memos_by_id.get(c.case_id)
            and memos_by_id[c.case_id].gene_symbol not in top_half_ids
        ]
        if missed:
            severity = len(missed) / len(success_cases)
            if severity >= severity_floor:
                errors.append(
                    SystematicError(
                        error_kind="missed_efficacy",
                        affected_case_ids=tuple(missed),
                        severity=severity,
                        description=(
                            f"Swarm ranked {len(missed)}/{len(success_cases)} "
                            f"phase-2-positive targets outside the top half"
                        ),
                    )
                )

    return errors


def _ids_for_ranks(report: SwarmReport, rank_range) -> list[str]:
    return [report.ranking[i] for i in rank_range if 0 <= i < len(report.ranking)]


# ── Proposer ──────────────────────────────────────────────────────────────────

# Universal-layer policy instance — used as defense-in-depth check on
# every emitted rule. Stateless / immutable, safe to module-scope.
_UNIVERSAL_POLICY = LayerPolicy(layer=Layer.UNIVERSAL)


def propose_corrective_rules(
    errors: list[SystematicError],
    cases: list[TargetValidationCase],
    llm: LLMClient,
    *,
    proposer_run_id: Optional[str] = None,
    severity_floor: float = _DEFAULT_SEVERITY_FLOOR,
    seed: int = 0,
    memos: Optional[list[TargetMemo]] = None,
) -> list[ReasoningRule]:
    """Turn SystematicErrors into CANDIDATE-status UNIVERSAL-layer rules.

    For each ``SystematicError`` with ``severity >= severity_floor``,
    builds a de-identified prompt (see ``cross_agent_prompts``), calls
    the LLM, parses a ``{"rules": [...]}`` JSON shape, validates each
    candidate, and runs it through ``LayerPolicy.validate_rule`` for
    ``Layer.UNIVERSAL`` as defense-in-depth.

    Validation drops (silently — there is no ProposalResult wrapper at
    this layer):
      - empty ``action`` or ``semantic_predicate``
      - ``_count_words(action) > ACTION_MAX_WORDS``
      - ``_has_patient_identifier`` hit on any user-visible text
      - candidate dict carries a non-None ``customer_tag`` field (the
        LLM is not allowed to scope corrective rules to a tenant —
        these are universal)
      - ``LayerPolicy.validate_rule`` raises ``LayerPolicyViolation``

    Failure modes:
      - LLM ``complete()`` raises → that error is treated as zero rules
        for the offending SystematicError (other errors still proceed).
      - Malformed JSON → ONE repair attempt with a tighter prompt; if
        the repair also fails, that SystematicError contributes zero
        rules.
      - This function NEVER raises. Permanent failure → returns ``[]``
        (or the rules accumulated from earlier errors).

    Parameters
    ----------
    errors
        Systematic-error groupings from ``detect_systematic_errors``.
    cases
        Full ``TargetValidationCase`` list used by the prompt builder
        for ground-truth field lookup. ``InternalEvidence.readouts``
        is filtered at the prompt-builder boundary.
    llm
        Any ``LLMClient`` (tests pass ``FakeLLMClient``).
    proposer_run_id
        Optional stable id stamped into ``RuleEvidence.proposer_run_id``;
        defaults to ``crossagent_<8-hex>``.
    severity_floor
        Drop any error below this severity. Mirrors the floor used in
        ``detect_systematic_errors`` so callers can re-filter if they
        passed a different floor upstream.
    seed
        Forwarded to ``llm.complete`` so tests are deterministic.
    memos
        Optional swarm memos for richer prompt context (lets the
        analyzer cite the swarm's ``bypass_signals_seen``). Pure
        signal — never leaked into ``RuleEvidence``.
    """
    if not errors:
        return []

    run_id = proposer_run_id or f"crossagent_{uuid4().hex[:8]}"
    surviving: list[ReasoningRule] = []

    for err in errors:
        if err.severity < severity_floor:
            continue

        user_msg = build_proposer_prompt(err, cases, memos=memos)

        # 1) primary LLM call (best-effort; do not raise from this fn).
        data = _llm_complete_and_parse(
            llm,
            system=SYSTEM_PROMPT_CROSS_AGENT,
            user=user_msg,
            seed=seed,
        )

        # 2) repair-once on malformed JSON.
        if data is None:
            repaired_user = user_msg + REPAIR_SUFFIX
            data = _llm_complete_and_parse(
                llm,
                system=SYSTEM_PROMPT_CROSS_AGENT,
                user=repaired_user,
                seed=seed,
            )

        if data is None:
            continue

        rules_raw = data.get("rules")
        if not isinstance(rules_raw, list):
            continue

        for cand in rules_raw:
            rule = _try_build_corrective_rule(
                cand,
                error=err,
                model=getattr(llm, "model_version", ""),
                run_id=run_id,
            )
            if rule is None:
                continue
            surviving.append(rule)

    return surviving


def _llm_complete_and_parse(
    llm: LLMClient,
    *,
    system: str,
    user: str,
    seed: int,
) -> Optional[dict]:
    """Best-effort: call the LLM and parse a JSON object. Returns the
    parsed dict on success, or ``None`` on any failure (LLM exception,
    parse error, non-object). Never raises.
    """
    try:
        response = llm.complete(system=system, user=user, seed=seed)
    except Exception:
        return None
    raw = response.text or ""
    try:
        return parse_json_response(raw)
    except ResponseParseError:
        return None


def _try_build_corrective_rule(
    cand,
    *,
    error: SystematicError,
    model: str,
    run_id: str,
) -> Optional[ReasoningRule]:
    """Validate a single LLM candidate dict and build a CANDIDATE-status
    UNIVERSAL ReasoningRule. Returns ``None`` if the candidate fails any
    structural or policy check. Defense-in-depth: even after all the
    structural checks pass, the rule is run through
    ``LayerPolicy(layer=UNIVERSAL).validate_rule(rule, customer_tag=None)``
    so any future regression that lets a customer_tag leak in still gets
    blocked here.
    """
    if not isinstance(cand, dict):
        return None

    # Reject any LLM attempt to tag this rule to a tenant — corrective
    # rules from the cross-agent analyzer are universal by construction.
    if cand.get("customer_tag") not in (None, ""):
        return None

    semantic_predicate = str(cand.get("semantic_predicate", "")).strip()
    action = str(cand.get("action", "")).strip()
    rationale = str(cand.get("rationale", "")).strip()
    if not semantic_predicate or not action:
        return None
    if _count_words(action) > ACTION_MAX_WORDS:
        return None

    for text in (semantic_predicate, action, rationale):
        if _has_patient_identifier(text):
            return None

    polarity_raw = str(cand.get("polarity", "requires_check")).strip()
    if polarity_raw not in (
        "supports_approval",
        "supports_denial",
        "requires_check",
    ):
        polarity_raw = "requires_check"

    trigger = RuleTrigger(semantic_predicate=semantic_predicate)
    evidence = RuleEvidence(
        supporting_case_ids=list(error.affected_case_ids),
        source_policy_citation=(
            f"cross_agent:{error.error_kind}:n={len(error.affected_case_ids)}"
        ),
        proposer_model=model,
        proposer_run_id=run_id,
    )

    try:
        rule = ReasoningRule(
            status=RuleStatus.CANDIDATE,
            trigger=trigger,
            action=action,
            rationale=rationale,
            polarity=polarity_raw,  # validated above
            evidence=evidence,
        )
    except Exception:
        # Pydantic validation hiccup — drop the candidate, never raise.
        return None

    # Defense-in-depth: universal-layer policy must accept this rule.
    try:
        _UNIVERSAL_POLICY.validate_rule(rule, customer_tag=None)
    except LayerPolicyViolation:
        return None

    return rule


def run_cross_agent_analysis(
    report: SwarmReport,
    cases: list[TargetValidationCase],
    llm: LLMClient,
    *,
    bypass_risk_threshold: float = 0.5,
    severity_floor: float = _DEFAULT_SEVERITY_FLOOR,
    proposer_run_id: Optional[str] = None,
    seed: int = 0,
) -> CrossAgentAnalysis:
    """End-to-end: detect errors → propose rules → return analysis.

    The proposer is now wired: ``analysis.candidate_rules`` carries the
    CANDIDATE-status ``ReasoningRule``s emitted by ``propose_corrective_rules``.
    Caller is expected to feed these to ``LayerRouter.ingest_rule`` with
    ``source_layer=Layer.UNIVERSAL`` and ``customer_tag=None``.
    """
    errors = detect_systematic_errors(
        report,
        cases,
        bypass_risk_threshold=bypass_risk_threshold,
        severity_floor=severity_floor,
    )
    rules = propose_corrective_rules(
        errors,
        cases,
        llm,
        proposer_run_id=proposer_run_id,
        severity_floor=severity_floor,
        seed=seed,
        memos=report.memos,
    )
    return CrossAgentAnalysis(systematic_errors=errors, candidate_rules=rules)
