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

from ..llm.base import LLMClient
from ..ontology.rule import ReasoningRule
from .case import BypassMechanism, GroundTruthOutcome, TargetValidationCase
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


def propose_corrective_rules(
    errors: list[SystematicError],
    cases: list[TargetValidationCase],
    llm: LLMClient,
) -> list[ReasoningRule]:
    """For each SystematicError, ask the LLM what corrective rule would
    have prevented it. Returns CANDIDATE-status ReasoningRules — the
    LayerRouter promotion path runs the abstractor + gate before any of
    these enter the UniversalLayer.

    SKELETON: returns []. The wired implementation will:
    1. For each SystematicError with severity >= severity_floor, build a
       summary prompt that lists the affected cases (de-identified —
       opaque case_ids only), the swarm's reasoning, and the ground
       truth pattern.
    2. Reuse medreason.extraction.failure_analyzer.analyze_failure with
       a tweaked prompt that asks for a CROSS-CASE rule (not a
       per-case rule). The prompt requires the rule's predicate to
       describe the structural condition (e.g., "kinase target with
       paralog_count >= N") not specific gene symbols.
    3. Each returned ReasoningRule has CANDIDATE status. Promotion to
       ACTIVE requires the generalization_gate to pass on held-out
       targets.
    """
    _ = (errors, cases, llm)
    return []


def run_cross_agent_analysis(
    report: SwarmReport,
    cases: list[TargetValidationCase],
    llm: LLMClient,
    *,
    bypass_risk_threshold: float = 0.5,
    severity_floor: float = _DEFAULT_SEVERITY_FLOOR,
) -> CrossAgentAnalysis:
    """End-to-end: detect errors → propose rules → return analysis."""
    errors = detect_systematic_errors(
        report,
        cases,
        bypass_risk_threshold=bypass_risk_threshold,
        severity_floor=severity_floor,
    )
    rules = propose_corrective_rules(errors, cases, llm)
    return CrossAgentAnalysis(systematic_errors=errors, candidate_rules=rules)
