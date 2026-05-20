"""Tests for the cross-agent metacognitive analyzer.

Canned LLM JSON shape shared with ``mapk-curator``'s end-to-end test:
    {"rules": [{"semantic_predicate": "...", "action": "...",
                 "rationale": "...", "polarity": "requires_check"}]}
"""

from __future__ import annotations

import json

from medreason.llm.base import FakeLLMClient
from medreason.ontology.rule import RuleStatus
from medreason.targetval.case import (
    BypassMechanism,
    GroundTruthOutcome,
)
from medreason.targetval.cross_agent_analyzer import (
    SystematicError,
    detect_systematic_errors,
    propose_corrective_rules,
    run_cross_agent_analysis,
)
from medreason.targetval.swarm import SwarmReport, TargetMemo

from medreason_bench.targetval.synthetic import build_synthetic_targets


# ── Shared canned LLM JSON shapes ────────────────────────────────────────────


_GOOD_RULE_JSON = json.dumps(
    {
        "rules": [
            {
                "semantic_predicate": "kinase target with paralog_count >= 2",
                "action": "Raise bypass_risk by 0.3 when paralog count exceeds two.",
                "rationale": "Paralog redundancy enables compensation post-knockout.",
                "polarity": "requires_check",
            }
        ]
    }
)

_TWO_RULES_JSON = json.dumps(
    {
        "rules": [
            {
                "semantic_predicate": "kinase target with paralog_count >= 2",
                "action": "Raise bypass_risk by 0.3 when paralog count exceeds two.",
                "rationale": "Paralog redundancy enables compensation.",
                "polarity": "requires_check",
            },
            {
                "semantic_predicate": "RTK target lacking PI3K-axis feedback evidence",
                "action": "Penalise priority by 0.2 if downstream feedback unknown.",
                "rationale": "Unverified feedback often hides alternative-pathway escape.",
                "polarity": "supports_denial",
            },
        ]
    }
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _swarm_that_misses_kras_and_egfr_bypass() -> SwarmReport:
    """Swarm assigns low bypass_risk to KRAS and EGFR (which have
    downstream_feedback / alternative_pathway in ground truth).
    """
    return SwarmReport(
        campaign_id="synthetic",
        memos=[
            TargetMemo(
                case_id="TVS-001",
                gene_symbol="BRAF",
                priority_score=0.85,
                bypass_risk_score=0.55,  # caught it
                predicted_bypass=BypassMechanism.DOWNSTREAM_FEEDBACK,
            ),
            TargetMemo(
                case_id="TVS-002",
                gene_symbol="KRAS",
                priority_score=0.5,
                bypass_risk_score=0.2,  # missed
                predicted_bypass=BypassMechanism.NO_BYPASS_KNOWN,
                bypass_signals_seen=["downstream_feedback"],
            ),
            TargetMemo(
                case_id="TVS-003",
                gene_symbol="EGFR",
                priority_score=0.4,
                bypass_risk_score=0.3,  # missed
                predicted_bypass=BypassMechanism.NO_BYPASS_KNOWN,
                bypass_signals_seen=["alternative_pathway"],
            ),
        ],
        ranking=["BRAF", "KRAS", "EGFR"],
    )


# ── detect_systematic_errors (unchanged behaviour) ───────────────────────────


def test_detect_missed_bypass_categories():
    cases = build_synthetic_targets()
    report = _swarm_that_misses_kras_and_egfr_bypass()
    errors = detect_systematic_errors(
        report, cases, bypass_risk_threshold=0.5, severity_floor=0.5
    )
    kinds = {e.error_kind for e in errors}
    # Both downstream_feedback (KRAS) and alternative_pathway (EGFR) should fire
    assert "missed_bypass_downstream_feedback" in kinds
    assert "missed_bypass_alternative_pathway" in kinds


def test_severity_floor_filters_low_prevalence():
    cases = build_synthetic_targets()
    report = _swarm_that_misses_kras_and_egfr_bypass()
    # With a high floor (1.0 = "100% must be missed"), KRAS's 1/1 still
    # qualifies but EGFR's 1/1 also qualifies. Set floor above 1.0 to
    # eliminate both — sanity that the threshold is honored.
    errors = detect_systematic_errors(
        report, cases, bypass_risk_threshold=0.5, severity_floor=1.01
    )
    assert errors == []


# ── propose_corrective_rules ─────────────────────────────────────────────────


def _err(
    error_kind: str = "missed_bypass_downstream_feedback",
    affected: tuple[str, ...] = ("TVS-002",),
    severity: float = 0.9,
) -> SystematicError:
    return SystematicError(
        error_kind=error_kind,
        affected_case_ids=affected,
        severity=severity,
        description=f"Swarm under-scored bypass on {len(affected)} case(s)",
    )


def test_proposer_emits_no_rules_for_empty_errors():
    """No errors → no LLM call → []."""
    llm = FakeLLMClient(responses=[])
    rules = propose_corrective_rules([], build_synthetic_targets(), llm)
    assert rules == []
    assert llm.calls == []  # no wasted LLM calls


def test_proposer_emits_rule_for_missed_bypass():
    """Happy path: one error + scripted JSON → one CANDIDATE rule."""
    llm = FakeLLMClient(responses=[_GOOD_RULE_JSON])
    rules = propose_corrective_rules(
        [_err()], build_synthetic_targets(), llm
    )
    assert len(rules) == 1
    rule = rules[0]
    assert rule.status is RuleStatus.CANDIDATE
    assert rule.trigger.semantic_predicate.startswith("kinase target")
    # No prior-auth ontology fields populated on a cross-agent rule.
    assert rule.trigger.cpt_families == []
    assert rule.trigger.icd10_chapters == []
    assert rule.trigger.payers == []
    assert rule.action.startswith("Raise bypass_risk")
    assert rule.polarity == "requires_check"
    # Provenance stamps the cross-agent origin.
    assert rule.evidence.source_policy_citation == (
        "cross_agent:missed_bypass_downstream_feedback:n=1"
    )
    assert rule.evidence.proposer_run_id  # auto-stamped
    assert rule.evidence.proposer_model == llm.model_version


def test_proposer_emits_two_rules_from_three_errors():
    """Three errors with descending severity, LLM returns two rules per
    call — surviving rule count is errors_above_floor * rules_per_call.
    """
    errors = [
        _err(error_kind="missed_bypass_downstream_feedback", severity=0.9),
        _err(
            error_kind="missed_bypass_alternative_pathway",
            affected=("TVS-003",),
            severity=0.75,
        ),
        _err(error_kind="missed_efficacy", severity=0.1),  # below floor
    ]
    llm = FakeLLMClient(responses=[_TWO_RULES_JSON, _TWO_RULES_JSON])
    rules = propose_corrective_rules(
        errors, build_synthetic_targets(), llm, severity_floor=0.5
    )
    # 2 errors above floor × 2 rules each = 4. The below-floor error
    # must NOT contribute (else we wasted an LLM call on it).
    assert len(rules) == 4
    assert len(llm.calls) == 2  # third error skipped pre-LLM
    assert all(r.status is RuleStatus.CANDIDATE for r in rules)


def test_proposer_rejects_overlong_action():
    """40-word action exceeds ACTION_MAX_WORDS=25 → dropped."""
    long_action = " ".join(["raise"] * 40)
    llm = FakeLLMClient(
        responses=[
            json.dumps(
                {
                    "rules": [
                        {
                            "semantic_predicate": "kinase with paralog count >= 2",
                            "action": long_action,
                            "rationale": "ignored",
                            "polarity": "requires_check",
                        }
                    ]
                }
            )
        ]
    )
    rules = propose_corrective_rules(
        [_err()], build_synthetic_targets(), llm
    )
    assert rules == []


def test_proposer_rejects_patient_identifier_leak():
    """LLM tries to regurgitate a patient identifier → dropped."""
    llm = FakeLLMClient(
        responses=[
            json.dumps(
                {
                    "rules": [
                        {
                            "semantic_predicate": "kinase target",
                            "action": "Flag patient John's tumour for bypass review.",
                            "rationale": "ignored",
                            "polarity": "requires_check",
                        }
                    ]
                }
            )
        ]
    )
    rules = propose_corrective_rules(
        [_err()], build_synthetic_targets(), llm
    )
    assert rules == []


def test_proposer_rejects_empty_predicate_or_action():
    """Empty semantic_predicate or empty action → dropped."""
    llm = FakeLLMClient(
        responses=[
            json.dumps(
                {
                    "rules": [
                        {  # empty predicate
                            "semantic_predicate": "",
                            "action": "Raise bypass_risk by 0.3.",
                            "rationale": "",
                            "polarity": "requires_check",
                        },
                        {  # empty action
                            "semantic_predicate": "kinase target",
                            "action": "",
                            "rationale": "",
                            "polarity": "requires_check",
                        },
                    ]
                }
            )
        ]
    )
    rules = propose_corrective_rules(
        [_err()], build_synthetic_targets(), llm
    )
    assert rules == []


def test_proposer_filters_customer_tagged_candidates():
    """LLM tries to scope a corrective rule to a tenant — REJECT.

    Cross-agent corrective rules are UNIVERSAL by construction;
    ``LayerPolicy.validate_rule(rule, customer_tag=None)`` would block
    the write downstream, but the proposer drops the candidate at the
    JSON layer first (defense in depth).
    """
    llm = FakeLLMClient(
        responses=[
            json.dumps(
                {
                    "rules": [
                        {
                            "semantic_predicate": "kinase target",
                            "action": "Raise bypass_risk by 0.3.",
                            "rationale": "ignored",
                            "polarity": "requires_check",
                            "customer_tag": "tenant_recursion",
                        }
                    ]
                }
            )
        ]
    )
    rules = propose_corrective_rules(
        [_err()], build_synthetic_targets(), llm
    )
    assert rules == []


def test_proposer_supporting_case_ids_come_from_affected():
    """Round-trip: error.affected_case_ids → rule.evidence.supporting_case_ids."""
    err = _err(affected=("TVS-002", "TVS-003"))
    llm = FakeLLMClient(responses=[_GOOD_RULE_JSON])
    rules = propose_corrective_rules(
        [err], build_synthetic_targets(), llm
    )
    assert len(rules) == 1
    assert rules[0].evidence.supporting_case_ids == ["TVS-002", "TVS-003"]
    assert rules[0].evidence.source_policy_citation.endswith(":n=2")


def test_proposer_skips_below_severity_floor():
    """Severity 0.2 with floor 0.5 → no LLM call, no rule."""
    err = _err(severity=0.2)
    llm = FakeLLMClient(responses=[_GOOD_RULE_JSON])
    rules = propose_corrective_rules(
        [err], build_synthetic_targets(), llm, severity_floor=0.5
    )
    assert rules == []
    # Crucial: the floor short-circuits BEFORE the LLM call.
    assert llm.calls == []


def test_proposer_repair_recovers_from_malformed_json():
    """First LLM response is unparseable; second (repair) call succeeds."""
    llm = FakeLLMClient(responses=["not json at all", _GOOD_RULE_JSON])
    rules = propose_corrective_rules(
        [_err()], build_synthetic_targets(), llm
    )
    assert len(rules) == 1
    assert len(llm.calls) == 2  # primary + repair
    # The repair user message must differ from the primary one.
    assert llm.calls[0][1] != llm.calls[1][1]


def test_proposer_permanent_parse_failure_returns_empty():
    """Both primary and repair return junk → that error contributes []."""
    llm = FakeLLMClient(responses=["nope", "still nope"])
    rules = propose_corrective_rules(
        [_err()], build_synthetic_targets(), llm
    )
    assert rules == []
    assert len(llm.calls) == 2  # tried primary + repair, both failed


def test_proposer_zero_rules_from_llm_returns_empty():
    """LLM emits {"rules": []} (no generalisable pattern) → []."""
    llm = FakeLLMClient(responses=[json.dumps({"rules": []})])
    rules = propose_corrective_rules(
        [_err()], build_synthetic_targets(), llm
    )
    assert rules == []


def test_proposer_does_not_leak_internal_evidence_into_prompt():
    """The prompt builder must filter InternalEvidence.readouts — verify
    no readouts key sneaks into the user message even if a case
    carries one.

    Synthetic cases have no InternalEvidence, so this is a defensive
    check: we assert the user message does not contain the literal
    word ``readouts`` (the attribute name on InternalEvidence) nor
    any tenant-tag string.
    """
    llm = FakeLLMClient(responses=[_GOOD_RULE_JSON])
    propose_corrective_rules([_err()], build_synthetic_targets(), llm)
    assert llm.calls, "expected one LLM call"
    _system, user = llm.calls[0]
    assert "readouts" not in user
    assert "customer_tag" not in user


def test_proposer_prompt_includes_swarm_signals_seen():
    """When memos are passed, ``bypass_signals_seen`` is surfaced in the
    user message so the LLM can pinpoint signals the swarm saw but
    underweighted.
    """
    report = _swarm_that_misses_kras_and_egfr_bypass()
    llm = FakeLLMClient(responses=[_GOOD_RULE_JSON])
    propose_corrective_rules(
        [_err(affected=("TVS-002",))],
        build_synthetic_targets(),
        llm,
        memos=report.memos,
    )
    _system, user = llm.calls[0]
    assert "signals_seen_but_underweighted" in user
    assert "downstream_feedback" in user


def test_proposer_skips_fallback_memos():
    """A memo with parse_warnings ``memo_parse_error:`` is the fallback
    emitted on per-agent LLM failure. The prompt builder must skip it
    so a single bad LLM call doesn't drive rule extraction.
    """
    fallback = TargetMemo(
        case_id="TVS-002",
        gene_symbol="KRAS",
        priority_score=0.0,
        bypass_risk_score=0.0,
        predicted_bypass=BypassMechanism.UNKNOWN,
        bypass_signals_seen=["downstream_feedback"],
        parse_warnings=["memo_parse_error: bad JSON shape"],
    )
    llm = FakeLLMClient(responses=[_GOOD_RULE_JSON])
    propose_corrective_rules(
        [_err(affected=("TVS-002",))],
        build_synthetic_targets(),
        llm,
        memos=[fallback],
    )
    _system, user = llm.calls[0]
    # Fallback's signal must NOT appear in the prompt — it was skipped.
    assert "signals_seen_but_underweighted" not in user


def test_proposer_handles_llm_exception_silently():
    """If the LLM client raises, propose_corrective_rules returns []
    without bubbling the exception (analyzer must never crash a swarm
    run).
    """

    class _BrokenLLM:
        model_version = "broken-v0"

        def complete(self, *, system, user, max_tokens=2048, seed=0):
            raise RuntimeError("upstream LLM exploded")

    rules = propose_corrective_rules(
        [_err()], build_synthetic_targets(), _BrokenLLM()
    )
    assert rules == []


# ── run_cross_agent_analysis (end-to-end orchestration) ──────────────────────


def test_run_cross_agent_analysis_returns_analysis_record():
    """Smoke test: with a default FakeLLMClient (returns ``"{}"``), the
    analyzer detects errors but produces no rules — the LLM didn't
    emit a ``rules`` key. The record itself is well-formed.
    """
    cases = build_synthetic_targets()
    report = _swarm_that_misses_kras_and_egfr_bypass()
    analysis = run_cross_agent_analysis(
        report, cases, FakeLLMClient(), bypass_risk_threshold=0.5, severity_floor=0.5
    )
    assert analysis.systematic_errors  # at least one error detected
    # FakeLLMClient default_text="{}" parses but has no "rules" key →
    # zero candidate rules, but the orchestrator still returns a record.
    assert analysis.candidate_rules == []


def test_run_cross_agent_analysis_returns_candidate_rules_now():
    """Wired path: scripted LLM returns one rule per detected error →
    ``analysis.candidate_rules`` is non-empty.
    """
    cases = build_synthetic_targets()
    report = _swarm_that_misses_kras_and_egfr_bypass()
    # Up to 4 detected errors with default floor; supply enough scripted
    # responses that every error gets a parseable JSON.
    llm = FakeLLMClient(responses=[_GOOD_RULE_JSON] * 8)
    analysis = run_cross_agent_analysis(
        report,
        cases,
        llm,
        bypass_risk_threshold=0.5,
        severity_floor=0.5,
    )
    assert analysis.systematic_errors
    assert len(analysis.candidate_rules) >= 1
    assert all(
        r.status is RuleStatus.CANDIDATE for r in analysis.candidate_rules
    )
    # Each rule's provenance points at the cross-agent origin.
    for rule in analysis.candidate_rules:
        assert rule.evidence.source_policy_citation.startswith("cross_agent:")
