"""Tests for the cross-agent metacognitive analyzer.

These tests verify the detect_systematic_errors logic against a
hand-crafted SwarmReport. They do NOT test rule generation
(propose_corrective_rules is a stub).
"""

from __future__ import annotations

from medreason.llm.base import FakeLLMClient
from medreason.targetval.case import (
    BypassMechanism,
    GroundTruthOutcome,
)
from medreason.targetval.cross_agent_analyzer import (
    detect_systematic_errors,
    propose_corrective_rules,
    run_cross_agent_analysis,
)
from medreason.targetval.swarm import SwarmReport, TargetMemo

from medreason_bench.targetval.synthetic import build_synthetic_targets


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
            ),
            TargetMemo(
                case_id="TVS-003",
                gene_symbol="EGFR",
                priority_score=0.4,
                bypass_risk_score=0.3,  # missed
                predicted_bypass=BypassMechanism.NO_BYPASS_KNOWN,
            ),
        ],
        ranking=["BRAF", "KRAS", "EGFR"],
    )


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


def test_propose_corrective_rules_skeleton_returns_empty():
    cases = build_synthetic_targets()
    report = _swarm_that_misses_kras_and_egfr_bypass()
    errors = detect_systematic_errors(report, cases)
    rules = propose_corrective_rules(errors, cases, FakeLLMClient())
    # Skeleton: returns empty until the LLM-driven proposer is wired.
    assert rules == []


def test_run_cross_agent_analysis_returns_analysis_record():
    cases = build_synthetic_targets()
    report = _swarm_that_misses_kras_and_egfr_bypass()
    analysis = run_cross_agent_analysis(
        report, cases, FakeLLMClient(), bypass_risk_threshold=0.5, severity_floor=0.5
    )
    assert analysis.systematic_errors  # at least one error detected
    assert analysis.candidate_rules == []  # skeleton
