"""Tests for the customer-shaped ingest path."""

from __future__ import annotations

from medreason.targetval.case import BypassMechanism, GroundTruthOutcome, Modality
from medreason_bench.targetval.recursion_ingest import load_customer_targets


def test_ingest_minimal_target_row():
    rows = [
        {
            "case_id": "TGT_001",
            "gene_symbol": "BRAF",
            "disease_label": "melanoma",
            "modality": "small_molecule",
            "readouts_json": '{"phenoprint_score": 0.42}',
        }
    ]
    cases = load_customer_targets(rows, customer_tag="recursion")
    assert len(cases) == 1
    c = cases[0]
    assert c.target.gene_symbol == "BRAF"
    assert c.modality is Modality.SMALL_MOLECULE
    assert c.evidence.internal.customer_tag == "recursion"
    assert c.evidence.internal.readouts == {"phenoprint_score": 0.42}
    assert c.is_retrospective() is False  # no outcome provided


def test_ingest_with_outcomes_marks_retrospective():
    targets = [
        {
            "case_id": "TGT_002",
            "gene_symbol": "KRAS",
            "disease_label": "colorectal",
            "modality": "small_molecule",
        }
    ]
    outcomes = [
        {
            "case_id": "TGT_002",
            "outcome_label": "phase2_efficacy_no",
            "bypass_label": "downstream_feedback",
            "notes": "EGFR feedback documented",
        }
    ]
    cases = load_customer_targets(
        targets, customer_tag="recursion", outcomes_iter=outcomes
    )
    assert len(cases) == 1
    c = cases[0]
    assert c.ground_truth_outcome is GroundTruthOutcome.PHASE2_EFFICACY_NO
    assert c.ground_truth_bypass is BypassMechanism.DOWNSTREAM_FEEDBACK
    assert c.is_retrospective() is True


def test_ingest_unknown_modality_falls_back():
    rows = [
        {
            "case_id": "TGT_003",
            "gene_symbol": "EGFR",
            "disease_label": "glioblastoma",
            "modality": "weird_modality_we_dont_know",
        }
    ]
    cases = load_customer_targets(rows, customer_tag="x")
    assert cases[0].modality is Modality.UNKNOWN


def test_ingest_handles_unparseable_readouts_json():
    rows = [
        {
            "case_id": "TGT_004",
            "gene_symbol": "MEK1",
            "disease_label": "nf1",
            "readouts_json": "not valid json {{",
        }
    ]
    cases = load_customer_targets(rows, customer_tag="x")
    assert "_unparsed" in cases[0].evidence.internal.readouts


def test_ingest_handles_dict_readouts():
    rows = [
        {
            "case_id": "TGT_005",
            "gene_symbol": "ERK1",
            "disease_label": "x",
            "readouts_json": {"already": "dict", "score": 0.5},
        }
    ]
    cases = load_customer_targets(rows, customer_tag="x")
    assert cases[0].evidence.internal.readouts == {"already": "dict", "score": 0.5}
