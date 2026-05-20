"""Tests for the customer-shaped ingest path."""

from __future__ import annotations

import pytest

from medreason.targetval.case import BypassMechanism, GroundTruthOutcome, Modality
from medreason_bench.targetval.recursion_ingest import (
    IngestError,
    IngestReport,
    load_customer_targets,
)


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
    report = load_customer_targets(rows, customer_tag="recursion")
    assert isinstance(report, IngestReport)
    assert report.n_cases == 1
    assert report.n_rejected == 0
    c = report.cases[0]
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
    report = load_customer_targets(
        targets, customer_tag="recursion", outcomes_iter=outcomes
    )
    assert report.n_cases == 1
    c = report.cases[0]
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
    report = load_customer_targets(rows, customer_tag="x")
    assert report.cases[0].modality is Modality.UNKNOWN


def test_ingest_handles_unparseable_readouts_json():
    rows = [
        {
            "case_id": "TGT_004",
            "gene_symbol": "MEK1",
            "disease_label": "nf1",
            "readouts_json": "not valid json {{",
        }
    ]
    report = load_customer_targets(rows, customer_tag="x")
    assert "_unparsed" in report.cases[0].evidence.internal.readouts


def test_ingest_handles_dict_readouts():
    rows = [
        {
            "case_id": "TGT_005",
            "gene_symbol": "ERK1",
            "disease_label": "x",
            "readouts_json": {"already": "dict", "score": 0.5},
        }
    ]
    report = load_customer_targets(rows, customer_tag="x")
    assert report.cases[0].evidence.internal.readouts == {
        "already": "dict",
        "score": 0.5,
    }


# ── New v0.2 coverage: rejected rows + strict mode ────────────────────────────


def test_ingest_report_buckets_rejected_rows():
    rows = [
        {  # missing case_id → rejected
            "gene_symbol": "BRAF",
            "disease_label": "melanoma",
        },
        {  # good row
            "case_id": "TGT_OK",
            "gene_symbol": "MEK1",
            "disease_label": "nf1",
        },
    ]
    report = load_customer_targets(rows, customer_tag="recursion")
    assert report.n_cases == 1
    assert report.n_rejected == 1
    bad_row, reason = report.rejected[0]
    assert "case_id" in reason
    assert bad_row.get("gene_symbol") == "BRAF"


def test_ingest_strict_raises_on_bad_row():
    rows = [
        {"gene_symbol": "BRAF"},  # missing case_id
    ]
    with pytest.raises(IngestError, match="case_id"):
        load_customer_targets(rows, customer_tag="recursion", strict=True)


def test_ingest_rejects_blank_customer_tag():
    with pytest.raises(IngestError, match="customer_tag"):
        load_customer_targets([], customer_tag="")
