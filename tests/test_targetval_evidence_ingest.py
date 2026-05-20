"""Tests for the EvidencePipeline orchestrator + customer_csv helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from medreason.targetval.case import DiseaseContext, TargetID
from medreason.targetval.evidence import (
    GeneticsEvidence,
    InternalEvidence,
    KnockoutDependenceEvidence,
    PriorTrialEvidence,
)
from medreason.targetval.evidence_ingest import (
    CachedTopologyFetcher,
    EvidencePipeline,
    FakeGeneticsFetcher,
    FakeKnockoutFetcher,
    FakeTrialsFetcher,
    build_default_pipeline,
    build_evidence_bundle,
)
from medreason.targetval.topology import PathwayCache

from medreason_bench.targetval.customer_csv import parse_jsonl


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _braf_cache() -> PathwayCache:
    """A minimal cache that knows about BRAF — used by several tests."""
    return PathwayCache(
        paralogs_by_gene={"BRAF": ["ARAF", "RAF1"]},
        family_by_gene={"BRAF": "RAF"},
        feedback_loops_by_gene={"BRAF": ["RAS-GTP buildup → CRAF dimerisation"]},
        downstream_redundancy={"BRAF": 0.6},
        upstream_nodes_by_gene={"BRAF": ["RAS"]},
        downstream_nodes_by_gene={"BRAF": ["MEK1", "MEK2"]},
        cache_version="v0.1-braf-fixture",
    )


def _braf_target() -> TargetID:
    return TargetID(gene_symbol="BRAF")


def _melanoma() -> DiseaseContext:
    return DiseaseContext(disease_label="melanoma")


# ── EvidencePipeline ──────────────────────────────────────────────────────────


def test_evidence_pipeline_assembles_from_fakes():
    cache = _braf_cache()
    pipeline = build_default_pipeline(
        cache,
        genetics_table={
            "BRAF": GeneticsEvidence(overall_score=0.91, citations=["OT:BRAF"])
        },
        knockout_table={"BRAF": KnockoutDependenceEvidence(mean_dependency_score=-0.7)},
        trials_table={"BRAF": PriorTrialEvidence(n_trials_prior=12, n_approvals=2)},
    )
    bundle = pipeline.fetch(_braf_target(), _melanoma())

    assert bundle.genetics.overall_score == pytest.approx(0.91)
    assert bundle.knockout.mean_dependency_score == pytest.approx(-0.7)
    assert bundle.prior_trials.n_trials_prior == 12
    assert bundle.topology.paralogs == ["ARAF", "RAF1"]
    assert bundle.topology.paralog_count == 2
    assert bundle.topology.downstream_redundancy_index == pytest.approx(0.6)
    assert bundle.topology.upstream_node_count == 1
    assert bundle.topology.downstream_node_count == 2
    # cache_version propagates into the bundle so downstream consumers
    # can reproduce the topology snapshot that fed the run.
    assert bundle.topology.reference_pathway == "v0.1-braf-fixture"
    # No internal evidence wired → default empty.
    assert bundle.internal.customer_tag is None


def test_build_default_pipeline_with_empty_cache_returns_empty_topology():
    cache = PathwayCache(cache_version="v0.1-empty")
    pipeline = build_default_pipeline(cache)
    bundle = pipeline.fetch(_braf_target(), _melanoma())

    # The cache knows nothing about BRAF → paralog list is empty, count
    # is None (NOT zero — missing means missing).
    assert bundle.topology.paralogs == []
    assert bundle.topology.paralog_count is None
    assert bundle.topology.downstream_redundancy_index is None
    assert bundle.topology.upstream_node_count is None
    assert bundle.topology.downstream_node_count is None
    # cache_version still stamps through even for an empty cache.
    assert bundle.topology.reference_pathway == "v0.1-empty"


def test_evidence_pipeline_accepts_internal_evidence_override():
    pipeline = build_default_pipeline(_braf_cache())
    internal = InternalEvidence(
        customer_tag="recursion",
        readouts={"phenoprint_score": 0.42},
    )
    bundle = pipeline.fetch(_braf_target(), _melanoma(), internal=internal)
    assert bundle.internal.customer_tag == "recursion"
    assert bundle.internal.readouts == {"phenoprint_score": 0.42}
    assert bundle.has_internal_data() is True


def test_evidence_pipeline_with_no_fetchers_returns_empty_bundle():
    pipeline = EvidencePipeline()
    bundle = pipeline.fetch(_braf_target(), _melanoma())
    assert bundle.genetics.overall_score is None
    assert bundle.knockout.mean_dependency_score is None
    assert bundle.topology.paralogs == []
    assert bundle.topology.reference_pathway is None
    assert bundle.prior_trials.n_trials_prior == 0


def test_build_evidence_bundle_free_function_still_works():
    """The pre-v0.2 free-function entry-point still produces the same shape
    as the pipeline class. Keeps backward compatibility for callers that
    don't want to hold a pipeline instance.
    """
    cache = _braf_cache()
    bundle_fn = build_evidence_bundle(
        _braf_target(),
        _melanoma(),
        topology=CachedTopologyFetcher(cache),
        genetics=FakeGeneticsFetcher({"BRAF": GeneticsEvidence(overall_score=0.5)}),
        knockout=FakeKnockoutFetcher(),
        trials=FakeTrialsFetcher(),
    )
    pipeline = build_default_pipeline(
        cache,
        genetics_table={"BRAF": GeneticsEvidence(overall_score=0.5)},
    )
    bundle_pipe = pipeline.fetch(_braf_target(), _melanoma())
    assert bundle_fn.topology.paralogs == bundle_pipe.topology.paralogs
    assert bundle_fn.topology.reference_pathway == bundle_pipe.topology.reference_pathway
    assert bundle_fn.genetics.overall_score == bundle_pipe.genetics.overall_score


# ── customer_csv.parse_jsonl ──────────────────────────────────────────────────


def test_parse_jsonl_roundtrip(tmp_path: Path):
    rows = [
        {"case_id": "TGT_001", "gene_symbol": "BRAF", "disease_label": "melanoma"},
        {"case_id": "TGT_002", "gene_symbol": "KRAS", "disease_label": "colorectal"},
    ]
    p = tmp_path / "targets.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    out = parse_jsonl(p)
    assert out == rows


def test_parse_jsonl_skips_blank_lines(tmp_path: Path):
    p = tmp_path / "targets.jsonl"
    p.write_text(
        '{"case_id": "A"}\n\n   \n{"case_id": "B"}\n', encoding="utf-8"
    )
    assert parse_jsonl(p) == [{"case_id": "A"}, {"case_id": "B"}]


def test_parse_jsonl_rejects_non_object_line(tmp_path: Path):
    p = tmp_path / "targets.jsonl"
    p.write_text('{"case_id": "A"}\n[1, 2, 3]\n', encoding="utf-8")
    with pytest.raises(ValueError, match="expected JSON object"):
        parse_jsonl(p)


def test_parse_jsonl_reports_offending_line_number(tmp_path: Path):
    p = tmp_path / "targets.jsonl"
    p.write_text(
        '{"case_id": "A"}\n{"case_id": "B"}\nnot-json\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match=":3:"):
        parse_jsonl(p)
