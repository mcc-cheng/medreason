"""End-to-end integration test for the targetval pipeline.

This is the validation surface for the Phase 2 swarm. It exercises every
builder's work in sequence — without touching the network or a real LLM:

    build_mapk_retro_seed(version="v0.2")        # mapk-curator
        → SwarmRunner(pathway_cache=cache)        # swarm-llm-builder
            → memos with bypass_risk under-scored # (FakeLLMClient)
        → run_cross_agent_analysis(...)            # proposer-builder
            → SystematicErrors detected            # cross_agent_analyzer
            → CANDIDATE rules emitted              # proposer-builder
        → LayerRouter.ingest_rule(UNIVERSAL)       # layer-store-builder
        → LayerRouter.retrieve_for_case(...)       # layer-store-builder

The retrieval at the end is the load-bearing assertion: a rule emitted
by the cross-agent analyzer on retrospective data round-trips through
the Universal-layer SQLite store and resurfaces on the next case query.
"""

from __future__ import annotations

import json
from pathlib import Path

from medreason.llm.base import FakeLLMClient
from medreason.ontology.rule import RuleStatus
from medreason.targetval.case import BypassMechanism
from medreason.targetval.cross_agent_analyzer import run_cross_agent_analysis
from medreason.targetval.layers import Layer, LayerRouter, LayerStorePaths
from medreason.targetval.swarm import SwarmRunner
from medreason.targetval.topology import PathwayCache

from medreason_bench.targetval.mapk_retro import build_mapk_retro_seed


# ── Test helpers ─────────────────────────────────────────────────────────────


def _mapk_retro_cache() -> PathwayCache:
    """Hand-built PathwayCache covering the v0.2 retro genes.

    Paralog families and feedback loops mirror the curated entries in
    ``mapk_retro_data.py`` so the cache and the retro cases agree on
    topology. Built by hand per the dispatch constraint: NO Reactome
    flat-file load, NO network. Keep this fixture close to the entries
    it serves — the moment the data table grows, this cache grows with
    it.
    """
    return PathwayCache(
        paralogs_by_gene={
            # RAF family
            "BRAF": ["ARAF", "RAF1"],
            "ARAF": ["BRAF", "RAF1"],
            "RAF1": ["BRAF", "ARAF"],
            # MEK / ERK
            "MAP2K1": ["MAP2K2"],
            "MAP2K2": ["MAP2K1"],
            "MAPK1": ["MAPK3"],
            "MAPK3": ["MAPK1"],
            # RAS GTPases
            "KRAS": ["NRAS", "HRAS"],
            "NRAS": ["KRAS", "HRAS"],
            "HRAS": ["KRAS", "NRAS"],
            # RTKs
            "EGFR": ["ERBB2", "ERBB3", "ERBB4"],
            "MET": ["MST1R"],
            "ALK": ["LTK"],
            # Other v0.2 targets
            "PTPN11": ["PTPN6"],
            "FGFR2": ["FGFR1", "FGFR3", "FGFR4"],
            "CDK4": ["CDK6"],
            "PIK3CA": ["PIK3CB", "PIK3CD", "PIK3CG"],
            "WEE1": ["PKMYT1"],
            "AXL": ["MERTK", "TYRO3"],
            "MDM2": ["MDM4"],
        },
        family_by_gene={
            "BRAF": "RAF_kinase",
            "ARAF": "RAF_kinase",
            "RAF1": "RAF_kinase",
            "MAP2K1": "MEK",
            "MAP2K2": "MEK",
            "MAPK1": "ERK",
            "MAPK3": "ERK",
            "KRAS": "RAS_GTPase",
            "NRAS": "RAS_GTPase",
            "HRAS": "RAS_GTPase",
            "EGFR": "RTK",
            "MET": "RTK",
            "ALK": "RTK",
            "PTPN11": "phosphatase",
            "FGFR2": "RTK",
            "CDK4": "CDK",
            "PIK3CA": "PI3K",
            "WEE1": "checkpoint_kinase",
            "AXL": "RTK",
            "MDM2": "E3_ligase",
        },
        feedback_loops_by_gene={
            "BRAF": [
                "MEK-ERK rebound after BRAF-i monotherapy",
                "RAS-GTP reactivation",
            ],
            "MAP2K1": ["ERK negative feedback to RAF"],
            "MAPK1": ["ERK auto-feedback to RAF", "DUSP6 suppression"],
            "KRAS": [
                "RTK reactivation (EGFR, AXL, MET)",
                "wild-type RAS amplification",
            ],
            "EGFR": [
                "MET amplification bypass",
                "AXL upregulation",
            ],
            "MET": ["EGFR cross-activation"],
            "ALK": ["bypass via EGFR / KIT activation"],
            "PTPN11": [
                "RAS-GTP feedback after KRAS-G12C inhibition",
            ],
            "FGFR2": ["gatekeeper V564F resistance"],
            "CDK4": [
                "RB1 loss",
                "cyclin E1 amplification (CCNE1) bypass",
            ],
            "PIK3CA": [
                "AKT-mTOR rebound",
                "ERK pathway compensation",
            ],
            "WEE1": ["ATR-CHK1 backup checkpoint"],
            "AXL": ["TAM family compensation"],
            "MDM2": [
                "MDM2 transcriptional auto-induction by p53",
                "MDM4 compensatory binding",
            ],
            "RAF1": [
                "RAF dimer paradox in WT-RAF cells",
                "MEK-ERK feedback",
            ],
        },
        downstream_redundancy={
            "BRAF": 0.4,
            "MAP2K1": 0.5,
            "MAPK1": 0.45,
            "KRAS": 0.45,
            "NRAS": 0.55,
            "HRAS": 0.5,
            "EGFR": 0.55,
            "MET": 0.4,
            "ALK": 0.35,
            "PTPN11": 0.35,
            "FGFR2": 0.5,
            "CDK4": 0.4,
            "PIK3CA": 0.6,
            "WEE1": 0.35,
            "AXL": 0.55,
            "MDM2": 0.45,
            "RAF1": 0.55,
        },
        cache_version="mapk_retro_v0.2_e2e",
    )


# Memo JSON that under-scores bypass: priority high, bypass_risk low.
# This is what the swarm's FakeLLMClient returns for every case so
# ``detect_systematic_errors`` sees a systematic miss across many cases.
_UNDERSCORE_BYPASS_MEMO_JSON = json.dumps(
    {
        "priority_score": 0.7,
        "bypass_risk_score": 0.15,
        "predicted_bypass": "no_bypass_known",
        "supporting_evidence": [
            "Strong genetic association and clean knockout signal."
        ],
        "weakening_evidence": [],
        "proposed_experiments": [
            "Confirm dependency in patient-derived organoids."
        ],
        "rationale": "Target appears clean; recommend Phase 2 prioritisation.",
    }
)


# Cross-agent proposer canned JSON. The shape is the contract
# ``proposer-builder`` parses; matches the shared template at the top of
# ``tests/test_targetval_cross_agent.py``.
_PARALOG_FEEDBACK_RULE_JSON = json.dumps(
    {
        "rules": [
            {
                "semantic_predicate": (
                    "kinase target with paralog_count >= 2 and a known "
                    "downstream feedback loop"
                ),
                "action": (
                    "Raise bypass_risk by at least 0.3 and require combo "
                    "strategy."
                ),
                "rationale": (
                    "Paralog redundancy plus documented feedback drives "
                    "systematic bypass across MAPK retrospectives."
                ),
                "polarity": "requires_check",
            }
        ]
    }
)


# ── Tests ────────────────────────────────────────────────────────────────────


def test_mapk_retro_v0_2_size():
    """The curated v0.2 fixture must hit the dispatch's ≥20 case bar with
    every entry carrying a known bypass label (no UNKNOWN — that's an
    inference-time-only value)."""
    cases = build_mapk_retro_seed(version="v0.2")
    assert len(cases) >= 20
    assert all(
        c.ground_truth_bypass is not BypassMechanism.UNKNOWN for c in cases
    )
    # Every case is Universal-safe: empty internal sub-bundle. The
    # leak-guard cares about this — drift would silently break the moat.
    assert all(not c.evidence.has_internal_data() for c in cases)


def test_mapk_retro_v0_1_legacy_still_works():
    """Legacy 3-entry seed remains addressable for back-compat."""
    cases = build_mapk_retro_seed(version="v0.1")
    assert len(cases) == 3
    assert {c.case_id for c in cases} == {"MAPK-001", "MAPK-002", "MAPK-003"}


def test_mapk_retro_v0_2_default_matches_explicit_version():
    """No-arg call is the v0.2 path — keeps __main__.py wiring honest."""
    default = build_mapk_retro_seed()
    explicit = build_mapk_retro_seed(version="v0.2")
    assert len(default) == len(explicit)
    assert [c.case_id for c in default] == [c.case_id for c in explicit]


def test_mapk_retro_unknown_version_raises():
    """Defensive: callers passing a typo'd version get a clear error
    rather than a silent empty list."""
    import pytest

    with pytest.raises(ValueError, match="Unknown mapk_retro version"):
        build_mapk_retro_seed(version="v9.9")


def test_full_pipeline_mapk_retro_with_fakellm(tmp_path: Path):
    """End-to-end: 22 retro cases → swarm under-scores bypass → cross-
    agent analyzer detects systematic miss → proposes a CANDIDATE
    UNIVERSAL rule → router ingests it → next retrieve_for_case returns
    it.

    No real LLM, no network. Real SQLite-backed RuleStore via tmp_path.
    """
    # 1) Build the v0.2 retro fixture.
    cases = build_mapk_retro_seed(version="v0.2")
    assert len(cases) >= 20

    # 2) PathwayCache covering every gene in the fixture.
    cache = _mapk_retro_cache()
    for c in cases:
        # Every gene the swarm will look up must be in the cache. If
        # this assert fires, the fixture grew but the cache didn't.
        assert c.target.gene_symbol in cache.family_by_gene

    # 3) FakeLLMClient for the swarm: every memo under-scores bypass.
    #    ``default_text`` makes this work regardless of case count.
    swarm_llm = FakeLLMClient(default_text=_UNDERSCORE_BYPASS_MEMO_JSON)

    # 4) LayerRouter on real tmp_path SQLite — universal layer only;
    #    no disease store wired (retrieve falls back to universal alone).
    router = LayerRouter(
        LayerStorePaths(
            universal=tmp_path / "universal.db",
            disease={},
            campaign={},
        )
    )

    # 5) Run the swarm. ``max_workers=1`` keeps the test deterministic
    #    and well under the 10 s budget.
    runner = SwarmRunner(
        swarm_llm, router, max_workers=1, pathway_cache=cache
    )
    report = runner.run(cases, campaign_id="mapk_retro_v0.2", seed=11)
    assert report.n_targets == len(cases)
    # Each memo parsed cleanly to the under-scored shape.
    assert all(m.bypass_risk_score < 0.5 for m in report.memos)
    assert all(m.priority_score > 0.0 for m in report.memos)

    # 6) Cross-agent analysis. The proposer LLM uses ``default_text`` so
    #    it can answer any number of SystematicErrors above the floor.
    proposer_llm = FakeLLMClient(default_text=_PARALOG_FEEDBACK_RULE_JSON)
    analysis = run_cross_agent_analysis(
        report,
        cases,
        proposer_llm,
        bypass_risk_threshold=0.5,
        severity_floor=0.3,
        proposer_run_id="e2e_test_run",
        seed=11,
    )
    assert len(analysis.systematic_errors) >= 1
    assert len(analysis.candidate_rules) >= 1
    # Every emitted rule is CANDIDATE and Universal-policy-clean.
    for rule in analysis.candidate_rules:
        assert rule.status is RuleStatus.CANDIDATE
        assert rule.evidence.source_policy_citation.startswith("cross_agent:")
        assert rule.evidence.proposer_run_id == "e2e_test_run"

    # 7) Ingest the corrective rule(s) into the UNIVERSAL layer. The
    #    LayerPolicy gate runs again here — defense in depth.
    for rule in analysis.candidate_rules:
        rule.status = RuleStatus.ACTIVE  # ACTIVE so retrieve_for_case surfaces it
        router.ingest_rule(
            rule, source_layer=Layer.UNIVERSAL, customer_tag=None
        )

    # 8) Retrieve for an arbitrary MAPK case scope; the universal rule(s)
    #    surface regardless of scope (that's the whole point of Universal).
    retrieved = router.retrieve_for_case(
        case_disease_scope="oncology_mapk",
        customer_tag=None,
        top_k=10,
    )
    retrieved_ids = {r.rule_id for r in retrieved}
    candidate_ids = {r.rule_id for r in analysis.candidate_rules}
    assert candidate_ids & retrieved_ids, (
        f"None of the ingested universal rules round-tripped. "
        f"Ingested={candidate_ids}, retrieved={retrieved_ids}"
    )


def test_full_pipeline_e2e_runs_under_budget(tmp_path: Path):
    """The end-to-end pipeline must complete well under 10 s — that's
    the dispatch's hard budget. Track it explicitly so a future
    regression (e.g., accidental N^2 work in the swarm) shows up as
    a test failure rather than a slow CI run."""
    import time

    cases = build_mapk_retro_seed(version="v0.2")
    cache = _mapk_retro_cache()
    swarm_llm = FakeLLMClient(default_text=_UNDERSCORE_BYPASS_MEMO_JSON)
    router = LayerRouter(
        LayerStorePaths(
            universal=tmp_path / "universal.db",
            disease={},
            campaign={},
        )
    )
    runner = SwarmRunner(
        swarm_llm, router, max_workers=1, pathway_cache=cache
    )

    start = time.monotonic()
    report = runner.run(cases, campaign_id="mapk_retro_v0.2", seed=11)
    proposer_llm = FakeLLMClient(default_text=_PARALOG_FEEDBACK_RULE_JSON)
    run_cross_agent_analysis(
        report, cases, proposer_llm, severity_floor=0.3, seed=11
    )
    elapsed = time.monotonic() - start

    # Generous ceiling — empirically this runs in well under a second on
    # a laptop. The 5 s ceiling is the canary for accidental regression.
    assert elapsed < 5.0, (
        f"e2e pipeline took {elapsed:.2f}s — investigate regression"
    )
