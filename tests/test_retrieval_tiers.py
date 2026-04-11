"""Tests for retrieval tiers — Phase 5 Commit 2.

Covers ontology_lookup (Tier 1), dense_rerank (Tier 2), rerank (Tier 3),
and the composed pipeline.retrieve() function. All tests use in-memory
SQLite stores with the FakeEmbedder; the LLM reranker uses FakeLLMClient.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from medreason.llm import FakeLLMClient
from medreason.ontology import (
    BenchmarkCase,
    CPTFamily,
    Difficulty,
    FacilityType,
    ICD10Chapter,
    Outcome,
    Payer,
    PriorAuthTaskConfig,
    ReasoningRule,
    ReasoningStep,
    RuleEvidence,
    RuleStatus,
    RuleTrigger,
)
from medreason.retrieval import (
    FakeEmbedder,
    RetrievalResult,
    build_rule_checklist,
    dense_rerank,
    lookup_by_trigger,
    rerank,
    retrieve,
)
from medreason.store import RuleStore


def _mk_rule(
    *,
    rule_id: str,
    cpt_families=(CPTFamily.IMAGING_MRI,),
    icd_chapters=(ICD10Chapter.MUSCULOSKELETAL,),
    payers=(Payer.AETNA,),
    action: str = "Require conservative therapy before approval.",
    rationale: str = "Policy requires conservative trial.",
    semantic_predicate: str = "lumbar MRI chronic back pain",
    status: RuleStatus = RuleStatus.ACTIVE,
    success_count: int = 10,
    failure_count: int = 2,
) -> ReasoningRule:
    return ReasoningRule(
        rule_id=rule_id,
        status=status,
        trigger=RuleTrigger(
            cpt_families=list(cpt_families),
            icd10_chapters=list(icd_chapters),
            payers=list(payers),
            semantic_predicate=semantic_predicate,
        ),
        action=action,
        rationale=rationale,
        evidence=RuleEvidence(
            supporting_case_ids=["train_001"],
            source_policy_citation="CMS LCD L1 §C.1",
            proposer_model="test",
        ),
        success_count=success_count,
        failure_count=failure_count,
    )


def _mk_case(
    *,
    case_id: str = "test_001",
    payer: Payer = Payer.AETNA,
    cpt: str = "72148",
    icds: list[str] | None = None,
    notes: str = "lumbar MRI conservative physical therapy chronic back pain",
    facility: FacilityType = FacilityType.OUTPATIENT,
    outcome: Outcome = Outcome.APPROVED,
) -> BenchmarkCase:
    return BenchmarkCase(
        case_id=case_id,
        task_config=PriorAuthTaskConfig(
            payer=payer,
            cpt_code=cpt,
            icd10_codes=icds or ["M54.5"],
            facility_type=facility,
        ),
        clinical_notes=notes,
        policy_excerpt="policy",
        ground_truth_outcome=outcome,
        difficulty=Difficulty.MEDIUM,
    )


@pytest.fixture
def populated_store(sqlite_conn):
    store = RuleStore(sqlite_conn)
    store.put(_mk_rule(rule_id="r_relevant",
                       action="Six weeks physical therapy required."))
    store.put(_mk_rule(rule_id="r_offtopic",
                       cpt_families=(CPTFamily.SURGERY_ORTHO,),
                       action="Cardiac clearance required pre-op."))
    store.put(_mk_rule(rule_id="r_wrong_payer",
                       payers=(Payer.CIGNA,),
                       action="Peer to peer review required."))
    store.put(_mk_rule(rule_id="r_candidate",
                       status=RuleStatus.CANDIDATE,
                       action="Candidate rule not yet promoted."))
    return store


# ── Tier 1: ontology_lookup ─────────────────────────────────────────────────


def test_ontology_lookup_filters_by_payer(populated_store):
    case = _mk_case()
    out = lookup_by_trigger(populated_store, case.task_config)
    ids = [r.rule_id for r in out]
    assert "r_relevant" in ids
    assert "r_wrong_payer" not in ids  # wrong payer
    assert "r_offtopic" not in ids  # wrong CPT family
    assert "r_candidate" not in ids  # not ACTIVE by default


def test_ontology_lookup_include_candidate(populated_store):
    case = _mk_case()
    out = lookup_by_trigger(
        populated_store, case.task_config, include_candidate=True
    )
    ids = [r.rule_id for r in out]
    assert "r_candidate" in ids
    assert "r_relevant" in ids


def test_ontology_lookup_returns_sorted_by_posterior(populated_store):
    # Add two relevant rules with different posteriors
    populated_store.put(_mk_rule(
        rule_id="r_highconf", success_count=50, failure_count=2
    ))
    populated_store.put(_mk_rule(
        rule_id="r_lowconf", success_count=3, failure_count=5
    ))
    case = _mk_case()
    out = lookup_by_trigger(populated_store, case.task_config)
    ids = [r.rule_id for r in out]
    # r_highconf (50/52) should rank above r_lowconf (3/8)
    assert ids.index("r_highconf") < ids.index("r_lowconf")


def test_ontology_lookup_empty_store(sqlite_conn):
    store = RuleStore(sqlite_conn)
    case = _mk_case()
    assert lookup_by_trigger(store, case.task_config) == []


def test_ontology_lookup_respects_max_candidates(populated_store):
    out = lookup_by_trigger(
        populated_store, _mk_case().task_config, max_candidates=1
    )
    assert len(out) == 1


def test_ontology_lookup_wildcard_trigger_matches_any_payer(sqlite_conn):
    store = RuleStore(sqlite_conn)
    store.put(_mk_rule(rule_id="r_wild", payers=()))  # no payer filter
    case = _mk_case(payer=Payer.HUMANA)  # any payer
    out = lookup_by_trigger(store, case.task_config)
    assert [r.rule_id for r in out] == ["r_wild"]


# ── Tier 2: dense_rerank ────────────────────────────────────────────────────


def test_dense_rerank_ranks_by_vocabulary_overlap(populated_store):
    embedder = FakeEmbedder()
    case = _mk_case(notes="lumbar MRI physical therapy chronic back pain")

    r_close = _mk_rule(
        rule_id="r_close",
        semantic_predicate="lumbar MRI chronic back pain",
        action="Require six weeks physical therapy before MRI.",
    )
    r_far = _mk_rule(
        rule_id="r_far",
        semantic_predicate="cardiac angiography contrast",
        action="Require cardiac clearance.",
    )
    candidates = [r_far, r_close]

    scored = dense_rerank(case, candidates, embedder, top_k=2)
    ranked_ids = [rule.rule_id for rule, _ in scored]
    assert ranked_ids[0] == "r_close"  # vocabulary overlap wins
    assert scored[0][1] > scored[1][1]  # higher cosine


def test_dense_rerank_min_similarity_filter(populated_store):
    embedder = FakeEmbedder()
    case = _mk_case(notes="totally unrelated cardiac symptoms")
    r = _mk_rule(
        rule_id="r_far",
        semantic_predicate="lumbar spine",
        action="PT required.",
    )
    out = dense_rerank(case, [r], embedder, min_similarity=0.5)
    assert out == []


def test_dense_rerank_top_k_caps_results():
    embedder = FakeEmbedder()
    case = _mk_case()
    rules = [
        _mk_rule(
            rule_id=f"r_{i}",
            semantic_predicate=f"lumbar MRI back pain {i}",
        )
        for i in range(10)
    ]
    out = dense_rerank(case, rules, embedder, top_k=3)
    assert len(out) == 3


def test_dense_rerank_computes_rule_embedding_on_the_fly():
    """A rule with no cached embedding should still be ranked."""
    embedder = FakeEmbedder()
    rule = _mk_rule(rule_id="r_nocache")
    assert rule.trigger.semantic_embedding is None
    out = dense_rerank(_mk_case(), [rule], embedder)
    assert len(out) == 1
    assert out[0][0].rule_id == "r_nocache"


# ── Tier 3: rerank ──────────────────────────────────────────────────────────


def _rerank_response(entries: list[dict]) -> str:
    return json.dumps({"scores": entries})


def test_rerank_applies_scores_and_filters(populated_store):
    case = _mk_case()
    candidates = [
        _mk_rule(rule_id="r_a", action="rule a"),
        _mk_rule(rule_id="r_b", action="rule b"),
        _mk_rule(rule_id="r_c", action="rule c"),
    ]
    llm = FakeLLMClient(responses=[_rerank_response([
        {"rule_id": "r_a", "applicability": 9, "expected_utility": 8, "conflict_with": []},
        {"rule_id": "r_b", "applicability": 2, "expected_utility": 1, "conflict_with": []},  # filtered
        {"rule_id": "r_c", "applicability": 6, "expected_utility": 5, "conflict_with": []},
    ])])
    out = rerank(case, candidates, llm, top_k=6)
    ids = [r.rule_id for r in out]
    assert "r_a" in ids
    assert "r_c" in ids
    assert "r_b" not in ids  # below min_applicability=4


def test_rerank_final_score_uses_posterior(populated_store):
    case = _mk_case()
    candidates = [
        _mk_rule(rule_id="r_hi_app_low_post",
                 success_count=1, failure_count=9),  # posterior ~0.17
        _mk_rule(rule_id="r_mid_app_high_post",
                 success_count=50, failure_count=2),  # posterior ~0.94
    ]
    llm = FakeLLMClient(responses=[_rerank_response([
        {"rule_id": "r_hi_app_low_post", "applicability": 10, "expected_utility": 10},
        {"rule_id": "r_mid_app_high_post", "applicability": 6, "expected_utility": 6},
    ])])
    out = rerank(case, candidates, llm, top_k=2)
    # 1.0 * 0.17 = 0.17  vs  0.36 * 0.94 = 0.338
    # High posterior wins even with lower applicability.
    assert out[0].rule_id == "r_mid_app_high_post"


def test_rerank_empty_candidates():
    llm = FakeLLMClient(responses=["{}"])
    assert rerank(_mk_case(), [], llm) == []


def test_rerank_llm_error_falls_back_to_posterior_order():
    class BoomClient:
        model_version = "boom"
        def complete(self, **kwargs):
            raise RuntimeError("down")
    candidates = [
        _mk_rule(rule_id="r_hi", success_count=50, failure_count=2),
        _mk_rule(rule_id="r_lo", success_count=3, failure_count=5),
    ]
    out = rerank(_mk_case(), candidates, BoomClient(), top_k=2)
    # Fallback: posterior order preserved, no applicability filter
    assert [r.rule_id for r in out] == ["r_hi", "r_lo"]


def test_rerank_parse_error_falls_back():
    candidates = [_mk_rule(rule_id="r_a")]
    llm = FakeLLMClient(responses=["garbage"])
    out = rerank(_mk_case(), candidates, llm, top_k=1)
    assert out == candidates  # fallback returns full list in posterior order


def test_rerank_missing_entries_get_dropped():
    """Reranker skips a candidate → that rule is treated as
    applicability=0 and filtered out."""
    candidates = [
        _mk_rule(rule_id="r_a"),
        _mk_rule(rule_id="r_b"),
    ]
    llm = FakeLLMClient(responses=[_rerank_response([
        {"rule_id": "r_a", "applicability": 9, "expected_utility": 8},
        # r_b is missing
    ])])
    out = rerank(_mk_case(), candidates, llm, top_k=2)
    assert [r.rule_id for r in out] == ["r_a"]


def test_rerank_all_excluded_returns_empty():
    """Reranker drops everything → empty list. This is valid
    'no useful memory for this case' behavior, not a bug."""
    candidates = [_mk_rule(rule_id="r_a")]
    llm = FakeLLMClient(responses=[_rerank_response([
        {"rule_id": "r_a", "applicability": 1, "expected_utility": 1},
    ])])
    out = rerank(_mk_case(), candidates, llm, top_k=2)
    assert out == []


# ── Composed pipeline.retrieve ──────────────────────────────────────────────


def test_pipeline_retrieve_without_reranker(populated_store):
    """skip_rerank mode should return Tier 2 top-k without an LLM call."""
    case = _mk_case()
    result = retrieve(
        populated_store,
        case,
        embedder=FakeEmbedder(),
        reranker_llm=None,
    )
    assert isinstance(result, RetrievalResult)
    assert result.reranker_called is False
    assert result.tier1_count == 1  # only r_relevant matches the trigger
    assert len(result.rules) >= 1
    assert result.rules[0].rule_id == "r_relevant"


def test_pipeline_retrieve_with_reranker(populated_store):
    case = _mk_case()
    llm = FakeLLMClient(responses=[_rerank_response([
        {"rule_id": "r_relevant", "applicability": 9, "expected_utility": 8},
    ])])
    result = retrieve(
        populated_store,
        case,
        embedder=FakeEmbedder(),
        reranker_llm=llm,
    )
    assert result.reranker_called is True
    assert [r.rule_id for r in result.rules] == ["r_relevant"]


def test_pipeline_empty_store_returns_empty(sqlite_conn):
    store = RuleStore(sqlite_conn)
    result = retrieve(store, _mk_case(), embedder=FakeEmbedder())
    assert result.rules == []
    assert result.tier1_count == 0


def test_pipeline_tier1_miss_widens_to_full_store(populated_store):
    """If no rule structurally matches the trigger, the pipeline
    falls back to the full ACTIVE set so dense + rerank still run."""
    # Case with a CPT family that no rule in the store has
    case = _mk_case(cpt="90837")  # psychotherapy — no rule matches
    result = retrieve(
        populated_store,
        case,
        embedder=FakeEmbedder(),
    )
    # Widened to ACTIVE rules (r_relevant, r_offtopic, r_wrong_payer)
    assert result.tier1_count == 3


def test_pipeline_include_candidate_propagates(populated_store):
    case = _mk_case()
    result = retrieve(
        populated_store,
        case,
        embedder=FakeEmbedder(),
        include_candidate=True,
    )
    ids = [r.rule_id for r in result.rules]
    assert "r_candidate" in ids


def test_pipeline_preserves_top_k_limits(populated_store):
    """tier3_top_k caps the final result regardless of how many
    rules Tier 2 returned."""
    # Populate with 8 matching rules
    for i in range(8):
        populated_store.put(_mk_rule(
            rule_id=f"r_extra_{i}",
            semantic_predicate=f"lumbar MRI conservative therapy {i}",
        ))
    result = retrieve(
        populated_store,
        _mk_case(),
        embedder=FakeEmbedder(),
        tier3_top_k=3,
    )
    assert len(result.rules) <= 3
