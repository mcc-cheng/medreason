"""Tests for the LayerRouter policy + store wiring.

Two test surfaces:

1. Pure policy validation — LayerPolicy.validate_rule() raises the right
   thing on customer-tag leaks (these run without touching any store).
2. End-to-end router wiring — LayerRouter.ingest_rule writes to a real
   sqlite-backed RuleStore, retrieve_for_case unions across layers, and
   promote_rule abstracts + writes to the higher layer.

The second surface uses tmp_path so each test gets a fresh sqlite file.
The `/dev/null` path used by the legacy tests is preserved as a request
for an ephemeral in-memory store.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytest

from medreason.llm.base import FakeLLMClient, LLMResponse
from medreason.ontology.rule import (
    ReasoningRule,
    RuleEvidence,
    RuleStatus,
    RuleTrigger,
)
from medreason.targetval.layers import (
    Layer,
    LayerPolicy,
    LayerPolicyViolation,
    LayerRouter,
    LayerStorePaths,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _rule(
    rule_id: str = "rule_x",
    *,
    status: RuleStatus = RuleStatus.CANDIDATE,
    supporting_case_ids: Optional[list[str]] = None,
    proposer_run_id: str = "run_42",
    semantic_predicate: str = "kinase target with paralog_count >= 2",
) -> ReasoningRule:
    return ReasoningRule(
        rule_id=rule_id,
        status=status,
        trigger=RuleTrigger(semantic_predicate=semantic_predicate),
        action="Raise bypass_risk by 0.3.",
        rationale="Paralog compensation common in kinases.",
        polarity="requires_check",
        evidence=RuleEvidence(
            supporting_case_ids=supporting_case_ids
            if supporting_case_ids is not None
            else ["tv_braf"],
            source_policy_citation="aggregate of 7 cases",
            proposer_model="cross_agent_v0",
            proposer_run_id=proposer_run_id,
        ),
    )


def _dev_null_paths() -> LayerStorePaths:
    """The /dev/null preset used by legacy policy-only tests."""
    return LayerStorePaths(
        universal=Path("/dev/null"), disease={}, campaign={}
    )


def _tmp_paths(tmp_path: Path) -> LayerStorePaths:
    """Persistent-store preset: real sqlite files under tmp_path, one
    disease scope, one campaign tenant."""
    return LayerStorePaths(
        universal=tmp_path / "universal.sqlite",
        disease={"oncology_mapk": tmp_path / "disease_oncology_mapk.sqlite"},
        campaign={"recursion": tmp_path / "campaign_recursion.sqlite"},
    )


# ── 1. Policy-only tests (preserved from baseline) ──────────────────────────


def test_universal_layer_rejects_customer_tagged_rule():
    policy = LayerPolicy(layer=Layer.UNIVERSAL)
    with pytest.raises(LayerPolicyViolation):
        policy.validate_rule(_rule(), customer_tag="recursion")


def test_universal_layer_accepts_public_rule():
    policy = LayerPolicy(layer=Layer.UNIVERSAL)
    policy.validate_rule(_rule(), customer_tag=None)  # no raise


def test_disease_layer_rejects_customer_tagged_rule():
    policy = LayerPolicy(
        layer=Layer.DISEASE, allowed_disease_scope="oncology_mapk"
    )
    with pytest.raises(LayerPolicyViolation):
        policy.validate_rule(_rule(), customer_tag="recursion")


def test_campaign_layer_requires_customer_tag():
    policy = LayerPolicy(
        layer=Layer.CAMPAIGN,
        allowed_customer_tags=frozenset({"recursion"}),
    )
    with pytest.raises(LayerPolicyViolation):
        policy.validate_rule(_rule(), customer_tag=None)


def test_campaign_layer_rejects_wrong_tenant():
    policy = LayerPolicy(
        layer=Layer.CAMPAIGN,
        allowed_customer_tags=frozenset({"recursion"}),
    )
    with pytest.raises(LayerPolicyViolation):
        policy.validate_rule(_rule(), customer_tag="some_other_pharma")


def test_campaign_layer_accepts_correct_tenant():
    policy = LayerPolicy(
        layer=Layer.CAMPAIGN,
        allowed_customer_tags=frozenset({"recursion"}),
    )
    policy.validate_rule(_rule(), customer_tag="recursion")  # no raise


def test_router_ingest_rule_validates_policy():
    """The router refuses to write a customer-tagged rule into the
    universal layer, even with the /dev/null preset."""
    router = LayerRouter(_dev_null_paths())
    with pytest.raises(LayerPolicyViolation):
        router.ingest_rule(
            _rule(), source_layer=Layer.UNIVERSAL, customer_tag="recursion"
        )


def test_router_ingest_rule_accepts_public_universal():
    """The /dev/null preset MUST stay non-fatal: legacy callers use it."""
    router = LayerRouter(_dev_null_paths())
    router.ingest_rule(
        _rule(), source_layer=Layer.UNIVERSAL, customer_tag=None
    )  # no raise


# ── 2. Round-trip ingest/retrieve tests ─────────────────────────────────────


def test_router_ingest_then_retrieve_universal(tmp_path):
    router = LayerRouter(_tmp_paths(tmp_path))
    rule = _rule(rule_id="rule_universal_1", status=RuleStatus.ACTIVE)
    router.ingest_rule(rule, source_layer=Layer.UNIVERSAL)

    rules = router.retrieve_for_case(
        case_disease_scope="oncology_mapk", customer_tag=None
    )
    assert [r.rule_id for r in rules] == ["rule_universal_1"]
    # Round-trips the full body, not just the id.
    assert rules[0].action == rule.action


def test_router_ingest_only_active_is_retrievable(tmp_path):
    """retrieve_for_case returns ACTIVE rules only — CANDIDATEs sit
    inert until promoted."""
    router = LayerRouter(_tmp_paths(tmp_path))
    router.ingest_rule(
        _rule(rule_id="rule_candidate", status=RuleStatus.CANDIDATE),
        source_layer=Layer.UNIVERSAL,
    )
    router.ingest_rule(
        _rule(rule_id="rule_active", status=RuleStatus.ACTIVE),
        source_layer=Layer.UNIVERSAL,
    )
    rules = router.retrieve_for_case(
        case_disease_scope="oncology_mapk", customer_tag=None
    )
    assert [r.rule_id for r in rules] == ["rule_active"]


def test_router_ingest_is_idempotent_on_rule_id(tmp_path):
    """Calling ingest_rule twice with the same rule_id replaces — no
    duplicates and no error."""
    router = LayerRouter(_tmp_paths(tmp_path))
    r1 = _rule(rule_id="rule_dup", status=RuleStatus.ACTIVE)
    r2 = _rule(rule_id="rule_dup", status=RuleStatus.ACTIVE)
    r2.action = "Updated action."
    router.ingest_rule(r1, source_layer=Layer.UNIVERSAL)
    router.ingest_rule(r2, source_layer=Layer.UNIVERSAL)
    rules = router.retrieve_for_case(
        case_disease_scope="oncology_mapk", customer_tag=None
    )
    assert len(rules) == 1
    assert rules[0].action == "Updated action."


def test_router_retrieve_unions_across_layers(tmp_path):
    """A universal, disease, and campaign rule should all appear when
    the caller's case matches each scope."""
    router = LayerRouter(_tmp_paths(tmp_path))

    router.ingest_rule(
        _rule(rule_id="universal_rule", status=RuleStatus.ACTIVE),
        source_layer=Layer.UNIVERSAL,
    )
    router.ingest_rule(
        _rule(rule_id="disease_rule", status=RuleStatus.ACTIVE),
        source_layer=Layer.DISEASE,
        disease_scope="oncology_mapk",
    )
    router.ingest_rule(
        _rule(rule_id="campaign_rule", status=RuleStatus.ACTIVE),
        source_layer=Layer.CAMPAIGN,
        customer_tag="recursion",
    )

    rules = router.retrieve_for_case(
        case_disease_scope="oncology_mapk",
        customer_tag="recursion",
        top_k=10,
    )
    ids = {r.rule_id for r in rules}
    assert ids == {"universal_rule", "disease_rule", "campaign_rule"}


def test_router_retrieve_omits_other_tenant_campaign_rules(tmp_path):
    """A campaign rule for tenant A must not appear in retrieve for
    tenant B, even on the same disease scope."""
    paths = LayerStorePaths(
        universal=tmp_path / "universal.sqlite",
        disease={"oncology_mapk": tmp_path / "disease.sqlite"},
        campaign={
            "recursion": tmp_path / "campaign_recursion.sqlite",
            "other_pharma": tmp_path / "campaign_other.sqlite",
        },
    )
    router = LayerRouter(paths)

    router.ingest_rule(
        _rule(rule_id="recursion_only_rule", status=RuleStatus.ACTIVE),
        source_layer=Layer.CAMPAIGN,
        customer_tag="recursion",
    )

    rules = router.retrieve_for_case(
        case_disease_scope="oncology_mapk",
        customer_tag="other_pharma",
        top_k=10,
    )
    assert all(r.rule_id != "recursion_only_rule" for r in rules)


def test_router_retrieve_omits_unmatched_disease_scope(tmp_path):
    """A disease rule for scope X should not appear when retrieving for
    a different disease scope Y."""
    paths = LayerStorePaths(
        universal=tmp_path / "universal.sqlite",
        disease={
            "oncology_mapk": tmp_path / "mapk.sqlite",
            "oncology_pi3k": tmp_path / "pi3k.sqlite",
        },
        campaign={},
    )
    router = LayerRouter(paths)
    router.ingest_rule(
        _rule(rule_id="mapk_rule", status=RuleStatus.ACTIVE),
        source_layer=Layer.DISEASE,
        disease_scope="oncology_mapk",
    )
    rules = router.retrieve_for_case(
        case_disease_scope="oncology_pi3k", customer_tag=None
    )
    assert all(r.rule_id != "mapk_rule" for r in rules)


def test_router_retrieve_top_k_truncates(tmp_path):
    router = LayerRouter(_tmp_paths(tmp_path))
    for i in range(5):
        router.ingest_rule(
            _rule(rule_id=f"rule_{i}", status=RuleStatus.ACTIVE),
            source_layer=Layer.UNIVERSAL,
        )
    rules = router.retrieve_for_case(
        case_disease_scope="oncology_mapk", customer_tag=None, top_k=3
    )
    assert len(rules) == 3


# ── 3. Promotion tests ──────────────────────────────────────────────────────


def _abstractor_response_for(action: str, predicate: str, rationale: str) -> str:
    """Build the JSON shape rule_abstractor.parse_json_response expects."""
    import json

    return json.dumps(
        {
            "abstracted_action": action,
            "abstracted_semantic_predicate": predicate,
            "abstracted_rationale": rationale,
        }
    )


def test_promote_campaign_to_disease_strips_customer_tag(tmp_path):
    """Ingest a campaign-tagged rule whose supporting_case_ids embed the
    tenant tag; promote to DISEASE; assert the higher-layer rule no
    longer carries the tenant tag and is retrievable."""
    router = LayerRouter(_tmp_paths(tmp_path))

    campaign_rule = _rule(
        rule_id="r_campaign",
        status=RuleStatus.ACTIVE,
        supporting_case_ids=["recursion:tv_braf", "public:tv_kras"],
        proposer_run_id="recursion_run_1",
    )
    router.ingest_rule(
        campaign_rule,
        source_layer=Layer.CAMPAIGN,
        customer_tag="recursion",
    )

    llm = FakeLLMClient(
        responses=[
            _abstractor_response_for(
                "Raise bypass_risk when paralog count >= 2.",
                "kinase target with >= 2 paralogs in same family",
                "Paralog compensation generalizes across cancers.",
            )
        ]
    )

    promoted = router.promote_rule(
        "r_campaign",
        source_layer=Layer.CAMPAIGN,
        target_layer=Layer.DISEASE,
        llm=llm,
        source_customer_tag="recursion",
        target_disease_scope="oncology_mapk",
    )

    assert promoted is not None
    assert promoted.status is RuleStatus.ACTIVE
    # The recursion-tagged case_id is stripped; the public one survives.
    assert "recursion:tv_braf" not in promoted.evidence.supporting_case_ids
    assert "public:tv_kras" in promoted.evidence.supporting_case_ids
    # proposer_run_id contained "recursion" → cleared
    assert "recursion" not in (promoted.evidence.proposer_run_id or "")

    # And the promoted rule is now retrievable from the DISEASE layer.
    rules = router.retrieve_for_case(
        case_disease_scope="oncology_mapk", customer_tag=None
    )
    assert any(r.rule_id == "r_campaign" for r in rules)


def test_promote_campaign_to_universal_disallowed(tmp_path):
    """Only one-hop promotion is allowed; campaign→universal must fail."""
    router = LayerRouter(_tmp_paths(tmp_path))
    router.ingest_rule(
        _rule(rule_id="r1", status=RuleStatus.ACTIVE),
        source_layer=Layer.CAMPAIGN,
        customer_tag="recursion",
    )
    with pytest.raises(LayerPolicyViolation):
        router.promote_rule(
            "r1",
            source_layer=Layer.CAMPAIGN,
            target_layer=Layer.UNIVERSAL,
            llm=FakeLLMClient(),
            source_customer_tag="recursion",
        )


def test_promote_disease_to_universal_allowed(tmp_path):
    """The other allowed edge: DISEASE → UNIVERSAL."""
    router = LayerRouter(_tmp_paths(tmp_path))
    router.ingest_rule(
        _rule(rule_id="r_disease", status=RuleStatus.ACTIVE),
        source_layer=Layer.DISEASE,
        disease_scope="oncology_mapk",
    )
    llm = FakeLLMClient(
        responses=[
            _abstractor_response_for(
                "Generalised action.",
                "Generalised predicate.",
                "Generalised rationale.",
            )
        ]
    )
    promoted = router.promote_rule(
        "r_disease",
        source_layer=Layer.DISEASE,
        target_layer=Layer.UNIVERSAL,
        llm=llm,
        source_disease_scope="oncology_mapk",
    )
    assert promoted is not None
    rules = router.retrieve_for_case(
        case_disease_scope="oncology_mapk", customer_tag=None
    )
    assert any(r.rule_id == "r_disease" for r in rules)


def test_promote_rejects_candidate_status(tmp_path):
    """CANDIDATE rules haven't passed the generalization gate; they
    must not propagate up the hierarchy. promote_rule returns None
    (not raise) so the caller can keep iterating."""
    router = LayerRouter(_tmp_paths(tmp_path))
    router.ingest_rule(
        _rule(rule_id="r_pending", status=RuleStatus.CANDIDATE),
        source_layer=Layer.CAMPAIGN,
        customer_tag="recursion",
    )
    promoted = router.promote_rule(
        "r_pending",
        source_layer=Layer.CAMPAIGN,
        target_layer=Layer.DISEASE,
        llm=FakeLLMClient(),
        source_customer_tag="recursion",
        target_disease_scope="oncology_mapk",
    )
    assert promoted is None


def test_promote_returns_none_for_missing_rule(tmp_path):
    router = LayerRouter(_tmp_paths(tmp_path))
    promoted = router.promote_rule(
        "no_such_rule",
        source_layer=Layer.CAMPAIGN,
        target_layer=Layer.DISEASE,
        llm=FakeLLMClient(),
        source_customer_tag="recursion",
        target_disease_scope="oncology_mapk",
    )
    assert promoted is None


def test_promote_raises_when_customer_tag_survives_strip(tmp_path):
    """If the abstractor doesn't reword and supporting_case_ids ALL
    reference the tenant, stripping leaves the rule still tenant-tied.
    We then hit the leak check: customer_tag in source_policy_citation
    survives the strip pass (we don't rewrite the citation), so the
    promote must raise."""
    router = LayerRouter(_tmp_paths(tmp_path))

    # Build a rule whose source_policy_citation mentions the tenant,
    # which the strip helper does NOT rewrite (it only touches
    # supporting_case_ids + proposer_run_id). This forces the survives
    # check to fire.
    leaky_rule = ReasoningRule(
        rule_id="r_leaky",
        status=RuleStatus.ACTIVE,
        trigger=RuleTrigger(semantic_predicate="leaky"),
        action="Leaky action.",
        rationale="Leaky rationale.",
        polarity="requires_check",
        evidence=RuleEvidence(
            supporting_case_ids=["public:tv_x"],
            source_policy_citation="recursion proprietary screen note",
            proposer_model="cross_agent_v0",
            proposer_run_id="run_clean",
        ),
    )
    router.ingest_rule(
        leaky_rule,
        source_layer=Layer.CAMPAIGN,
        customer_tag="recursion",
    )

    with pytest.raises(LayerPolicyViolation):
        router.promote_rule(
            "r_leaky",
            source_layer=Layer.CAMPAIGN,
            target_layer=Layer.DISEASE,
            llm=FakeLLMClient(),
            source_customer_tag="recursion",
            target_disease_scope="oncology_mapk",
        )


def test_promote_disease_to_campaign_disallowed(tmp_path):
    """Promotion goes UP, never down. DISEASE→CAMPAIGN is not allowed."""
    router = LayerRouter(_tmp_paths(tmp_path))
    router.ingest_rule(
        _rule(rule_id="r_d", status=RuleStatus.ACTIVE),
        source_layer=Layer.DISEASE,
        disease_scope="oncology_mapk",
    )
    with pytest.raises(LayerPolicyViolation):
        router.promote_rule(
            "r_d",
            source_layer=Layer.DISEASE,
            target_layer=Layer.CAMPAIGN,
            llm=FakeLLMClient(),
            source_disease_scope="oncology_mapk",
        )


def test_promote_calls_optional_gate(tmp_path):
    """When a gate is provided, a failing gate rejects the promotion."""
    router = LayerRouter(_tmp_paths(tmp_path))
    router.ingest_rule(
        _rule(rule_id="r_gated", status=RuleStatus.ACTIVE),
        source_layer=Layer.DISEASE,
        disease_scope="oncology_mapk",
    )

    class _RejectingGate:
        def validate(self, rule):
            from medreason.extraction.generalization_gate import GateResult

            return GateResult(
                rule_id=rule.rule_id,
                new_status=RuleStatus.DEPRECATED,
                score=0.0,
                total_trials=5,
                correct=0,
                reason="rejected",
                evaluated_case_ids=[],
            )

    llm = FakeLLMClient(
        responses=[
            _abstractor_response_for("a", "p", "r"),
        ]
    )
    promoted = router.promote_rule(
        "r_gated",
        source_layer=Layer.DISEASE,
        target_layer=Layer.UNIVERSAL,
        llm=llm,
        gate=_RejectingGate(),
        source_disease_scope="oncology_mapk",
    )
    assert promoted is None
