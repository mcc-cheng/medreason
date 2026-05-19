"""Tests for the LayerRouter policy enforcement.

We validate the leak-guard semantics WITHOUT requiring a real SQLite
store — LayerRouter.ingest_rule() validates the policy first and only
later writes to a store. Until the store wiring lands, the test surface
is policy-only.
"""

from __future__ import annotations

from pathlib import Path

import pytest

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


def _rule(rule_id: str = "rule_x") -> ReasoningRule:
    return ReasoningRule(
        rule_id=rule_id,
        status=RuleStatus.CANDIDATE,
        trigger=RuleTrigger(semantic_predicate="kinase target with paralog_count >= 2"),
        action="Raise bypass_risk by 0.3.",
        rationale="Paralog compensation common in kinases.",
        polarity="requires_check",
        evidence=RuleEvidence(
            supporting_case_ids=["tv_braf"],
            source_policy_citation="aggregate of 7 cases",
            proposer_model="cross_agent_v0",
            proposer_run_id="run_42",
        ),
    )


def test_universal_layer_rejects_customer_tagged_rule():
    policy = LayerPolicy(layer=Layer.UNIVERSAL)
    with pytest.raises(LayerPolicyViolation):
        policy.validate_rule(_rule(), customer_tag="recursion")


def test_universal_layer_accepts_public_rule():
    policy = LayerPolicy(layer=Layer.UNIVERSAL)
    policy.validate_rule(_rule(), customer_tag=None)  # no raise


def test_disease_layer_rejects_customer_tagged_rule():
    policy = LayerPolicy(layer=Layer.DISEASE, allowed_disease_scope="oncology_mapk")
    with pytest.raises(LayerPolicyViolation):
        policy.validate_rule(_rule(), customer_tag="recursion")


def test_campaign_layer_requires_customer_tag():
    policy = LayerPolicy(
        layer=Layer.CAMPAIGN, allowed_customer_tags=frozenset({"recursion"})
    )
    with pytest.raises(LayerPolicyViolation):
        policy.validate_rule(_rule(), customer_tag=None)


def test_campaign_layer_rejects_wrong_tenant():
    policy = LayerPolicy(
        layer=Layer.CAMPAIGN, allowed_customer_tags=frozenset({"recursion"})
    )
    with pytest.raises(LayerPolicyViolation):
        policy.validate_rule(_rule(), customer_tag="some_other_pharma")


def test_campaign_layer_accepts_correct_tenant():
    policy = LayerPolicy(
        layer=Layer.CAMPAIGN, allowed_customer_tags=frozenset({"recursion"})
    )
    policy.validate_rule(_rule(), customer_tag="recursion")  # no raise


def test_router_ingest_rule_validates_policy():
    """The router should refuse to write a customer-tagged rule into the
    universal layer even before store wiring is complete."""
    router = LayerRouter(
        LayerStorePaths(universal=Path("/dev/null"), disease={}, campaign={})
    )
    with pytest.raises(LayerPolicyViolation):
        router.ingest_rule(
            _rule(), source_layer=Layer.UNIVERSAL, customer_tag="recursion"
        )


def test_router_ingest_rule_accepts_public_universal():
    router = LayerRouter(
        LayerStorePaths(universal=Path("/dev/null"), disease={}, campaign={})
    )
    # Should NOT raise — policy passes; store wiring deferred.
    router.ingest_rule(_rule(), source_layer=Layer.UNIVERSAL, customer_tag=None)


def test_promotion_is_intentionally_unimplemented():
    router = LayerRouter(
        LayerStorePaths(universal=Path("/dev/null"), disease={}, campaign={})
    )
    with pytest.raises(NotImplementedError):
        router.promote_rule(
            "rule_x", source_layer=Layer.CAMPAIGN, target_layer=Layer.DISEASE
        )
