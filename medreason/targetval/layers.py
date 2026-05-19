"""Three-layer memory split: Universal / Disease / Campaign.

Each layer is a RuleStore with its own LeakGuard policy. LayerRouter is
the only thing that talks to all three — it composes them for retrieval,
and enforces per-layer write-time policies.

Why three layers (and not one big store)?

- Different transfer rules. Universal rules must be safe to apply across
  every customer, so their training data must be public. Disease rules
  apply within a disease cluster but still across customers. Campaign
  rules are strictly per-customer.
- Different leak-guard policies. Universal store rejects any rule whose
  provenance includes a campaign-tagged case_id (else customer data
  leaks cross-tenant via "universal" rules). Disease store rejects rules
  whose disease tag does not match its scope. Campaign store rejects
  rules from other tenants.
- Different promotion lifecycles. A campaign-layer rule may be
  *promoted* to a disease-layer rule (or universal-layer rule) only via
  the abstractor + generalization gate, which strips customer-specific
  references and validates against held-out targets.

The data-sharing constraint that makes this real:
- A pharma customer will let universal rules trained on public data
  inform their analysis.
- They will NOT let a rule trained on their proprietary screen feed
  another customer's analysis.
- LayerPolicy enforces this at write time. There is no "trust me" path.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Optional

from ..ontology.rule import ReasoningRule, RuleStatus


class Layer(str, Enum):
    UNIVERSAL = "universal"
    DISEASE = "disease"
    CAMPAIGN = "campaign"


# ── Policy ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LayerPolicy:
    """Write-time rules for a layer.

    Attributes
    ----------
    layer
        Which layer this policy guards.
    allowed_customer_tags
        For CAMPAIGN: exactly one tag — the tenant. For DISEASE /
        UNIVERSAL: empty set; any customer_tag in the rule's provenance
        is a violation.
    allowed_disease_scope
        For DISEASE: the disease cluster slug this store covers (e.g.,
        "oncology_mapk"). For UNIVERSAL: empty; rules must be disease-
        agnostic. For CAMPAIGN: empty; campaign scope is the tenant.
    """

    layer: Layer
    allowed_customer_tags: frozenset[str] = frozenset()
    allowed_disease_scope: Optional[str] = None

    def validate_rule(self, rule: ReasoningRule, customer_tag: Optional[str]) -> None:
        """Raise LayerPolicyViolation if the rule violates this layer's policy.

        Customer_tag is the tag of the case the rule was learned from
        (None if the rule was learned from public data).
        """
        if self.layer is Layer.UNIVERSAL:
            if customer_tag is not None:
                raise LayerPolicyViolation(
                    f"Universal layer rejects rule learned from customer "
                    f"tag {customer_tag!r}; universal rules must come from "
                    f"public data only."
                )
        elif self.layer is Layer.DISEASE:
            if customer_tag is not None:
                raise LayerPolicyViolation(
                    f"Disease layer rejects rule learned from customer "
                    f"tag {customer_tag!r} without abstraction. Promote "
                    f"via rule_abstractor + generalization_gate first."
                )
            # Disease-scope check is enforced by the trigger schema, not here.
        elif self.layer is Layer.CAMPAIGN:
            if customer_tag is None:
                raise LayerPolicyViolation(
                    "Campaign layer requires a customer_tag — a rule "
                    "without one belongs in DISEASE or UNIVERSAL."
                )
            if customer_tag not in self.allowed_customer_tags:
                raise LayerPolicyViolation(
                    f"Campaign layer for tenants "
                    f"{sorted(self.allowed_customer_tags)} rejects rule "
                    f"learned from tenant {customer_tag!r}."
                )


class LayerPolicyViolation(Exception):
    """A write that violates the per-layer policy. Always fatal."""


# ── Router ────────────────────────────────────────────────────────────────────


@dataclass
class LayerStorePaths:
    """SQLite paths for each layer's RuleStore. Disease/campaign maps from
    scope tag → path so a single LayerRouter can serve multiple disease
    clusters or tenants.
    """

    universal: Path
    disease: dict[str, Path]
    campaign: dict[str, Path]


class LayerRouter:
    """Façade that owns three RuleStore instances (or a dict per layer)
    and routes retrieval / promotion / writes by layer.

    Skeleton: stores are not instantiated yet. The constructor takes
    paths so a real wiring step can build the stores later.

    The retrieval flow:

        retrieve(case)
        → tier1 ontology_lookup on UNIVERSAL → top N1
        → tier1 ontology_lookup on DISEASE store matching case.disease → top N2
        → tier1 ontology_lookup on CAMPAIGN store matching case.customer → top N3
        → union, dense-rerank (Tier 2), LLM-rerank (Tier 3)
        → return top-K rules with layer-of-origin annotation

    The write flow:

        ingest_rule(rule, source_layer, customer_tag, disease_scope)
        → policy.validate_rule(rule, customer_tag)
        → underlying RuleStore.put(rule)

    The promotion flow (campaign → disease, disease → universal):

        promote_rule(rule_id, target_layer)
        → load rule from source layer
        → run rule_abstractor (strips customer-specific refs)
        → run generalization_gate on target_layer's eval set
        → if gate passes: target_layer.put(abstracted_rule)
    """

    def __init__(self, paths: LayerStorePaths):
        self.paths = paths
        # NOTE: RuleStore wiring deferred. Plan: each (layer, scope) gets
        # its own sqlite3.connect() + RuleStore + LeakGuard tuned to that
        # layer's policy.
        self._stores_initialised = False

    # ── Retrieval (skeleton) ─────────────────────────────────────────────

    def retrieve_for_case(
        self,
        case_disease_scope: str,
        customer_tag: Optional[str],
        top_k: int = 5,
    ) -> list[ReasoningRule]:
        """Return rules from all three layers, deduped, with layer-of-
        origin annotation in `rule.evidence.proposer_run_id` (the existing
        field doubles as a layer marker for now).

        SKELETON: returns []. Real flow uses the existing
        medreason.retrieval.pipeline + tier1 ontology_lookup against each
        layer's store, then unions + reranks.
        """
        _ = (case_disease_scope, customer_tag, top_k)
        return []

    # ── Write (skeleton) ─────────────────────────────────────────────────

    def ingest_rule(
        self,
        rule: ReasoningRule,
        *,
        source_layer: Layer,
        customer_tag: Optional[str] = None,
        disease_scope: Optional[str] = None,
    ) -> None:
        """Validate and write a rule into its destination layer.

        SKELETON: validates policy but does NOT write — store wiring
        comes next. Raises LayerPolicyViolation if the rule and target
        layer disagree.
        """
        policy = self._policy_for(source_layer, customer_tag, disease_scope)
        policy.validate_rule(rule, customer_tag)
        # TODO: RuleStore.put — deferred until store wiring lands.

    def promote_rule(
        self,
        rule_id: str,
        *,
        source_layer: Layer,
        target_layer: Layer,
    ) -> None:
        """Promote a rule between layers via abstraction + generalization gate.

        SKELETON: raises NotImplementedError. The flow is:
            1. load rule from source_layer store
            2. medreason.extraction.rule_abstractor.abstract_rule(rule, llm)
            3. medreason.extraction.generalization_gate.gate(abstracted,
               eval_set=target_layer.eval_set)
            4. on pass: target_layer.put(abstracted_rule)
        """
        raise NotImplementedError(
            "promote_rule is intentionally stubbed — wire after RuleStore "
            "wiring in LayerRouter is implemented."
        )

    # ── Internal ─────────────────────────────────────────────────────────

    def _policy_for(
        self,
        layer: Layer,
        customer_tag: Optional[str],
        disease_scope: Optional[str],
    ) -> LayerPolicy:
        if layer is Layer.UNIVERSAL:
            return LayerPolicy(layer=Layer.UNIVERSAL)
        if layer is Layer.DISEASE:
            return LayerPolicy(layer=Layer.DISEASE, allowed_disease_scope=disease_scope)
        if layer is Layer.CAMPAIGN:
            if customer_tag is None:
                raise ValueError("CAMPAIGN policy requires customer_tag")
            return LayerPolicy(
                layer=Layer.CAMPAIGN,
                allowed_customer_tags=frozenset({customer_tag}),
            )
        raise ValueError(f"Unknown layer: {layer!r}")


def candidate_rules_active(rules: Iterable[ReasoningRule]) -> list[ReasoningRule]:
    """Helper: filter to ACTIVE-status rules. The retrieval pipeline already
    does this — exposed here for tests + non-pipeline callers."""
    return [r for r in rules if r.status is RuleStatus.ACTIVE]
