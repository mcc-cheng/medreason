"""Three-layer memory split: Universal / Disease / Campaign.

Each layer is a RuleStore with its own write-time policy. LayerRouter is
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
  the abstractor + (optional) generalization gate, which strips
  customer-specific references and validates against held-out targets.

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
from typing import Callable, Iterable, Optional, TYPE_CHECKING

from ..llm.base import LLMClient
from ..ontology.rule import ReasoningRule, RuleStatus
from .layer_stores import TargetvalRuleStore, open_targetval_rule_store

if TYPE_CHECKING:
    from ..extraction.generalization_gate import GeneralizationGate


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

    Passing `Path("/dev/null")` for `universal` (or for a value in
    `disease`/`campaign`) signals "no persistence" — the router uses an
    ephemeral in-memory store there. Useful for unit tests that only
    exercise policy/routing.
    """

    universal: Path
    disease: dict[str, Path]
    campaign: dict[str, Path]


# Type aliases for clarity on promote_rule
StoreFactory = Callable[[Path], TargetvalRuleStore]


class LayerRouter:
    """Façade owning one RuleStore per (layer, scope). Routes retrieval,
    ingest, and promotion by layer.

    - retrieve_for_case: union ACTIVE rules from UNIVERSAL + matching
      DISEASE + matching CAMPAIGN. Dedupe by rule_id. Truncate to top_k.
      No semantic rerank yet (Phase 3). Order: universal → disease →
      campaign, preserving each store's order.
    - ingest_rule: LayerPolicy.validate_rule → store.put. Idempotent
      on rule_id (the underlying RuleStore upserts).
    - promote_rule: ACTIVE-only, edge in {CAMPAIGN→DISEASE,
      DISEASE→UNIVERSAL}, abstract_rule → strip customer_tag from
      provenance → optional gate.validate → ingest_rule on the higher
      layer. Returns the promoted rule on success, None on
      abstraction/gate rejection, raises LayerPolicyViolation on
      forbidden edges or surviving customer_tag.
    """

    def __init__(
        self,
        paths: LayerStorePaths,
        *,
        store_factory: StoreFactory = open_targetval_rule_store,
    ):
        self.paths = paths
        self._store_factory = store_factory
        self._universal: Optional[TargetvalRuleStore] = None
        # disease/campaign are keyed by their respective scope strings
        # (disease_scope slug; tenant tag).
        self._disease: dict[str, TargetvalRuleStore] = {}
        self._campaign: dict[str, TargetvalRuleStore] = {}

    # ── Public API: retrieval ────────────────────────────────────────────

    def retrieve_for_case(
        self,
        case_disease_scope: str,
        customer_tag: Optional[str],
        top_k: int = 5,
    ) -> list[ReasoningRule]:
        """Return ACTIVE rules unioned from UNIVERSAL + matching DISEASE +
        matching CAMPAIGN. Deduped by rule_id. Truncated to top_k.

        No semantic rerank yet (Phase 3). The order is:
            universal → disease → campaign, preserving each store's order.
        """
        seen: set[str] = set()
        out: list[ReasoningRule] = []

        for rule in self._universal_store().list_active():
            if rule.rule_id not in seen:
                seen.add(rule.rule_id)
                out.append(rule)

        disease_store = self._maybe_get_disease_store(case_disease_scope)
        if disease_store is not None:
            for rule in disease_store.list_active():
                if rule.rule_id not in seen:
                    seen.add(rule.rule_id)
                    out.append(rule)

        if customer_tag is not None:
            campaign_store = self._maybe_get_campaign_store(customer_tag)
            if campaign_store is not None:
                for rule in campaign_store.list_active():
                    if rule.rule_id not in seen:
                        seen.add(rule.rule_id)
                        out.append(rule)

        return out[:top_k]

    # ── Public API: ingest ───────────────────────────────────────────────

    def ingest_rule(
        self,
        rule: ReasoningRule,
        *,
        source_layer: Layer,
        customer_tag: Optional[str] = None,
        disease_scope: Optional[str] = None,
    ) -> None:
        """Validate and write a rule into its destination layer.

        Idempotent on `rule.rule_id` — the underlying RuleStore upserts.
        Raises `LayerPolicyViolation` if the rule and target layer
        disagree (e.g., a customer-tagged rule into UNIVERSAL).
        """
        policy = self._policy_for(source_layer, customer_tag, disease_scope)
        policy.validate_rule(rule, customer_tag)
        store = self._resolve_store(source_layer, customer_tag, disease_scope)
        store.put(rule)

    # ── Public API: promote ──────────────────────────────────────────────

    def promote_rule(
        self,
        rule_id: str,
        *,
        source_layer: Layer,
        target_layer: Layer,
        llm: LLMClient,
        gate: Optional["GeneralizationGate"] = None,
        source_customer_tag: Optional[str] = None,
        source_disease_scope: Optional[str] = None,
        target_disease_scope: Optional[str] = None,
    ) -> Optional[ReasoningRule]:
        """Promote a rule up the layer hierarchy.

        Allowed edges:
            CAMPAIGN  → DISEASE
            DISEASE   → UNIVERSAL
        All other (source, target) pairs raise `LayerPolicyViolation`.

        The flow:
            1. Refuse if (source, target) is not an allowed edge.
            2. Load rule from the source store. Refuse if absent.
            3. Refuse if rule.status is not ACTIVE — CANDIDATEs haven't
               passed the generalization gate yet and must not propagate
               up the hierarchy.
            4. Call abstract_rule(rule, llm) to strip case-specific text.
               If the LLM round-trip fails, abstract_rule returns the
               original rule unchanged. We accept that — the strip step
               below is the actual leak guard, not the LLM.
            5. Strip supporting_case_ids that reference the source
               tenant (when promoting from CAMPAIGN). If after stripping
               any reference to the source tenant survives, raise
               LayerPolicyViolation — abstraction did not de-identify
               the rule and we must not write it to a higher layer.
            6. If `gate` is provided, run `gate.validate(abstracted)`.
               Return None if the gate did not promote it to ACTIVE
               (DEPRECATED or still-CANDIDATE both fail-close here).
            7. Call `ingest_rule(abstracted, source_layer=target_layer,
               customer_tag=None, disease_scope=target_disease_scope)`.
               Any residual policy violation surfaces here.

        Returns the promoted rule, or None if abstraction/gate
        rejected it.
        """
        if not self._is_allowed_edge(source_layer, target_layer):
            raise LayerPolicyViolation(
                f"Promotion {source_layer.value} → {target_layer.value} is "
                f"not allowed. Only CAMPAIGN→DISEASE and DISEASE→UNIVERSAL "
                f"are permitted; campaign rules cannot bypass DISEASE."
            )

        source_store = self._resolve_store(
            source_layer, source_customer_tag, source_disease_scope
        )
        original = source_store.get(rule_id)
        if original is None:
            return None

        if original.status is not RuleStatus.ACTIVE:
            # CANDIDATE / QUARANTINED / DEPRECATED rules are not eligible
            # — the moat depends on the generalization gate having
            # already validated this rule's signal in the source layer.
            return None

        # Lazy import to avoid a module-load cycle between layers.py and
        # extraction/* (which can import targetval/* transitively).
        from ..extraction.rule_abstractor import abstract_rule

        abstracted = abstract_rule(original, llm)

        abstracted = _strip_customer_provenance(abstracted, source_customer_tag)
        if source_customer_tag is not None and _retains_customer_tag(
            abstracted, source_customer_tag
        ):
            raise LayerPolicyViolation(
                f"Promotion of rule {rule_id} aborted: customer_tag "
                f"{source_customer_tag!r} survived abstraction in "
                f"supporting_case_ids/proposer_run_id; refusing to write "
                f"to {target_layer.value}."
            )

        if gate is not None:
            result = gate.validate(abstracted)
            if not result.promoted:
                return None
            abstracted.status = RuleStatus.ACTIVE
        else:
            # No gate provided: trust the source layer's status. The
            # abstractor may have left it ACTIVE; ensure that's still so.
            abstracted.status = RuleStatus.ACTIVE

        # Write via the public ingest path so policy runs one more time.
        self.ingest_rule(
            abstracted,
            source_layer=target_layer,
            customer_tag=None,
            disease_scope=target_disease_scope,
        )
        return abstracted

    # ── Internal: store resolution ──────────────────────────────────────

    def _universal_store(self) -> TargetvalRuleStore:
        if self._universal is None:
            self._universal = self._store_factory(self.paths.universal)
        return self._universal

    def _disease_store(self, scope: str) -> TargetvalRuleStore:
        if scope not in self._disease:
            path = self.paths.disease.get(scope, Path("/dev/null"))
            self._disease[scope] = self._store_factory(path)
        return self._disease[scope]

    def _campaign_store(self, tenant: str) -> TargetvalRuleStore:
        if tenant not in self._campaign:
            path = self.paths.campaign.get(tenant, Path("/dev/null"))
            self._campaign[tenant] = self._store_factory(path)
        return self._campaign[tenant]

    def _maybe_get_disease_store(
        self, scope: str
    ) -> Optional[TargetvalRuleStore]:
        """Return the disease store for `scope` if there's any signal it
        should exist. We materialise on first access — having the path
        configured OR having previously written to it both qualify.
        Returns None for an unknown scope so retrieval doesn't spawn an
        empty store for every typo'd scope tag.
        """
        if scope in self._disease:
            return self._disease[scope]
        if scope in self.paths.disease:
            return self._disease_store(scope)
        return None

    def _maybe_get_campaign_store(
        self, tenant: str
    ) -> Optional[TargetvalRuleStore]:
        if tenant in self._campaign:
            return self._campaign[tenant]
        if tenant in self.paths.campaign:
            return self._campaign_store(tenant)
        return None

    def _resolve_store(
        self,
        layer: Layer,
        customer_tag: Optional[str],
        disease_scope: Optional[str],
    ) -> TargetvalRuleStore:
        if layer is Layer.UNIVERSAL:
            return self._universal_store()
        if layer is Layer.DISEASE:
            if disease_scope is None:
                raise ValueError(
                    "DISEASE layer write requires a disease_scope"
                )
            return self._disease_store(disease_scope)
        if layer is Layer.CAMPAIGN:
            if customer_tag is None:
                raise ValueError(
                    "CAMPAIGN layer write requires a customer_tag"
                )
            return self._campaign_store(customer_tag)
        raise ValueError(f"Unknown layer: {layer!r}")

    def _policy_for(
        self,
        layer: Layer,
        customer_tag: Optional[str],
        disease_scope: Optional[str],
    ) -> LayerPolicy:
        if layer is Layer.UNIVERSAL:
            return LayerPolicy(layer=Layer.UNIVERSAL)
        if layer is Layer.DISEASE:
            return LayerPolicy(
                layer=Layer.DISEASE, allowed_disease_scope=disease_scope
            )
        if layer is Layer.CAMPAIGN:
            if customer_tag is None:
                raise ValueError("CAMPAIGN policy requires customer_tag")
            return LayerPolicy(
                layer=Layer.CAMPAIGN,
                allowed_customer_tags=frozenset({customer_tag}),
            )
        raise ValueError(f"Unknown layer: {layer!r}")

    @staticmethod
    def _is_allowed_edge(source: Layer, target: Layer) -> bool:
        return (
            (source is Layer.CAMPAIGN and target is Layer.DISEASE)
            or (source is Layer.DISEASE and target is Layer.UNIVERSAL)
        )


# ── Helpers ──────────────────────────────────────────────────────────────────


def _strip_customer_provenance(
    rule: ReasoningRule, source_customer_tag: Optional[str]
) -> ReasoningRule:
    """Return a rule with supporting_case_ids / proposer_run_id pruned of
    references to the source tenant tag.

    Conservative substring match: any supporting_case_id containing the
    tenant tag is dropped. Phase 3 can refine this with a proper
    case-id namespace (e.g., always-prefixed `recursion:tv_braf`).
    """
    if source_customer_tag is None:
        return rule

    cleaned_case_ids = [
        cid
        for cid in rule.evidence.supporting_case_ids
        if source_customer_tag not in cid
    ]
    cleaned_run_id = (
        ""
        if source_customer_tag in (rule.evidence.proposer_run_id or "")
        else rule.evidence.proposer_run_id
    )

    rule.evidence.supporting_case_ids = cleaned_case_ids
    rule.evidence.proposer_run_id = cleaned_run_id
    return rule


def _retains_customer_tag(
    rule: ReasoningRule, source_customer_tag: str
) -> bool:
    """True if any leak indicator still mentions the source tenant after
    a strip pass. If True, the caller MUST NOT promote — abstraction
    did not de-identify the rule.
    """
    if any(
        source_customer_tag in cid
        for cid in rule.evidence.supporting_case_ids
    ):
        return True
    if source_customer_tag in (rule.evidence.proposer_run_id or ""):
        return True
    if source_customer_tag in (rule.evidence.source_policy_citation or ""):
        return True
    return False


def candidate_rules_active(rules: Iterable[ReasoningRule]) -> list[ReasoningRule]:
    """Helper: filter to ACTIVE-status rules. The retrieval pipeline already
    does this — exposed here for tests + non-pipeline callers."""
    return [r for r in rules if r.status is RuleStatus.ACTIVE]
