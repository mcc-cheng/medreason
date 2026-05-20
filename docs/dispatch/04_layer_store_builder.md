# Dispatch — `layer-store-builder`

**From:** architect
**Phase 2 plan:** `/Users/davidzhang/Desktop/Origin/Personal/medreason/docs/targetval_phase2_plan.md` (§Builder 4)
**Working dir:** `/Users/davidzhang/Desktop/Origin/Personal/medreason`
**Python:** `/opt/miniconda3/bin/python`
**Verify command:** `cd /Users/davidzhang/Desktop/Origin/Personal/medreason && /opt/miniconda3/bin/python -m pytest tests/test_targetval_*.py -q`

## Scope

Wire real per-(layer, scope) rule stores behind `LayerRouter`. After
this builder lands, `LayerRouter.ingest_rule` writes, `retrieve_for_case`
reads (unioned across layers, leak-guarded), and `promote_rule` runs
the abstractor + (optional) generalization gate. No back-door write
paths — every write goes through the policy validator.

## Files to edit / create

- `medreason/targetval/layers.py` — replace the
  `_stores_initialised = False` stub with real per-(layer, scope) store
  resolution.
- (new) `medreason/targetval/layer_stores.py` — thin
  `TargetvalRuleStore` Protocol + `open_targetval_rule_store(path)`
  factory that wraps the existing medreason `RuleStore`. Inspect
  `medreason/` for the canonical RuleStore location (tests import it as
  `from medreason.<somewhere> import RuleStore`; reuse, do not
  reimplement). If the existing RuleStore can't be reused directly,
  build a thin in-memory + sqlite-backed implementation that satisfies
  the Protocol — but reuse is strongly preferred.
- `tests/test_targetval_layers.py` — add round-trip tests; REPLACE
  `test_promotion_is_intentionally_unimplemented` with a real promotion
  test.

## Interface signatures

```python
# medreason/targetval/layer_stores.py

class TargetvalRuleStore(Protocol):
    def put(self, rule: ReasoningRule) -> None: ...
    def get(self, rule_id: str) -> Optional[ReasoningRule]: ...
    def list_active(self) -> list[ReasoningRule]: ...
    def set_status(self, rule_id: str, status: RuleStatus) -> None: ...

def open_targetval_rule_store(path: Path) -> TargetvalRuleStore: ...
```

```python
# medreason/targetval/layers.py

class LayerRouter:
    def __init__(
        self,
        paths: LayerStorePaths,
        *,
        store_factory: Callable[[Path], TargetvalRuleStore] = open_targetval_rule_store,
    ):
        self.paths = paths
        self._universal: Optional[TargetvalRuleStore] = None
        self._disease: dict[str, TargetvalRuleStore] = {}
        self._campaign: dict[str, TargetvalRuleStore] = {}
        self._store_factory = store_factory

    def retrieve_for_case(
        self,
        case_disease_scope: str,
        customer_tag: Optional[str],
        top_k: int = 5,
    ) -> list[ReasoningRule]: ...

    def ingest_rule(
        self,
        rule: ReasoningRule,
        *,
        source_layer: Layer,
        customer_tag: Optional[str] = None,
        disease_scope: Optional[str] = None,
    ) -> None: ...

    def promote_rule(
        self,
        rule_id: str,
        *,
        source_layer: Layer,
        target_layer: Layer,
        llm: LLMClient,
        gate: Optional["GeneralizationGate"] = None,
    ) -> Optional[ReasoningRule]: ...
```

Behavioural contract:
- `Path("/dev/null")` MUST remain non-fatal — treat it as
  "open an ephemeral in-memory store" so existing tests using
  `LayerStorePaths(universal=Path("/dev/null"), disease={}, campaign={})`
  still work.
- `retrieve_for_case` unions active rules from UNIVERSAL +
  matching DISEASE (`scope == case_disease_scope`) + matching CAMPAIGN
  (`tenant == customer_tag`). Dedupe by `rule_id`. Truncate to `top_k`.
  No semantic reranking yet — that's Phase 3.
- `promote_rule(CAMPAIGN → UNIVERSAL)` MUST fail. Allowed promotion
  edges: `CAMPAIGN → DISEASE`, `DISEASE → UNIVERSAL`. Anything else
  raises `LayerPolicyViolation`.
- `promote_rule` calls `medreason.extraction.rule_abstractor.abstract_rule(rule, llm)`,
  optionally runs `gate.validate(abstracted_rule)`, and on pass calls
  `ingest_rule(abstracted_rule, source_layer=target_layer, ...)` — so
  the policy validator runs on the destination layer's terms. On gate
  fail or abstraction fail, returns None.

## Test expectations

New tests:
- `test_router_ingest_then_retrieve_universal` (tmp_path-backed).
- `test_router_retrieve_unions_across_layers`.
- `test_router_retrieve_omits_other_tenant_campaign_rules`.
- `test_promote_campaign_to_disease_with_fakellm_passes`.
- `test_promote_universal_rejects_customer_tagged_source`.
- `test_promote_campaign_to_universal_disallowed` (one-hop only).

REPLACE: `test_promotion_is_intentionally_unimplemented` —
no longer applicable; remove it.

Existing tests that MUST still pass:
- `test_universal_layer_rejects_customer_tagged_rule`
- `test_universal_layer_accepts_public_rule`
- `test_disease_layer_rejects_customer_tagged_rule`
- `test_campaign_layer_requires_customer_tag`
- `test_campaign_layer_rejects_wrong_tenant`
- `test_campaign_layer_accepts_correct_tenant`
- `test_router_ingest_rule_validates_policy`
- `test_router_ingest_rule_accepts_public_universal`

The last test uses `Path("/dev/null")` — that path must keep working.

## Dependencies on other builders

- `proposer-builder` writes through `ingest_rule`. Lock this API before
  `proposer-builder` starts. Coordinate via `tester` if signatures
  shift.

## Hard constraints

- No back-door write paths. Every write goes through `ingest_rule` /
  `promote_rule`, both of which run `LayerPolicy.validate_rule`.
- `Path("/dev/null")` stays usable as a no-op store.
- Files under 500 lines — `layers.py` is already ~250; if it grows
  past 450, split helpers into `layer_stores.py` (which you're
  creating anyway).
- Do NOT touch `dashboard/`.
- Do NOT run `git add` / `git commit`.

## When done

SendMessage `tester` with a summary:
- the locked `LayerRouter` write/read/promote API (so
  `proposer-builder` and `mapk-curator` can call it),
- which RuleStore implementation you reused (or, if you had to roll
  a minimal one, the shape),
- pytest count before vs after.
