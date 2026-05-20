# Targetval Phase 2 — Build Plan

**Scope:** Convert the targetval skeleton (43 baseline tests passing as of
2026-05-18) into a functioning pipeline. Phase 2 keeps the constraints:
no real-network ingest, no live LLM calls in tests (use `FakeLLMClient`),
no edits to `dashboard/`, no git commits (the Python tree stays untracked
on purpose). The 3-layer memory split (Universal / Disease / Campaign)
and its leak-guard semantics are **load-bearing** — every builder must
preserve them.

This plan splits Phase 2 across **6 named builder agents**. Each
section below is the brief for one builder.

The hand-off shape between builders is:

```
topology-builder      ─┐
ingest-builder        ─┤
swarm-llm-builder     ─┼─→ proposer-builder → mapk-curator
layer-store-builder   ─┘
```

`swarm-llm-builder` finalises the `TargetMemo` shape that `proposer-builder`
extracts errors from. `layer-store-builder` finalises the `LayerRouter`
write API that `proposer-builder` writes corrective rules into.
`mapk-curator` expands the retro fixture and re-runs the full pipeline
end-to-end with the FakeLLM, so it depends on everyone else.

Every builder MUST report back to `tester` via SendMessage when done.

---

## Hard constraints (apply to every builder)

1. **No real HTTP / API calls.** Anywhere a real fetcher would hit
   Reactome / OpenTargets / ChEMBL / clinicaltrials.gov, ship a
   `Protocol` and a `Fake…Fetcher` with deterministic in-memory data.
   Real network fetchers are explicitly deferred to Phase 3.
2. **Tests use `FakeLLMClient` only.** Never import claude / openai /
   gemini clients in tests. Use `medreason.llm.base.FakeLLMClient` with
   canned `responses=[...]` to drive multi-turn paths deterministically.
3. **Files stay under 500 lines.** If a module is going to push past
   ~450 lines, split (e.g., put bypass-mechanism reasoning in a new
   `topology_signals.py` rather than bloating `topology.py`).
4. **Validate at boundaries.** Any function that takes external input
   (CSV row, JSON dict, LLM response) must validate types and raise a
   typed exception on bad input. Mirror `rule_proposer._build_rule_from_candidate`
   for the pattern: parse → typed errors → caller buckets into rejected.
5. **Universal/Disease layers never see `customer_tag`.** The guard is
   already enforced by `LayerPolicy.validate_rule`. Don't add a back-door
   path: any new write site goes through `LayerRouter.ingest_rule`.
6. **No edits to `dashboard/`** (Next.js production app, unrelated to
   targetval Python tree).
7. **No `git commit` / `git add`.** The Python tree is intentionally
   untracked per user instruction.
8. **The 43 baseline tests must still pass after every builder's work.**
   Run `cd /Users/davidzhang/Desktop/Origin/Personal/medreason &&
   /opt/miniconda3/bin/python -m pytest tests/test_targetval_*.py` to
   verify.

---

## Builder 1 — `topology-builder`

**Goal:** Make `derive_bypass_signals` rich enough to feed a real
`predicted_bypass` decision, and add a serialisable `PathwayCache`
snapshot so swarm runs reproduce.

### Files to edit

- `medreason/targetval/topology.py` — extend, do not rewrite.
- (new) `medreason/targetval/topology_signals.py` — split off if
  `topology.py` would exceed ~400 lines.
- `tests/test_targetval_topology.py` — add cases.

### Interface signatures

```python
# medreason/targetval/topology.py

@dataclass(frozen=True)
class BypassSignal:
    mechanism: str           # one of BypassMechanism.value strings
    strength: float          # [0, 1]
    evidence_summary: str
    contributing_features: tuple[str, ...] = ()  # NEW: ("paralog_count=2", ...)

@dataclass
class PathwayCache:
    paralogs_by_gene: dict[str, list[str]] = field(default_factory=dict)
    family_by_gene: dict[str, str] = field(default_factory=dict)
    feedback_loops_by_gene: dict[str, list[str]] = field(default_factory=dict)
    downstream_redundancy: dict[str, float] = field(default_factory=dict)
    # NEW
    upstream_nodes_by_gene: dict[str, list[str]] = field(default_factory=dict)
    downstream_nodes_by_gene: dict[str, list[str]] = field(default_factory=dict)
    cache_version: str = "v0.1"  # snapshot id, written into PathwayTopologyEvidence.reference_pathway

    def to_json(self) -> str: ...
    @classmethod
    def from_json(cls, payload: str) -> "PathwayCache": ...
    def write(self, path: str | Path) -> Path: ...
    @classmethod
    def load(cls, path: str | Path) -> "PathwayCache": ...

def derive_bypass_signals(
    gene_symbol: str,
    cache: PathwayCache,
    *,
    modality: Optional["Modality"] = None,  # NEW: filter signals by modality
) -> list[BypassSignal]: ...

def dominant_bypass_mechanism(
    signals: list[BypassSignal],
) -> "BypassMechanism":
    """Return the highest-strength signal as a BypassMechanism enum,
    or BypassMechanism.NO_BYPASS_KNOWN if list is empty / all-zero."""
```

- The modality filter: PROTAC + small molecule see paralog and feedback
  signals; antibodies see paralog + alternative-pathway but NOT
  intracellular feedback; ASO/siRNA see paralog + downstream-redundancy.
  Default (`modality=None`) keeps current behavior (all signals).
- `dominant_bypass_mechanism` maps the free-text `mechanism` strings to
  the `BypassMechanism` enum (the strings already match `.value`).
- `to_json`/`from_json` use a deterministic key order so the snapshot
  hashes the same on every machine.

### Test plan

New tests in `test_targetval_topology.py`:

- `test_modality_filter_drops_feedback_for_antibody` — pass `modality=Modality.ANTIBODY`, assert no `downstream_feedback` signal.
- `test_pathway_cache_roundtrip_json` — write, read back, assert equality.
- `test_pathway_cache_file_roundtrip` — write + load via `tmp_path`.
- `test_dominant_bypass_mechanism_empty_returns_no_bypass`.
- `test_dominant_bypass_mechanism_picks_highest_strength`.
- `test_bypass_signal_contributing_features_populated` — assert that the
  paralog signal carries `("paralog_count=N",)`.

Existing topology tests must still pass unchanged.

### Dependencies on other builders

None — `topology-builder` is upstream of everyone. Run first.

### Hard constraints reminder

- No external HTTP.
- Keep `topology.py` under 500 lines; split into `topology_signals.py`
  if needed.
- Do NOT import LLM clients — pure-Python deterministic logic only.

---

## Builder 2 — `ingest-builder`

**Goal:** Wire `build_evidence_bundle` end-to-end with the real (still
in-memory) cache + fake fetchers, and finish the customer-CSV ingest
path with rigorous validation.

### Files to edit

- `medreason/targetval/evidence_ingest.py` — add a `build_default_pipeline`
  factory and harden parsing.
- `medreason_bench/targetval/recursion_ingest.py` — add typed errors
  for malformed rows, expose a `IngestReport` so callers see rejected
  rows like `ProposalResult.rejected`.
- (new) `medreason_bench/targetval/customer_csv.py` — Parquet/JSONL
  parser helpers, kept separate so `recursion_ingest.py` stays small.
- `tests/test_targetval_recursion_ingest.py` — add cases for rejected
  rows.
- (new) `tests/test_targetval_evidence_ingest.py` — test the new
  `build_evidence_bundle` wiring with fakes.

### Interface signatures

```python
# medreason/targetval/evidence_ingest.py

@dataclass
class EvidencePipeline:
    """Holds wired fetchers + a cache. One pipeline serves a whole campaign."""
    genetics: Optional[GeneticsFetcher] = None
    knockout: Optional[KnockoutFetcher] = None
    topology: Optional[TopologyFetcher] = None
    trials: Optional[TrialsFetcher] = None

    def fetch(
        self, target: TargetID, disease: DiseaseContext,
        *, internal: Optional[InternalEvidence] = None,
    ) -> EvidenceBundle: ...

def build_default_pipeline(
    cache: PathwayCache,
    *,
    genetics_table: Optional[dict[str, GeneticsEvidence]] = None,
    knockout_table: Optional[dict[str, KnockoutDependenceEvidence]] = None,
    trials_table: Optional[dict[str, PriorTrialEvidence]] = None,
) -> EvidencePipeline:
    """Wire Fake* fetchers + CachedTopologyFetcher. The standard offline
    pipeline tests use end-to-end."""
```

```python
# medreason_bench/targetval/recursion_ingest.py

class IngestError(ValueError):
    """A customer row could not be parsed."""

@dataclass
class IngestReport:
    cases: list[TargetValidationCase] = field(default_factory=list)
    rejected: list[tuple[dict, str]] = field(default_factory=list)

    @property
    def n_cases(self) -> int: ...
    @property
    def n_rejected(self) -> int: ...

def load_customer_targets(
    targets_iter: Iterable[dict[str, Any]],
    *,
    customer_tag: str,
    outcomes_iter: Optional[Iterable[dict[str, Any]]] = None,
    strict: bool = False,
) -> IngestReport:  # CHANGED return type — was list[TargetValidationCase]
    """When strict=True, raise IngestError on the first bad row.
    Otherwise bucket bad rows into IngestReport.rejected."""
```

**Backwards-compat note for `ingest-builder`:** Tests currently assert
`load_customer_targets(...)` returns a `list`. Either:
(a) keep the list return and add a sibling `load_customer_targets_report`, OR
(b) change to `IngestReport` and update the 5 existing tests in lockstep.
Option (b) is preferred — change the return type, update tests, fewer APIs
to maintain. Document the decision in the module docstring.

```python
# medreason_bench/targetval/customer_csv.py (new)

def parse_jsonl(path: str | Path) -> list[dict[str, Any]]: ...
def parse_parquet(path: str | Path) -> list[dict[str, Any]]:
    """Lazy-imports pyarrow. Raises ImportError with a helpful message
    if pyarrow is not installed."""
```

### Test plan

- `test_ingest_report_buckets_rejected_rows` — feed a row missing
  `case_id`, assert `report.n_rejected == 1`.
- `test_ingest_strict_raises_on_bad_row`.
- `test_evidence_pipeline_assembles_from_fakes` — wire pipeline, fetch
  for `BRAF`, assert non-empty `topology.paralogs`.
- `test_build_default_pipeline_with_empty_cache_returns_empty_topology`.
- `test_parse_jsonl_roundtrip` (in tmp_path).

Existing recursion-ingest tests must pass after the return-type migration.

### Dependencies on other builders

- Reads `PathwayCache` API from `topology-builder` (specifically
  the new `cache_version` field). Wait for `topology-builder` to finish
  before adding the snapshot wiring.

### Hard constraints reminder

- No HTTP.
- Validate every external dict at the boundary; use typed errors.
- Parquet helper must lazy-import.

---

## Builder 3 — `swarm-llm-builder`

**Goal:** Replace the placeholder `TargetMemo` with a real LLM-driven
memo with strict JSON parsing. The output is the `proposer-builder`'s
input, so the schema lock matters.

### Files to edit

- `medreason/targetval/swarm.py` — extend `SwarmAgent.run`, finalize
  `TargetMemo` shape.
- (new) `medreason/targetval/swarm_prompts.py` — system + user prompt
  builders. Kept separate from `swarm.py` for readability.
- (new) `medreason/targetval/swarm_parsing.py` — JSON extractor +
  repair, modelled on `leadop/agent_llm.py`'s `_extract_json` and
  `_repair_*` helpers.
- `tests/test_targetval_swarm.py` — add parse-path tests using
  `FakeLLMClient(responses=[...])`.

### Interface signatures

```python
# medreason/targetval/swarm.py

@dataclass
class TargetMemo:
    case_id: str
    gene_symbol: str
    priority_score: float            # [0, 1]
    bypass_risk_score: float         # [0, 1]
    predicted_bypass: BypassMechanism = BypassMechanism.UNKNOWN
    supporting_evidence: list[str] = field(default_factory=list)
    weakening_evidence: list[str] = field(default_factory=list)
    proposed_experiments: list[str] = field(default_factory=list)
    rationale: str = ""
    retrieved_rule_ids: list[str] = field(default_factory=list)
    applied_rule_ids: list[str] = field(default_factory=list)
    # NEW for proposer-builder consumption:
    bypass_signals_seen: list[str] = field(default_factory=list)  # mechanism strings
    parse_warnings: list[str] = field(default_factory=list)        # non-fatal issues
    cost_usd: float = 0.0
    seed: int = 0
```

```python
# medreason/targetval/swarm_prompts.py

SYSTEM_PROMPT_TARGETVAL: str  # constant string, locked

def build_user_prompt(
    case: TargetValidationCase,
    *,
    retrieved_rules: list[ReasoningRule],
    bypass_signals: list[BypassSignal],
) -> str: ...
```

```python
# medreason/targetval/swarm_parsing.py

class MemoParseError(ValueError): ...

def extract_memo_json(text: str) -> dict:
    """Mirror leadop/agent_llm.py:_extract_json. Accept fenced code blocks,
    fall back to the first balanced {…}. Raises MemoParseError if no JSON."""

def parse_memo(
    text: str,
    *,
    case_id: str,
    gene_symbol: str,
    retrieved_rule_ids: list[str],
    bypass_signals: list[BypassSignal],
    cost_usd: float,
    seed: int,
) -> TargetMemo:
    """Build a TargetMemo from raw LLM text.
    Tolerant: missing fields default; unknown bypass strings → UNKNOWN
    with a parse_warning. Score values are clamped to [0,1]."""
```

```python
# medreason/targetval/swarm.py (SwarmAgent.run updated)

class SwarmAgent:
    def run(self) -> TargetMemo:
        bypass_signals = derive_bypass_signals(
            self.case.target.gene_symbol,
            self._pathway_cache,  # NEW kwarg on SwarmAgent
            modality=self.case.modality,
        )
        retrieved = self.router.retrieve_for_case(...)
        system = SYSTEM_PROMPT_TARGETVAL
        user = build_user_prompt(self.case, retrieved_rules=retrieved,
                                  bypass_signals=bypass_signals)
        response = self.llm.complete(system=system, user=user, seed=self.seed)
        return parse_memo(response.text, ...)
```

- `SwarmAgent.__init__` gains an optional `pathway_cache: PathwayCache`
  kwarg. Default to an empty cache so tests that don't care still work.
- `SwarmRunner.__init__` likewise gains `pathway_cache`. It's
  threaded through to each agent.

### Test plan

- `test_parse_memo_happy_path` — canned LLM JSON with all fields → memo.
- `test_parse_memo_fenced_json` — fenced ```json``` blocks parse cleanly.
- `test_parse_memo_clamps_scores` — `priority_score=1.5` clamps to 1.0.
- `test_parse_memo_unknown_bypass_warns` — unknown mechanism → UNKNOWN
  + `parse_warnings` non-empty.
- `test_parse_memo_garbage_raises_memo_parse_error`.
- `test_swarm_agent_uses_bypass_signals_in_prompt` — wire
  PathwayCache(BRAF=…), inspect `llm.calls` for the strength language.

Existing swarm tests (`test_swarm_runs_one_agent_per_case`,
`test_swarm_parallel_path_same_results_as_serial`, etc.) must still
pass — they use empty cache + default_text="{}", which should still
yield a default-shaped TargetMemo (scores 0.0 or whatever the parser
defaults to). Update the existing tests' expectations as needed (e.g.,
they may currently expect `priority_score == 0.5`; loosen to "is a float
in [0,1]").

### Dependencies on other builders

- Reads `PathwayCache` shape (incl. `to_json`) from `topology-builder`.
- Locked output schema is read by `proposer-builder`.

### Hard constraints reminder

- `FakeLLMClient` only in tests — never wire `ClaudeLLMClient`.
- Strict-JSON parser must NEVER call `eval`. Use `json.loads` and regex,
  per leadop's `_extract_json` pattern.
- Keep `swarm.py` under 500 lines by moving prompts + parsing into the
  two new modules.

---

## Builder 4 — `layer-store-builder`

**Goal:** Wire actual `RuleStore` instances behind `LayerRouter` so
`ingest_rule` writes a rule, `retrieve_for_case` returns it, and
`promote_rule` runs abstractor + gate. The 3-layer leak guard must
remain bulletproof — no back-door writes.

### Files to edit

- `medreason/targetval/layers.py` — replace the stubbed
  `_stores_initialised` flag with a real per-(layer, scope) store map.
- (new) `medreason/targetval/layer_stores.py` — thin wrapper around
  the existing `RuleStore` (which lives in `medreason/ontology/`-ish or
  `medreason/storage/`; check the import path used by tests like
  `test_rule_store.py`). Keep the wrapper minimal: provide
  `put / get / list / set_status / iter_active`.
- `tests/test_targetval_layers.py` — add round-trip tests.

### Interface signatures

```python
# medreason/targetval/layer_stores.py

class TargetvalRuleStore(Protocol):
    """Subset of the existing RuleStore API the LayerRouter relies on."""
    def put(self, rule: ReasoningRule) -> None: ...
    def get(self, rule_id: str) -> Optional[ReasoningRule]: ...
    def list_active(self) -> list[ReasoningRule]: ...
    def set_status(self, rule_id: str, status: RuleStatus) -> None: ...

def open_targetval_rule_store(path: Path) -> TargetvalRuleStore:
    """Open or create a sqlite-backed rule store at `path`. Uses the
    existing medreason RuleStore implementation; this function just
    wraps construction."""
```

```python
# medreason/targetval/layers.py

class LayerRouter:
    def __init__(self, paths: LayerStorePaths,
                  *, store_factory: Callable[[Path], TargetvalRuleStore] = open_targetval_rule_store):
        # Lazily-instantiated per-(layer, scope) stores
        self._universal: Optional[TargetvalRuleStore] = None
        self._disease: dict[str, TargetvalRuleStore] = {}
        self._campaign: dict[str, TargetvalRuleStore] = {}
        self._store_factory = store_factory
        self.paths = paths

    def retrieve_for_case(
        self,
        case_disease_scope: str,
        customer_tag: Optional[str],
        top_k: int = 5,
    ) -> list[ReasoningRule]:
        """Union active rules from UNIVERSAL + matching DISEASE + matching
        CAMPAIGN. Deduped by rule_id. No reranking in v0.2 — semantic
        rerank is Phase 3."""

    def ingest_rule(
        self,
        rule: ReasoningRule,
        *,
        source_layer: Layer,
        customer_tag: Optional[str] = None,
        disease_scope: Optional[str] = None,
    ) -> None:
        """Validate policy then write to the resolved store."""

    def promote_rule(
        self,
        rule_id: str,
        *,
        source_layer: Layer,
        target_layer: Layer,
        llm: LLMClient,
        gate: Optional["GeneralizationGate"] = None,
    ) -> Optional[ReasoningRule]:
        """Load rule from source_layer; abstract via abstract_rule(); if
        gate is provided, run gate.validate(); on pass, write to target_layer
        store with the appropriate policy.
        Returns the promoted rule or None if abstraction/gate rejects it."""
```

- The `paths.universal == Path("/dev/null")` path used by existing tests
  must remain non-fatal. Treat `Path("/dev/null")` as "no store" — lazy
  init returns a no-op in-memory store that records `put`/`get` calls
  for tests. This preserves the existing `test_router_ingest_rule_accepts_public_universal`
  test which uses `/dev/null`.
- `promote_rule` no longer raises `NotImplementedError`. The existing
  `test_promotion_is_intentionally_unimplemented` test MUST be replaced
  with a real promotion test — coordinate the test rename with `tester`.

### Test plan

- `test_router_ingest_then_retrieve_universal` — tmp_path-backed store,
  ingest a universal rule, retrieve_for_case returns it.
- `test_router_retrieve_unions_across_layers` — write a universal rule
  + a disease rule (scope="oncology_mapk") + a campaign rule
  (tenant="recursion"); retrieve_for_case(scope="oncology_mapk",
  customer_tag="recursion") returns all three.
- `test_router_retrieve_omits_other_tenant_campaign_rules` — a campaign
  rule for tenant A must NOT show up in a retrieve call with tenant=B.
- `test_promote_campaign_to_disease_with_fakellm_passes` — wire a fake
  abstractor + a passing gate, assert rule lands in disease layer with
  customer_tag stripped from supporting_case_ids.
- `test_promote_universal_rejects_customer_tagged_source` — promoting a
  CAMPAIGN rule directly to UNIVERSAL must fail (you have to go through
  DISEASE first; this is the moat).
- Replace `test_promotion_is_intentionally_unimplemented` with the above.
- All other existing layer tests must still pass.

### Dependencies on other builders

- `proposer-builder` writes through `ingest_rule`, so the API must be
  locked before `proposer-builder` finishes. Coordinate via `tester`.

### Hard constraints reminder

- NO back-door writes. The only path to a store is `ingest_rule` /
  `promote_rule`, both of which run `LayerPolicy.validate_rule`.
- `Path("/dev/null")` must remain a usable no-op for callers that don't
  want persistence.
- Keep `layers.py` under 500 lines; split into `layer_stores.py` if it
  starts to crowd.

---

## Builder 5 — `proposer-builder`

**Goal:** Replace `propose_corrective_rules`'s `return []` stub with a
real LLM-driven proposer that emits CANDIDATE `ReasoningRule`s for each
`SystematicError`. End-to-end, this is the moat — the cross-agent
analyzer turning systematic miss-patterns into universal-layer rules.

### Files to edit

- `medreason/targetval/cross_agent_analyzer.py` — implement
  `propose_corrective_rules`.
- (new) `medreason/targetval/cross_agent_prompts.py` — the prompt
  builder for cross-case rules.
- `tests/test_targetval_cross_agent.py` — add proposer tests.

### Interface signatures

```python
# medreason/targetval/cross_agent_analyzer.py

@dataclass(frozen=True)
class CorrectiveRuleCandidate:
    """Wraps a ReasoningRule plus the SystematicError it was extracted
    from, so the LayerRouter can record the provenance link."""
    rule: ReasoningRule
    source_error: SystematicError

def propose_corrective_rules(
    errors: list[SystematicError],
    cases: list[TargetValidationCase],
    llm: LLMClient,
    *,
    proposer_run_id: Optional[str] = None,
    severity_floor: float = _DEFAULT_SEVERITY_FLOOR,
    seed: int = 0,
) -> list[ReasoningRule]:
    """For each SystematicError with severity >= floor:
       1. Render a de-identified prompt: list affected case_ids (opaque),
          the swarm's average bypass_risk_score, the ground-truth pattern.
          DO NOT include any InternalEvidence.readouts content — even
          for retro mode where there is no customer data, the prompt
          builder must filter it.
       2. Call llm.complete(...) with the cross_agent_proposer prompt.
       3. Parse JSON output. For each candidate dict, build a
          ReasoningRule with:
             - rule_id auto-generated
             - status=RuleStatus.CANDIDATE
             - evidence.proposer_model = llm.model_version
             - evidence.proposer_run_id = proposer_run_id
             - evidence.supporting_case_ids = list(error.affected_case_ids)
             - evidence.source_policy_citation = f"cross_agent:{error.error_kind}:n={len(error.affected_case_ids)}"
             - trigger.semantic_predicate = parsed predicate text
             - action = parsed action text (≤25 words enforced, same
               as rule_proposer)
       4. Reject candidates that violate action-length / patient-id /
          empty-predicate (reuse rule_proposer._has_patient_identifier
          and _count_words; do NOT depend on LCD policy citation
          validation since cross_agent rules don't reference policies).
    Returns list[ReasoningRule]. (The dataclass wrapper is internal —
    public return stays list[ReasoningRule] so existing callers don't
    break.)"""
```

- Reuse `rule_proposer._count_words`, `_has_patient_identifier`,
  `ACTION_MAX_WORDS`. They are public-enough to import.
- The candidate `RuleTrigger` for cross-agent rules is intentionally
  free of CPT/ICD/payer fields (those are prior-auth ontology). Only
  `semantic_predicate` is populated. The Tier-1 ontology prefilter
  will return `True` for these rules in the targetval retrieval path
  (no structural filters means "match any targetval case").
- Patient-ID scan still runs because cross_agent rules go into the
  UNIVERSAL layer and must not regurgitate identifiers from
  `InternalEvidence`.

### Test plan

- `test_proposer_emits_no_rules_for_empty_errors`.
- `test_proposer_emits_rule_for_missed_bypass` — `FakeLLMClient(
  responses=['{"rules": [{"semantic_predicate": "...", "action": "...",
  "rationale": "...", "polarity": "requires_check"}]}'])`, assert one
  rule comes back with status=CANDIDATE.
- `test_proposer_rejects_overlong_action` — canned LLM emits a 40-word
  action → rejected.
- `test_proposer_rejects_patient_identifier_leak` — LLM emits
  "patient John ..." → rejected.
- `test_proposer_supporting_case_ids_come_from_affected` — round-trip
  the case_ids.
- `test_proposer_skips_below_severity_floor` — sev=0.2 with floor=0.5
  → no rule.
- `test_run_cross_agent_analysis_returns_candidate_rules_now` — replace
  the existing `analysis.candidate_rules == []` assertion with `len(
  analysis.candidate_rules) >= 1` when given a wired fake LLM.

The existing `test_propose_corrective_rules_skeleton_returns_empty` test
must be REPLACED (not left passing) — its assertion stops being true.
Coordinate with `tester` so the suite still passes.

### Dependencies on other builders

- `TargetMemo.bypass_signals_seen` from `swarm-llm-builder` — used in
  the prompt to remind the analyzer what features the swarm DID see
  but failed to act on.
- `LayerRouter.ingest_rule` from `layer-store-builder` — `proposer-builder`
  doesn't call it directly (that's `mapk-curator`'s job in the end-to-end
  pipeline), but the rule shape it emits must satisfy
  `LayerPolicy.validate_rule(rule, customer_tag=None)` for the
  UNIVERSAL layer.

### Hard constraints reminder

- `FakeLLMClient` only in tests, with `responses=[...]` driving each
  scenario.
- NO `InternalEvidence.readouts` text in the prompt — filter explicitly.
- NEVER pass a `customer_tag` into the corrective-rule provenance.
  These are universal rules.

---

## Builder 6 — `mapk-curator`

**Goal:** Expand `build_mapk_retro_seed` from 3 illustrative cases to
the full ~20-30 documented MAPK targets, and write an integration test
that exercises the full pipeline (build cases → run swarm with
FakeLLM → detect errors → propose corrective rules → ingest into
universal layer) end-to-end.

### Files to edit

- `medreason_bench/targetval/mapk_retro.py` — expand `build_mapk_retro_seed`.
- (new) `medreason_bench/targetval/mapk_retro_data.py` — the raw
  table of curated entries, kept separate so `mapk_retro.py` stays a
  thin builder under 500 lines.
- (new) `tests/test_targetval_end_to_end.py` — the integration test.
- `medreason_bench/targetval/__main__.py` — wire the
  `data build-mapk-retro` and `swarm dryrun --campaign mapk_retro`
  paths to use the expanded set.

### Interface signatures

```python
# medreason_bench/targetval/mapk_retro_data.py

@dataclass(frozen=True)
class MapkRetroEntry:
    case_id: str
    gene: str
    family: str
    disease: str
    paralogs: tuple[str, ...]
    paralog_count: int
    redundancy: float
    feedback_loops: tuple[str, ...]
    n_trials: int
    n_p2_fail: int
    n_approvals: int
    trial_summary: str
    outcome: GroundTruthOutcome
    bypass: BypassMechanism
    notes: str = ""

MAPK_RETRO_ENTRIES_V0_2: tuple[MapkRetroEntry, ...]
# Exactly 20 entries minimum, drawn from the working list in
# mapk_retro.py's module docstring (BRAF/MEK/ERK/KRAS/NRAS/HRAS/SHP2/
# RAF-dimerization/MET/ALK/FGFR/AXL/DDR/CDK4-6/MDM2/WEE1/PI3K-alpha
# etc.). Each entry's outcome + bypass MUST be defensible from
# clinicaltrials.gov + a primary-lit citation in `notes`.
```

```python
# medreason_bench/targetval/mapk_retro.py (refactor)

MAPK_RETRO_CAMPAIGN_ID = "mapk_retro_v0.2"  # bumped

def build_mapk_retro_seed(
    *,
    version: str = "v0.2",
) -> list[TargetValidationCase]:
    if version == "v0.1":
        # Preserve legacy 3-entry seed for back-compat with existing tests.
        return _legacy_v0_1_seed()
    return [_entry_from(e) for e in MAPK_RETRO_ENTRIES_V0_2]

def _entry_from(e: MapkRetroEntry) -> TargetValidationCase: ...
```

```python
# tests/test_targetval_end_to_end.py (new)

def test_full_pipeline_mapk_retro_with_fakellm():
    """Build mapk retro (≥20 cases) → swarm.run with FakeLLMClient
    emitting canned 'underestimates bypass' memos → detect_systematic_errors
    fires ≥1 error → propose_corrective_rules emits ≥1 candidate rule
    → LayerRouter.ingest_rule writes it to UNIVERSAL → retrieve_for_case
    returns it. No real LLM, no network. Asserts:
      - report.n_targets >= 20
      - len(analysis.systematic_errors) >= 1
      - len(analysis.candidate_rules) >= 1
      - retrieved = router.retrieve_for_case(scope="oncology_mapk",
                                              customer_tag=None)
      - any(r.rule_id == proposed.rule_id for r in retrieved)
    """

def test_mapk_retro_v0_2_size():
    cases = build_mapk_retro_seed(version="v0.2")
    assert len(cases) >= 20
    # Every case has a valid bypass label (no UNKNOWN in v0.2 retro)
    assert all(c.ground_truth_bypass is not BypassMechanism.UNKNOWN for c in cases)

def test_mapk_retro_v0_1_legacy_still_works():
    cases = build_mapk_retro_seed(version="v0.1")
    assert len(cases) == 3  # the original three
```

### Test plan

- The three tests above.
- All existing tests must still pass (especially the existing
  `test_targetval_recursion_ingest.py` / metrics tests that import the
  synthetic + legacy fixtures).

### Dependencies on other builders

- ALL of `topology-builder`, `ingest-builder`, `swarm-llm-builder`,
  `layer-store-builder`, `proposer-builder`. `mapk-curator` is the last
  to land. Wait for `tester` to confirm the other five passed before
  starting the integration test.

### Hard constraints reminder

- Every entry needs a defensible source — write a 1-sentence rationale
  in `MapkRetroEntry.notes`. No fabricated outcomes.
- `FakeLLMClient` for the integration test. The integration test must
  run in well under 10s on a laptop.
- Don't load Reactome flat files. Pre-populate a `PathwayCache` in the
  integration test by hand for the 20 genes (paralog families are the
  expensive part; hard-code them).

---

## Cross-builder coordination contract (read this if you're `tester`)

- After each builder reports done, run the full targetval test suite:
  `cd /Users/davidzhang/Desktop/Origin/Personal/medreason && /opt/miniconda3/bin/python -m pytest tests/test_targetval_*.py -q`
- Expected counts:
  - After `topology-builder`: 43 + ~6 new = ~49 tests.
  - After `ingest-builder`: ~49 + ~5 new = ~54 tests.
  - After `swarm-llm-builder`: ~54 + ~5 new (minus tolerance changes
    in existing swarm tests) = ~58 tests.
  - After `layer-store-builder`: ~58 + ~5 new (minus 1 replaced
    `test_promotion_is_intentionally_unimplemented`) = ~62 tests.
  - After `proposer-builder`: ~62 + ~5 new (minus 1 replaced
    `test_propose_corrective_rules_skeleton_returns_empty`) = ~66 tests.
  - After `mapk-curator`: ~66 + 3 new = ~69 tests.
- If any builder breaks an existing test that wasn't on the
  "intentionally replaced" list above, that's a regression — `tester`
  bounces it back to the builder.

End of plan.
