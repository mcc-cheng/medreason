# Dispatch — `ingest-builder`

**From:** architect
**Phase 2 plan:** `/Users/davidzhang/Desktop/Origin/Personal/medreason/docs/targetval_phase2_plan.md` (§Builder 2)
**Working dir:** `/Users/davidzhang/Desktop/Origin/Personal/medreason`
**Python:** `/opt/miniconda3/bin/python`
**Verify command:** `cd /Users/davidzhang/Desktop/Origin/Personal/medreason && /opt/miniconda3/bin/python -m pytest tests/test_targetval_*.py -q`

## Scope

1. Add an `EvidencePipeline` orchestrator + `build_default_pipeline`
   factory in `medreason/targetval/evidence_ingest.py` so callers wire
   fakes once and re-use them across cases.
2. Harden `recursion_ingest.load_customer_targets` with typed errors
   and an `IngestReport` return type. Existing tests must be updated in
   lockstep with the return-type change.
3. Add a JSONL/Parquet helper module so `recursion_ingest.py` stays small.

## Files to edit / create

- `medreason/targetval/evidence_ingest.py` — add `EvidencePipeline`,
  `build_default_pipeline`.
- `medreason_bench/targetval/recursion_ingest.py` — return `IngestReport`,
  add `IngestError`, add `strict` kwarg.
- (new) `medreason_bench/targetval/customer_csv.py` — `parse_jsonl`,
  `parse_parquet` (lazy-import pyarrow).
- `tests/test_targetval_recursion_ingest.py` — migrate existing 5 tests
  to the new return type, add rejected-row coverage.
- (new) `tests/test_targetval_evidence_ingest.py`.

## Interface signatures

```python
# medreason/targetval/evidence_ingest.py

@dataclass
class EvidencePipeline:
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
) -> EvidencePipeline: ...
```

```python
# medreason_bench/targetval/recursion_ingest.py

class IngestError(ValueError): ...

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
) -> IngestReport:  # CHANGED return type
    ...
```

Update the 5 existing recursion_ingest tests:
- They call `cases = load_customer_targets(...)` and treat the result
  as a list. After this change, `report = load_customer_targets(...)`
  and `cases = report.cases`. Make this swap in the existing tests as
  part of this builder's work (they're listed in
  `tests/test_targetval_recursion_ingest.py`).

```python
# medreason_bench/targetval/customer_csv.py (new)
def parse_jsonl(path: str | Path) -> list[dict[str, Any]]: ...
def parse_parquet(path: str | Path) -> list[dict[str, Any]]:
    """Lazy import pyarrow; ImportError with helpful message if missing."""
```

## Test expectations

New tests:
- `test_ingest_report_buckets_rejected_rows` — missing `case_id` → 1 reject.
- `test_ingest_strict_raises_on_bad_row`.
- `test_evidence_pipeline_assembles_from_fakes`.
- `test_build_default_pipeline_with_empty_cache_returns_empty_topology`.
- `test_parse_jsonl_roundtrip` (tmp_path).

Existing tests (after migration to IngestReport):
- `test_ingest_minimal_target_row`
- `test_ingest_with_outcomes_marks_retrospective`
- `test_ingest_unknown_modality_falls_back`
- `test_ingest_handles_unparseable_readouts_json`
- `test_ingest_handles_dict_readouts`

All must pass with the new shape.

## Dependencies on other builders

- Wait for `topology-builder` to land `PathwayCache.cache_version` and
  `to_json/from_json`. Use those in `EvidencePipeline` wiring (the
  CachedTopologyFetcher already reads from the cache; just be sure
  not to recreate the cache shape — import it).

## Hard constraints

- No HTTP.
- Validate every external dict at the boundary with typed errors.
- Parquet helper lazy-imports `pyarrow` (NOT a top-level import).
- Do NOT touch `dashboard/`.
- Do NOT run `git add` / `git commit`.

## When done

SendMessage `tester` with a summary:
- new + migrated tests,
- pytest count before vs after,
- whether the return-type migration broke any caller (it shouldn't —
  recursion_ingest's only callers in-tree are tests + `__main__.py`'s
  ingest-customer command, both of which you update).
