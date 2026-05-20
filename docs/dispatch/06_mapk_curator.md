# Dispatch — `mapk-curator`

**From:** architect
**Phase 2 plan:** `/Users/davidzhang/Desktop/Origin/Personal/medreason/docs/targetval_phase2_plan.md` (§Builder 6)
**Working dir:** `/Users/davidzhang/Desktop/Origin/Personal/medreason`
**Python:** `/opt/miniconda3/bin/python`
**Verify command:** `cd /Users/davidzhang/Desktop/Origin/Personal/medreason && /opt/miniconda3/bin/python -m pytest tests/test_targetval_*.py -q`

## Scope

1. Expand `build_mapk_retro_seed()` from 3 illustrative cases to ≥20
   curated MAPK targets with defensible outcome + bypass labels.
2. Write `test_full_pipeline_mapk_retro_with_fakellm` — the integration
   test that exercises every builder's work end-to-end.

You are the last builder to land. Wait for `tester` to confirm the
other 5 builders passed before starting.

## Files to edit / create

- `medreason_bench/targetval/mapk_retro.py` — refactor to read entries
  from a new data module; keep `MAPK_RETRO_CAMPAIGN_ID` (bump to
  `v0.2`). Preserve a `_legacy_v0_1_seed()` so the existing 3-case
  shape is accessible via `version="v0.1"`.
- (new) `medreason_bench/targetval/mapk_retro_data.py` — the
  `MapkRetroEntry` dataclass + `MAPK_RETRO_ENTRIES_V0_2` tuple.
- (new) `tests/test_targetval_end_to_end.py` — the integration test.
- `medreason_bench/targetval/__main__.py` — wire `swarm dryrun
  --campaign mapk_retro` to use the expanded set.

## Interface signatures

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
    notes: str = ""  # MUST cite a defensible source (CT.gov ID or PMID)

MAPK_RETRO_ENTRIES_V0_2: tuple[MapkRetroEntry, ...]  # ≥ 20 entries
```

Coverage requirement: the 20+ entries must include at least one
representative for each of these axes (from the working list in
`mapk_retro.py`'s docstring):

- BRAF V600E melanoma (approved)
- BRAF V600E CRC (failed)
- MEK1/2
- ERK1/2
- KRAS G12C NSCLC (approved)
- KRAS G12C CRC (sub-par)
- NRAS melanoma
- ALK NSCLC (approved)
- FGFR cholangiocarcinoma (approved)
- CDK4/6 breast cancer (approved)
- PI3K-alpha breast cancer (approved)
- WEE1 solid tumors
- MET NSCLC bypass
- AXL TNBC (failed)
- SHP2 (PTPN11)

Total ≥ 20. Mix of approved + failed + mixed; mix of bypass mechanisms.

```python
# medreason_bench/targetval/mapk_retro.py (refactor)

MAPK_RETRO_CAMPAIGN_ID = "mapk_retro_v0.2"

def build_mapk_retro_seed(*, version: str = "v0.2") -> list[TargetValidationCase]:
    if version == "v0.1":
        return _legacy_v0_1_seed()
    return [_entry_from(e) for e in MAPK_RETRO_ENTRIES_V0_2]

def _entry_from(e: MapkRetroEntry) -> TargetValidationCase: ...
def _legacy_v0_1_seed() -> list[TargetValidationCase]: ...
```

## End-to-end test contract

```python
# tests/test_targetval_end_to_end.py

def test_full_pipeline_mapk_retro_with_fakellm(tmp_path):
    # 1) Build mapk retro v0.2 (≥20 cases).
    cases = build_mapk_retro_seed(version="v0.2")
    assert len(cases) >= 20

    # 2) Wire a PathwayCache pre-populated for these 20 genes.
    cache = PathwayCache(
        paralogs_by_gene={ "BRAF": ["ARAF", "RAF1"], ... },
        ...
    )

    # 3) FakeLLMClient that returns memos under-scoring bypass risk
    #    on the failed cases, so detect_systematic_errors fires.
    canned_memos = [
        '{"priority_score": 0.4, "bypass_risk_score": 0.2, ...}'
        for _ in cases
    ]
    swarm_llm = FakeLLMClient(responses=list(canned_memos))

    # 4) LayerRouter on tmp_path (real stores).
    router = LayerRouter(LayerStorePaths(
        universal=tmp_path / "universal.db",
        disease={"oncology_mapk": tmp_path / "disease_mapk.db"},
        campaign={},
    ))

    # 5) Run the swarm.
    runner = SwarmRunner(swarm_llm, router, max_workers=1, pathway_cache=cache)
    report = runner.run(cases, campaign_id="mapk_retro_v0.2", seed=11)
    assert report.n_targets >= 20

    # 6) Run the cross-agent analysis.
    proposer_llm = FakeLLMClient(responses=[
        '{"rules": [{"semantic_predicate": "kinase target with paralog_count >= 2 and a known downstream feedback loop",'
        ' "action": "Raise bypass_risk by 0.3 and require combo strategy.",'
        ' "rationale": "Paralog + feedback drives systematic bypass.",'
        ' "polarity": "requires_check"}]}'
    ])
    analysis = run_cross_agent_analysis(report, cases, proposer_llm, severity_floor=0.3)
    assert len(analysis.systematic_errors) >= 1
    assert len(analysis.candidate_rules) >= 1

    # 7) Ingest the corrective rule into the UNIVERSAL layer.
    for rule in analysis.candidate_rules:
        router.ingest_rule(rule, source_layer=Layer.UNIVERSAL, customer_tag=None)

    # 8) Retrieve and confirm.
    retrieved = router.retrieve_for_case(case_disease_scope="oncology_mapk",
                                          customer_tag=None, top_k=10)
    retrieved_ids = {r.rule_id for r in retrieved}
    assert any(c.rule_id in retrieved_ids for c in analysis.candidate_rules)


def test_mapk_retro_v0_2_size():
    cases = build_mapk_retro_seed(version="v0.2")
    assert len(cases) >= 20
    assert all(c.ground_truth_bypass is not BypassMechanism.UNKNOWN for c in cases)


def test_mapk_retro_v0_1_legacy_still_works():
    cases = build_mapk_retro_seed(version="v0.1")
    assert len(cases) == 3
```

Integration-test budget: must complete in < 10 s on a laptop. No real
network, no real LLM.

## Test expectations

New tests:
- `test_full_pipeline_mapk_retro_with_fakellm`
- `test_mapk_retro_v0_2_size`
- `test_mapk_retro_v0_1_legacy_still_works`

All previously-passing tests across the whole `tests/test_targetval_*.py`
glob must still pass.

## Dependencies on other builders

ALL of them:
- `topology-builder` — `PathwayCache.to_json`, `dominant_bypass_mechanism`.
- `ingest-builder` — `EvidencePipeline` if you use it for the integration
  test (you don't strictly need to — you can build cases directly via
  `MapkRetroEntry`).
- `swarm-llm-builder` — `SwarmRunner(pathway_cache=...)`, `parse_memo`
  contract.
- `layer-store-builder` — real `LayerRouter` write/read paths.
- `proposer-builder` — `propose_corrective_rules` emits real rules
  (the canned LLM JSON shape must match what their tests use).

Wait until `tester` confirms all five passed before starting.

## Hard constraints

- Every entry needs a defensible source (clinicaltrials.gov ID or PMID
  in `notes`). No fabricated outcomes.
- `FakeLLMClient` for the integration test. No real LLM calls.
- Integration test runs in well under 10 s.
- Do NOT load Reactome flat files. Hard-code the PathwayCache contents
  for the 20+ genes by hand.
- Files under 500 lines — that's why `mapk_retro_data.py` is split off.
- Do NOT touch `dashboard/`.
- Do NOT run `git add` / `git commit`.

## When done

SendMessage `tester` with a summary:
- count of curated entries shipped,
- pytest final count (target ~69 tests),
- any cases where you couldn't find a defensible source and substituted
  a "best-effort" entry (mark these in `notes` with `[best-effort]`).
