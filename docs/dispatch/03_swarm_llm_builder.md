# Dispatch — `swarm-llm-builder`

**From:** architect
**Phase 2 plan:** `/Users/davidzhang/Desktop/Origin/Personal/medreason/docs/targetval_phase2_plan.md` (§Builder 3)
**Working dir:** `/Users/davidzhang/Desktop/Origin/Personal/medreason`
**Python:** `/opt/miniconda3/bin/python`
**Verify command:** `cd /Users/davidzhang/Desktop/Origin/Personal/medreason && /opt/miniconda3/bin/python -m pytest tests/test_targetval_*.py -q`

## Scope

Replace the placeholder `TargetMemo` (currently fixed
`priority_score=0.5`) with a real LLM-driven memo via strict-JSON
parsing. Output schema is read by `proposer-builder` — lock it.

## Files to edit / create

- `medreason/targetval/swarm.py` — extend `SwarmAgent.run`, finalize
  `TargetMemo` shape with two new fields.
- (new) `medreason/targetval/swarm_prompts.py` — system + user prompt
  builders.
- (new) `medreason/targetval/swarm_parsing.py` — JSON extract + memo parser.
- `tests/test_targetval_swarm.py` — add parse-path tests; loosen any
  existing assertion that depends on `priority_score == 0.5`.

## Interface signatures

```python
# medreason/targetval/swarm.py

@dataclass
class TargetMemo:
    case_id: str
    gene_symbol: str
    priority_score: float
    bypass_risk_score: float
    predicted_bypass: BypassMechanism = BypassMechanism.UNKNOWN
    supporting_evidence: list[str] = field(default_factory=list)
    weakening_evidence: list[str] = field(default_factory=list)
    proposed_experiments: list[str] = field(default_factory=list)
    rationale: str = ""
    retrieved_rule_ids: list[str] = field(default_factory=list)
    applied_rule_ids: list[str] = field(default_factory=list)
    bypass_signals_seen: list[str] = field(default_factory=list)  # NEW
    parse_warnings: list[str] = field(default_factory=list)        # NEW
    cost_usd: float = 0.0
    seed: int = 0

class SwarmAgent:
    def __init__(
        self,
        case: TargetValidationCase,
        llm: LLMClient,
        layer_router: LayerRouter,
        *,
        customer_tag: Optional[str] = None,
        seed: int = 0,
        pathway_cache: Optional[PathwayCache] = None,  # NEW
    ): ...

class SwarmRunner:
    def __init__(
        self,
        llm: LLMClient,
        layer_router: LayerRouter,
        *,
        customer_tag: Optional[str] = None,
        max_workers: int = 8,
        pathway_cache: Optional[PathwayCache] = None,  # NEW
    ): ...
```

```python
# medreason/targetval/swarm_prompts.py

SYSTEM_PROMPT_TARGETVAL: str

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

def extract_memo_json(text: str) -> dict: ...

def parse_memo(
    text: str,
    *,
    case_id: str,
    gene_symbol: str,
    retrieved_rule_ids: list[str],
    bypass_signals: list[BypassSignal],
    cost_usd: float,
    seed: int,
) -> TargetMemo: ...
```

Parser contract:
- Tolerant on field presence (missing → defaults).
- Score clamping: `priority_score`, `bypass_risk_score` clamp to `[0, 1]`.
  Out-of-range adds an entry to `parse_warnings`.
- `predicted_bypass`: accept `BypassMechanism.value` strings; unknown
  values → `UNKNOWN` + warning.
- Bypass signals seen: copy `[s.mechanism for s in bypass_signals]`
  into `bypass_signals_seen`.
- Raise `MemoParseError` only if the JSON itself can't be extracted at all.

JSON extraction strategy (mirror `leadop/agent_llm.py:_extract_json`):
1. Try `json.loads(text.strip())`.
2. Try fenced ```json``` block regex.
3. Try first balanced `{...}` block regex.
4. Otherwise raise `MemoParseError`.

## Test expectations

New tests:
- `test_parse_memo_happy_path`
- `test_parse_memo_fenced_json`
- `test_parse_memo_clamps_scores`
- `test_parse_memo_unknown_bypass_warns`
- `test_parse_memo_garbage_raises_memo_parse_error`
- `test_swarm_agent_uses_bypass_signals_in_prompt`

Existing swarm tests (`test_swarm_runs_one_agent_per_case`,
`test_swarm_aggregate_ranking_total_and_ordered`,
`test_swarm_handles_empty_case_list`,
`test_swarm_parallel_path_same_results_as_serial`,
`test_aggregate_ranking_prefers_high_priority_low_bypass`) must still pass.
If any of them implicitly depended on `priority_score==0.5`, loosen
to "float in [0,1]" — the swarm now parses real JSON. The current
tests assert orchestration (call counts, ranking totality) so most
should pass as-is.

## Dependencies on other builders

- `PathwayCache` shape (incl. `to_json`) from `topology-builder`. The
  default-cache path (no cache passed → empty cache) MUST keep existing
  swarm tests working.

## Hard constraints

- Tests use `FakeLLMClient` ONLY. Never wire real LLM clients.
- JSON parser never uses `eval`. Only `json.loads` + regex.
- Files under 500 lines — that's why prompts + parsing are split out.
- Do NOT touch `dashboard/`.
- Do NOT run `git add` / `git commit`.

## When done

SendMessage `tester` with a summary:
- new + modified tests,
- the locked `TargetMemo` schema (so `proposer-builder` can read it),
- pytest count before vs after.
