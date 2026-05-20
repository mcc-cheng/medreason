# Dispatch — `topology-builder`

**From:** architect
**Phase 2 plan:** `/Users/davidzhang/Desktop/Origin/Personal/medreason/docs/targetval_phase2_plan.md` (§Builder 1)
**Working dir:** `/Users/davidzhang/Desktop/Origin/Personal/medreason`
**Python:** `/opt/miniconda3/bin/python`
**Verify command:** `cd /Users/davidzhang/Desktop/Origin/Personal/medreason && /opt/miniconda3/bin/python -m pytest tests/test_targetval_*.py -q`

## Scope

Extend `medreason/targetval/topology.py` so `derive_bypass_signals` is
modality-aware and `PathwayCache` is JSON-serialisable for snapshot
reproducibility.

## Files to edit / create

- `medreason/targetval/topology.py` — extend in place.
- (new, only if needed to stay under 500 lines) `medreason/targetval/topology_signals.py`.
- `tests/test_targetval_topology.py` — append cases.

## Interface signatures

```python
@dataclass(frozen=True)
class BypassSignal:
    mechanism: str
    strength: float
    evidence_summary: str
    contributing_features: tuple[str, ...] = ()  # NEW

@dataclass
class PathwayCache:
    paralogs_by_gene: dict[str, list[str]] = field(default_factory=dict)
    family_by_gene: dict[str, str] = field(default_factory=dict)
    feedback_loops_by_gene: dict[str, list[str]] = field(default_factory=dict)
    downstream_redundancy: dict[str, float] = field(default_factory=dict)
    upstream_nodes_by_gene: dict[str, list[str]] = field(default_factory=dict)   # NEW
    downstream_nodes_by_gene: dict[str, list[str]] = field(default_factory=dict) # NEW
    cache_version: str = "v0.1"                                                  # NEW
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
    modality: Optional["Modality"] = None,  # NEW
) -> list[BypassSignal]: ...

def dominant_bypass_mechanism(signals: list[BypassSignal]) -> "BypassMechanism":
    ...
```

Modality filter:
- `SMALL_MOLECULE` / `PROTAC`: all signals (paralog, feedback, alternative-pathway).
- `ANTIBODY`: paralog + alternative-pathway (drop intracellular feedback).
- `ASO` / `SIRNA`: paralog + alternative-pathway (drop feedback).
- `None`: all signals (preserves current behavior).

`dominant_bypass_mechanism` maps the highest-strength signal's
`mechanism` string to a `BypassMechanism` enum value; empty / all-zero
returns `BypassMechanism.NO_BYPASS_KNOWN`. The mechanism strings already
match the enum's `.value` strings — no aliasing needed beyond enum
construction.

## Test expectations

Add to `tests/test_targetval_topology.py`:
- `test_modality_filter_drops_feedback_for_antibody`
- `test_pathway_cache_roundtrip_json`
- `test_pathway_cache_file_roundtrip` (uses `tmp_path`)
- `test_dominant_bypass_mechanism_empty_returns_no_bypass`
- `test_dominant_bypass_mechanism_picks_highest_strength`
- `test_bypass_signal_contributing_features_populated`

The 5 existing topology tests must still pass unchanged.

## Dependencies on other builders

None. You're upstream of everyone.

## Hard constraints

- No external HTTP / network code.
- No LLM imports.
- Pure-Python deterministic logic.
- Files under 500 lines — split into `topology_signals.py` if needed.
- Do NOT touch `dashboard/`.
- Do NOT run `git add` / `git commit`.

## When done

SendMessage `tester` with a summary:
- list of new tests added,
- pytest count before vs after,
- any deviations from the signatures above (and why).
