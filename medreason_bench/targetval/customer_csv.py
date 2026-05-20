"""Customer-supplied tabular-format parsers for the target-validation
recursion ingest path.

Kept separate from ``recursion_ingest.py`` so the ingest module stays a
small, focused boundary validator. Each parser returns ``list[dict]`` —
the same shape ``load_customer_targets`` expects.

Formats:

- **JSONL** — one JSON object per line. Blank lines tolerated.
- **Parquet** — lazy-imports ``pyarrow``. Raises :class:`ImportError`
  with an actionable message if pyarrow isn't installed.

The parsers do NOT touch the network and perform no schema validation
themselves — that's the ingest layer's job (it buckets bad rows into
``IngestReport.rejected``). They only enforce that the on-disk file is
syntactically well-formed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def parse_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Parse a JSONL file into a list of dicts.

    Blank lines are skipped. Each non-blank line must decode to a JSON
    object (not an array / scalar) — a typed :class:`ValueError` is
    raised with the offending line number otherwise so the caller can
    point the customer at the specific row.
    """
    p = Path(path)
    rows: list[dict[str, Any]] = []
    with p.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{p}:{lineno}: invalid JSON ({exc.msg})"
                ) from exc
            if not isinstance(obj, dict):
                raise ValueError(
                    f"{p}:{lineno}: expected JSON object, got {type(obj).__name__}"
                )
            rows.append(obj)
    return rows


def parse_parquet(path: str | Path) -> list[dict[str, Any]]:
    """Parse a Parquet file into a list of dicts.

    Lazy-imports ``pyarrow.parquet``. If pyarrow isn't installed, raises
    :class:`ImportError` with an install hint instead of an opaque
    ModuleNotFoundError — the customer ingest path is opt-in.
    """
    try:
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError(
            "parse_parquet requires pyarrow. Install with "
            "`pip install pyarrow` to use Parquet ingest."
        ) from exc

    table = pq.read_table(str(Path(path)))
    return table.to_pylist()
