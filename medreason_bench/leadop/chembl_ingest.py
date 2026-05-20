"""ChEMBL ingest helper for the lead-op retro.

Thin wrapper over `chembl_webresource_client`. Caches raw API responses
to `medreason_bench/leadop/cache/` as JSONL so smoke tests and repeated
runs don't re-hit the network. The real heavy work is round
annotation, which is done by hand from the published SAR paper and
lives in a per-campaign manifest (see `campaigns/`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

CACHE_ROOT = Path(__file__).resolve().parent / "cache"
CACHE_ROOT.mkdir(exist_ok=True)


@dataclass(frozen=True)
class ChEMBLActivity:
    molecule_chembl_id: str
    canonical_smiles: str | None
    assay_chembl_id: str
    assay_description: str | None
    standard_type: str  # e.g., "IC50"
    standard_value: float | None
    standard_units: str | None
    target_chembl_id: str


def _cache_path(target_chembl_id: str) -> Path:
    return CACHE_ROOT / f"activities_{target_chembl_id}.jsonl"


def fetch_activities(
    target_chembl_id: str,
    *,
    standard_types: Iterable[str] = ("IC50", "Ki", "EC50"),
    limit: int | None = None,
    use_cache: bool = True,
) -> list[ChEMBLActivity]:
    """Fetch activities for a target. Cached to disk as JSONL."""
    cache = _cache_path(target_chembl_id)
    if use_cache and cache.exists():
        return [_activity_from_dict(json.loads(line)) for line in cache.read_text().splitlines()]

    from chembl_webresource_client.new_client import new_client  # noqa: PLC0415

    activity = new_client.activity
    query = activity.filter(
        target_chembl_id=target_chembl_id,
        standard_type__in=list(standard_types),
        standard_value__isnull=False,
    ).only(
        "molecule_chembl_id",
        "canonical_smiles",
        "assay_chembl_id",
        "assay_description",
        "standard_type",
        "standard_value",
        "standard_units",
        "target_chembl_id",
    )

    results: list[ChEMBLActivity] = []
    with cache.open("w", encoding="utf-8") as f:
        for i, row in enumerate(query):
            if limit is not None and i >= limit:
                break
            obj = dict(row)
            f.write(json.dumps(obj) + "\n")
            results.append(_activity_from_dict(obj))
    return results


def _activity_from_dict(row: dict) -> ChEMBLActivity:
    sv = row.get("standard_value")
    return ChEMBLActivity(
        molecule_chembl_id=str(row.get("molecule_chembl_id") or ""),
        canonical_smiles=row.get("canonical_smiles") or None,
        assay_chembl_id=str(row.get("assay_chembl_id") or ""),
        assay_description=row.get("assay_description") or None,
        standard_type=str(row.get("standard_type") or ""),
        standard_value=float(sv) if sv is not None else None,
        standard_units=row.get("standard_units") or None,
        target_chembl_id=str(row.get("target_chembl_id") or ""),
    )


def filter_by_molecule_ids(
    activities: list[ChEMBLActivity], ids: Iterable[str]
) -> list[ChEMBLActivity]:
    id_set = set(ids)
    return [a for a in activities if a.molecule_chembl_id in id_set]
