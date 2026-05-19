"""Recursion-shaped ingest layer for de-identified target-validation data.

This module defines the schema we'd ask a customer (Recursion or any
pharma) to export when running a target-validation engagement. It mirrors
the leadop DATA_FORMAT_SPEC.md pattern: a small set of tabular shapes
the customer can produce from ELN / DB without us seeing program-
sensitive identifiers.

What the customer hands us
--------------------------

Two CSVs (or JSONL / Parquet / DuckDB — any tabular format):

Table A — `targets.csv`
  - case_id (opaque)
  - gene_symbol (or opaque target_id if they want to hide gene names)
  - disease_label (free text — sanitised of program codename)
  - modality (small_molecule / antibody / ...)
  - therapeutic_area (free text)
  - readouts_json (free-form dict of their internal screen / phenotypic
    / CRISPR / phenoprint readouts — we pass this verbatim into
    InternalEvidence.readouts)

Table B — `target_outcomes.csv` (optional; only for retrospective
runs where the customer can share which targets panned out)
  - case_id (matches Table A)
  - outcome_label (one of GroundTruthOutcome values)
  - bypass_label (one of BypassMechanism values; "unknown" allowed)
  - notes (free text)

Status: SKELETON loaders. Real CSV / Parquet parsers + a sanitisation
pass land in a follow-up — the function signatures are pinned here so
downstream code can call them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional

from medreason.targetval.case import (
    BypassMechanism,
    DiseaseContext,
    GroundTruthOutcome,
    Modality,
    TargetID,
    TargetValidationCase,
)
from medreason.targetval.evidence import (
    EvidenceBundle,
    InternalEvidence,
)


CUSTOMER_INGEST_FORMAT_VERSION = "0.1"


def load_customer_targets(
    targets_iter: Iterable[dict[str, Any]],
    *,
    customer_tag: str,
    outcomes_iter: Optional[Iterable[dict[str, Any]]] = None,
) -> list[TargetValidationCase]:
    """Convert customer-ingest rows (already parsed from CSV/JSONL) into
    TargetValidationCase objects.

    `customer_tag` is the opaque tenant identifier. It is stamped into
    InternalEvidence.customer_tag so the layer-router's leak guard can
    enforce per-tenant boundaries on any rule extracted from these cases.

    The function does not touch the filesystem — caller passes iterables
    of already-parsed dicts. Real format-specific parsers (CSV / Parquet /
    DuckDB) live separately and call into this function.
    """
    outcomes_by_id: dict[str, dict[str, Any]] = {}
    if outcomes_iter:
        for row in outcomes_iter:
            cid = row.get("case_id")
            if cid:
                outcomes_by_id[str(cid)] = row

    cases: list[TargetValidationCase] = []
    for row in targets_iter:
        case_id = str(row["case_id"])
        gene = str(row.get("gene_symbol") or row.get("target_id") or case_id)

        modality_raw = (row.get("modality") or "unknown").lower()
        modality = (
            Modality(modality_raw) if modality_raw in Modality._value2member_map_
            else Modality.UNKNOWN
        )

        readouts_raw = row.get("readouts_json") or {}
        if isinstance(readouts_raw, str):
            try:
                readouts = json.loads(readouts_raw)
            except json.JSONDecodeError:
                readouts = {"_unparsed": readouts_raw}
        else:
            readouts = dict(readouts_raw)

        outcome_row = outcomes_by_id.get(case_id) or {}
        outcome_val = (outcome_row.get("outcome_label") or "unknown").lower()
        try:
            outcome = GroundTruthOutcome(outcome_val)
        except ValueError:
            outcome = GroundTruthOutcome.UNKNOWN

        bypass_val = (outcome_row.get("bypass_label") or "unknown").lower()
        try:
            bypass = BypassMechanism(bypass_val)
        except ValueError:
            bypass = BypassMechanism.UNKNOWN

        cases.append(
            TargetValidationCase(
                case_id=case_id,
                target=TargetID(gene_symbol=gene),
                disease=DiseaseContext(
                    disease_label=str(row.get("disease_label") or "unspecified"),
                    therapeutic_area=row.get("therapeutic_area"),
                ),
                modality=modality,
                evidence=EvidenceBundle(
                    internal=InternalEvidence(
                        customer_tag=customer_tag,
                        readouts=readouts,
                        notes=row.get("notes"),
                    ),
                ),
                ground_truth_outcome=outcome,
                ground_truth_bypass=bypass,
                ground_truth_notes=outcome_row.get("notes"),
                source=f"customer_{customer_tag}",
            )
        )
    return cases


def parse_targets_csv(path: str | Path) -> list[dict[str, Any]]:
    """Minimal CSV parser — DictReader semantics. Caller passes the
    result into load_customer_targets().
    """
    import csv

    path = Path(path)
    with path.open() as fh:
        return list(csv.DictReader(fh))


def parse_outcomes_csv(path: str | Path) -> list[dict[str, Any]]:
    """Same shape as parse_targets_csv, kept separate for symmetry."""
    return parse_targets_csv(path)
