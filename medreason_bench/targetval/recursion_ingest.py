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

Return-type contract (v0.2)
---------------------------

``load_customer_targets`` now returns an :class:`IngestReport` rather than
``list[TargetValidationCase]``. The report exposes both successful cases
and a parallel ``rejected`` list of ``(row, reason)`` tuples — mirroring
``ProposalResult`` in the leadop rule proposer. Callers that want the
old shape can access ``report.cases`` directly. Setting ``strict=True``
raises :class:`IngestError` on the first malformed row instead of
bucketing it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
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


CUSTOMER_INGEST_FORMAT_VERSION = "0.2"


class IngestError(ValueError):
    """A customer ingest row could not be parsed.

    Raised eagerly when ``load_customer_targets(..., strict=True)``. In
    non-strict mode the same condition is captured as a ``(row, reason)``
    entry in :attr:`IngestReport.rejected` instead.
    """


@dataclass
class IngestReport:
    """Outcome of a customer ingest pass.

    Holds the successfully parsed cases plus a parallel list of rejected
    rows. The shape mirrors ``ProposalResult`` from the leadop rule
    proposer so the failure-bucket pattern is consistent across the
    codebase: parse → typed errors → caller buckets into rejected.
    """

    cases: list[TargetValidationCase] = field(default_factory=list)
    rejected: list[tuple[dict[str, Any], str]] = field(default_factory=list)

    @property
    def n_cases(self) -> int:
        return len(self.cases)

    @property
    def n_rejected(self) -> int:
        return len(self.rejected)


def load_customer_targets(
    targets_iter: Iterable[dict[str, Any]],
    *,
    customer_tag: str,
    outcomes_iter: Optional[Iterable[dict[str, Any]]] = None,
    strict: bool = False,
) -> IngestReport:
    """Convert customer-ingest rows into :class:`TargetValidationCase` objects.

    ``customer_tag`` is the opaque tenant identifier. It is stamped into
    ``InternalEvidence.customer_tag`` so the layer-router's leak guard
    can enforce per-tenant boundaries on any rule extracted from these
    cases.

    The function does not touch the filesystem — caller passes iterables
    of already-parsed dicts. Format-specific parsers (CSV / JSONL /
    Parquet) live in :mod:`customer_csv` and call into this function.

    Parameters
    ----------
    targets_iter:
        Iterable of dicts. Each must carry a non-empty ``case_id``.
        Anything else is optional and defaults to a sensible fallback.
    customer_tag:
        Opaque tenant id stamped into ``InternalEvidence.customer_tag``.
        Must be a non-empty string.
    outcomes_iter:
        Optional iterable of outcome rows. Each must carry a ``case_id``
        matching a row in ``targets_iter``. Outcome rows with no matching
        target are silently dropped (not rejected — the customer often
        runs retrospective + prospective in the same export).
    strict:
        When ``True``, raise :class:`IngestError` on the first malformed
        target row. When ``False`` (default), accumulate failures in
        :attr:`IngestReport.rejected` and keep going.

    Returns
    -------
    :class:`IngestReport`
        ``report.cases`` carries successful cases; ``report.rejected``
        carries ``(row, reason)`` tuples for any row that failed
        validation.
    """
    if not isinstance(customer_tag, str) or not customer_tag:
        raise IngestError("customer_tag must be a non-empty string")

    outcomes_by_id = _index_outcomes(outcomes_iter)

    report = IngestReport()
    for row in targets_iter:
        try:
            case = _build_case_from_row(
                row, customer_tag=customer_tag, outcomes_by_id=outcomes_by_id
            )
        except IngestError as exc:
            if strict:
                raise
            report.rejected.append((dict(row) if isinstance(row, dict) else {"_raw": row}, str(exc)))
            continue
        report.cases.append(case)
    return report


# ── Internal helpers ──────────────────────────────────────────────────────────


def _index_outcomes(
    outcomes_iter: Optional[Iterable[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """Build a case_id → outcome-row index. Bad rows are silently skipped
    (outcomes are best-effort retrospective annotation, not load-bearing).
    """
    if not outcomes_iter:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in outcomes_iter:
        if not isinstance(row, dict):
            continue
        cid = row.get("case_id")
        if cid:
            out[str(cid)] = row
    return out


def _build_case_from_row(
    row: dict[str, Any],
    *,
    customer_tag: str,
    outcomes_by_id: dict[str, dict[str, Any]],
) -> TargetValidationCase:
    """Build a single case from a raw row. Raises :class:`IngestError`
    on any boundary-validation failure.
    """
    if not isinstance(row, dict):
        raise IngestError(f"row must be a dict, got {type(row).__name__}")

    case_id_raw = row.get("case_id")
    if case_id_raw is None or str(case_id_raw).strip() == "":
        raise IngestError("missing case_id")
    case_id = str(case_id_raw).strip()

    gene_raw = row.get("gene_symbol") or row.get("target_id") or case_id
    gene = str(gene_raw).strip()
    if not gene:
        raise IngestError(f"{case_id}: gene_symbol/target_id is empty")

    modality_raw = str(row.get("modality") or "unknown").lower()
    modality = (
        Modality(modality_raw)
        if modality_raw in Modality._value2member_map_
        else Modality.UNKNOWN
    )

    readouts = _coerce_readouts(row.get("readouts_json"))

    outcome_row = outcomes_by_id.get(case_id) or {}
    outcome = _coerce_outcome(outcome_row.get("outcome_label"))
    bypass = _coerce_bypass(outcome_row.get("bypass_label"))

    return TargetValidationCase(
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


def _coerce_readouts(value: Any) -> dict[str, Any]:
    """Return a dict for ``InternalEvidence.readouts``. Unparseable JSON
    strings are preserved verbatim under ``_unparsed`` so the customer
    doesn't silently lose data."""
    if value is None or value == "":
        return {}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"_unparsed": value}
        if isinstance(parsed, dict):
            return parsed
        return {"_unparsed": value}
    if isinstance(value, dict):
        return dict(value)
    # Unknown shape (list, scalar) — keep the raw value visible to the agent
    return {"_unparsed": value}


def _coerce_outcome(value: Any) -> GroundTruthOutcome:
    if value is None:
        return GroundTruthOutcome.UNKNOWN
    try:
        return GroundTruthOutcome(str(value).lower())
    except ValueError:
        return GroundTruthOutcome.UNKNOWN


def _coerce_bypass(value: Any) -> BypassMechanism:
    if value is None:
        return BypassMechanism.UNKNOWN
    try:
        return BypassMechanism(str(value).lower())
    except ValueError:
        return BypassMechanism.UNKNOWN


# ── Format-specific parsers (CSV stays here for back-compat; JSONL /
#    Parquet live in customer_csv.py) ─────────────────────────────────────────


def parse_targets_csv(path: str | Path) -> list[dict[str, Any]]:
    """Minimal CSV parser — DictReader semantics. Caller passes the
    result into :func:`load_customer_targets`.
    """
    import csv

    path = Path(path)
    with path.open() as fh:
        return list(csv.DictReader(fh))


def parse_outcomes_csv(path: str | Path) -> list[dict[str, Any]]:
    """Same shape as :func:`parse_targets_csv`, kept separate for symmetry."""
    return parse_targets_csv(path)
