"""Pathway topology features for target validation.

Computes the structural features that drive bypass-risk reasoning:

- Paralog count and identity (same gene family) — predicts paralog
  compensation bypass.
- Downstream redundancy index — predicts alternative-pathway bypass.
- Upstream / downstream node counts — proxy for how isolated vs
  hub-like a target is in its pathway.
- Known feedback loops — predicts pharmacologic-rebound bypass.

The functions here read from a pathway-DB cache (Reactome / OmniPath
flat files). v0.2 keeps the cache in-memory but adds JSON
serialisation so swarm runs can pin a deterministic snapshot. Real DB
ingest lands in Phase 3. The choice to keep this offline-only is
deliberate: the network fetcher lives in evidence_ingest.py and writes
to the cache, so this module stays pure-Python and deterministic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .case import BypassMechanism, Modality


# ── Public types ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ParalogFeature:
    """Result of a paralog lookup for a single target."""

    gene_symbol: str
    family: Optional[str]
    paralogs: tuple[str, ...]

    @property
    def count(self) -> int:
        return len(self.paralogs)


@dataclass(frozen=True)
class BypassSignal:
    """One bypass-mechanism signal computed for a target.

    `mechanism` matches the BypassMechanism enum's `.value` strings in
    case.py. `strength` ∈ [0, 1] is a soft prior — higher = more
    concerning. `evidence_summary` is short free text injected into the
    agent prompt. `contributing_features` records which raw cache
    features fed the signal so downstream rule provenance can cite them
    (e.g. ``("paralog_count=3",)``).
    """

    mechanism: str
    strength: float
    evidence_summary: str
    contributing_features: tuple[str, ...] = ()


# ── Cache (in-memory; persistent backing comes later) ─────────────────────────


_CACHE_JSON_FIELDS = (
    "paralogs_by_gene",
    "family_by_gene",
    "feedback_loops_by_gene",
    "downstream_redundancy",
    "upstream_nodes_by_gene",
    "downstream_nodes_by_gene",
    "cache_version",
)


@dataclass
class PathwayCache:
    """In-memory pathway DB cache with deterministic JSON snapshotting.

    Real implementation: a parquet snapshot of Reactome reactions +
    OmniPath family annotations, refreshed quarterly. v0.2 keeps the
    data in-memory but `to_json` / `write` produce a deterministic,
    sorted-key payload so two callers populating the same fixtures
    produce byte-identical snapshots — a precondition for hashing the
    cache version into `PathwayTopologyEvidence.reference_pathway`.
    """

    paralogs_by_gene: dict[str, list[str]] = field(default_factory=dict)
    family_by_gene: dict[str, str] = field(default_factory=dict)
    feedback_loops_by_gene: dict[str, list[str]] = field(default_factory=dict)
    downstream_redundancy: dict[str, float] = field(default_factory=dict)
    upstream_nodes_by_gene: dict[str, list[str]] = field(default_factory=dict)
    downstream_nodes_by_gene: dict[str, list[str]] = field(default_factory=dict)
    cache_version: str = "v0.1"

    # ── Lookups ───────────────────────────────────────────────────────────────

    def lookup_paralogs(self, gene_symbol: str) -> ParalogFeature:
        paralogs = tuple(self.paralogs_by_gene.get(gene_symbol, ()))
        family = self.family_by_gene.get(gene_symbol)
        return ParalogFeature(gene_symbol=gene_symbol, family=family, paralogs=paralogs)

    def lookup_feedback_loops(self, gene_symbol: str) -> list[str]:
        return list(self.feedback_loops_by_gene.get(gene_symbol, ()))

    def lookup_downstream_redundancy(self, gene_symbol: str) -> Optional[float]:
        return self.downstream_redundancy.get(gene_symbol)

    def lookup_upstream_nodes(self, gene_symbol: str) -> list[str]:
        return list(self.upstream_nodes_by_gene.get(gene_symbol, ()))

    def lookup_downstream_nodes(self, gene_symbol: str) -> list[str]:
        return list(self.downstream_nodes_by_gene.get(gene_symbol, ()))

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_json(self) -> str:
        """Render a deterministic JSON snapshot.

        Dict keys are sorted; list values preserve insertion order
        (paralog lists are biologically ordered, not alphabetical). Two
        caches populated with the same data hash to identical bytes.
        """
        payload = {
            "paralogs_by_gene": _sorted_dict(self.paralogs_by_gene),
            "family_by_gene": _sorted_dict(self.family_by_gene),
            "feedback_loops_by_gene": _sorted_dict(self.feedback_loops_by_gene),
            "downstream_redundancy": _sorted_dict(self.downstream_redundancy),
            "upstream_nodes_by_gene": _sorted_dict(self.upstream_nodes_by_gene),
            "downstream_nodes_by_gene": _sorted_dict(self.downstream_nodes_by_gene),
            "cache_version": self.cache_version,
        }
        return json.dumps(payload, sort_keys=True, indent=2)

    @classmethod
    def from_json(cls, payload: str) -> "PathwayCache":
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise ValueError("PathwayCache JSON payload must be a JSON object")
        unknown = set(data) - set(_CACHE_JSON_FIELDS)
        if unknown:
            raise ValueError(f"Unknown PathwayCache fields: {sorted(unknown)}")
        return cls(
            paralogs_by_gene={k: list(v) for k, v in data.get("paralogs_by_gene", {}).items()},
            family_by_gene=dict(data.get("family_by_gene", {})),
            feedback_loops_by_gene={
                k: list(v) for k, v in data.get("feedback_loops_by_gene", {}).items()
            },
            downstream_redundancy={
                k: float(v) for k, v in data.get("downstream_redundancy", {}).items()
            },
            upstream_nodes_by_gene={
                k: list(v) for k, v in data.get("upstream_nodes_by_gene", {}).items()
            },
            downstream_nodes_by_gene={
                k: list(v) for k, v in data.get("downstream_nodes_by_gene", {}).items()
            },
            cache_version=str(data.get("cache_version", "v0.1")),
        )

    def write(self, path: str | Path) -> Path:
        """Write the snapshot to ``path`` and return the resolved Path."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.to_json(), encoding="utf-8")
        return out

    @classmethod
    def load(cls, path: str | Path) -> "PathwayCache":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


def _sorted_dict(d: dict) -> dict:
    """Return a new dict with keys in sorted order (deterministic JSON)."""
    return {k: d[k] for k in sorted(d)}


# ── Bypass signal derivation ──────────────────────────────────────────────────


_PARALOG_STRENGTH = 0.15  # per paralog, capped at 0.75
_FEEDBACK_STRENGTH = 0.5
_FEEDBACK_REACTIVATION_STRENGTH = 0.65  # known MAPK reactivation, higher prior
_REDUNDANCY_STRENGTH_SCALE = 1.0  # multiplies the [0,1] redundancy index

# Known MAPK feedback reactivation loops — BRAF inhibition causes RAS-GTP
# buildup → CRAF dimerisation → ERK reactivation. The same shape applies to
# other class-I RAF inhibitors. Curated, not derived from a fetcher.
_FEEDBACK_REACTIVATION_GENES: frozenset[str] = frozenset({"BRAF", "CRAF", "RAF1", "ARAF"})


# Modality → allowed mechanism strings. None means "allow all".
# - SMALL_MOLECULE / PROTAC: all mechanisms (intracellular access).
# - ANTIBODY: paralog + alternative-pathway (extracellular only — intracellular
#   feedback rebound isn't an antibody-relevant bypass route).
# - ASO / SIRNA: paralog + alternative-pathway (no pharmacologic feedback
#   rebound since the protein is depleted, not inhibited).
# - CELL_THERAPY / GENE_EDITING / UNKNOWN: default to "all" (UNKNOWN preserves
#   pre-v0.2 behaviour).
_ALL_MECHS: frozenset[str] = frozenset({
    "paralog_compensation",
    "downstream_feedback",
    "feedback_reactivation",
    "alternative_pathway",
})

_MODALITY_ALLOWED: dict[Modality, frozenset[str]] = {
    Modality.SMALL_MOLECULE: _ALL_MECHS,
    Modality.PROTAC: _ALL_MECHS,
    Modality.ANTIBODY: frozenset({"paralog_compensation", "alternative_pathway"}),
    Modality.ASO: frozenset({"paralog_compensation", "alternative_pathway"}),
    Modality.SIRNA: frozenset({"paralog_compensation", "alternative_pathway"}),
    Modality.CELL_THERAPY: _ALL_MECHS,
    Modality.GENE_EDITING: _ALL_MECHS,
    Modality.UNKNOWN: _ALL_MECHS,
}


def derive_bypass_signals(
    gene_symbol: str,
    cache: PathwayCache,
    *,
    modality: Optional[Modality] = None,
) -> list[BypassSignal]:
    """Compute bypass signals for a single target from the pathway cache.

    Returns one BypassSignal per detected mechanism with non-zero
    strength. Empty list means no signals — NOT "low risk", just "we
    don't have evidence either way."

    The optional ``modality`` filter drops signals that are
    biologically implausible for the chosen modality (e.g. an antibody
    can't engage an intracellular feedback rebound). Passing
    ``modality=None`` preserves the pre-v0.2 behaviour and returns
    every detected signal.
    """
    allowed = _allowed_mechanisms(modality)
    signals: list[BypassSignal] = []

    paralog_feature = cache.lookup_paralogs(gene_symbol)
    if paralog_feature.count > 0 and "paralog_compensation" in allowed:
        strength = min(0.75, paralog_feature.count * _PARALOG_STRENGTH)
        features = (
            f"paralog_count={paralog_feature.count}",
            f"paralog_family={paralog_feature.family or 'unknown'}",
        )
        signals.append(
            BypassSignal(
                mechanism="paralog_compensation",
                strength=strength,
                evidence_summary=(
                    f"{paralog_feature.count} paralogs in family "
                    f"{paralog_feature.family or 'unknown'}: "
                    f"{', '.join(paralog_feature.paralogs)}"
                ),
                contributing_features=features,
            )
        )

    feedback = cache.lookup_feedback_loops(gene_symbol)
    if feedback:
        # Known MAPK reactivation gets its own higher-prior signal so the
        # swarm can distinguish a documented rebound (BRAFi → ERK
        # reactivation) from generic downstream feedback.
        is_reactivation = gene_symbol in _FEEDBACK_REACTIVATION_GENES
        mechanism = "feedback_reactivation" if is_reactivation else "downstream_feedback"
        if mechanism in allowed:
            strength = (
                _FEEDBACK_REACTIVATION_STRENGTH if is_reactivation else _FEEDBACK_STRENGTH
            )
            features = (
                f"feedback_loop_count={len(feedback)}",
                f"feedback_kind={'reactivation' if is_reactivation else 'generic'}",
            )
            label = "Known reactivation loops" if is_reactivation else "Known feedback loops"
            signals.append(
                BypassSignal(
                    mechanism=mechanism,
                    strength=strength,
                    evidence_summary=f"{label}: {', '.join(feedback)}",
                    contributing_features=features,
                )
            )

    redundancy = cache.lookup_downstream_redundancy(gene_symbol)
    if (
        redundancy is not None
        and redundancy > 0
        and "alternative_pathway" in allowed
    ):
        downstream = cache.lookup_downstream_nodes(gene_symbol)
        features = (
            f"redundancy_index={redundancy:.2f}",
            f"downstream_node_count={len(downstream)}",
        )
        signals.append(
            BypassSignal(
                mechanism="alternative_pathway",
                strength=min(1.0, redundancy * _REDUNDANCY_STRENGTH_SCALE),
                evidence_summary=(
                    f"Downstream redundancy index {redundancy:.2f}"
                ),
                contributing_features=features,
            )
        )

    return signals


def _allowed_mechanisms(modality: Optional[Modality]) -> frozenset[str]:
    if modality is None:
        return _ALL_MECHS
    return _MODALITY_ALLOWED.get(modality, _ALL_MECHS)


# ── Aggregation ───────────────────────────────────────────────────────────────


# Mechanism strings → BypassMechanism enum. The string values match the
# enum's `.value` strings 1:1 for the canonical mechanisms; the
# v0.2-only `feedback_reactivation` collapses into DOWNSTREAM_FEEDBACK
# because the public retro ground-truth label space doesn't yet
# distinguish them.
_MECHANISM_TO_ENUM: dict[str, BypassMechanism] = {
    "paralog_compensation": BypassMechanism.PARALOG_COMPENSATION,
    "downstream_feedback": BypassMechanism.DOWNSTREAM_FEEDBACK,
    "feedback_reactivation": BypassMechanism.DOWNSTREAM_FEEDBACK,
    "alternative_pathway": BypassMechanism.ALTERNATIVE_PATHWAY,
}


def dominant_bypass_mechanism(signals: list[BypassSignal]) -> BypassMechanism:
    """Pick the highest-strength signal and map it to a BypassMechanism.

    Returns ``BypassMechanism.NO_BYPASS_KNOWN`` if the list is empty or
    every signal has zero strength. Unknown mechanism strings (should
    not happen with the curated cache, but defensive) also resolve to
    ``NO_BYPASS_KNOWN`` rather than ``UNKNOWN``: at this point we've
    looked, found nothing actionable, so the assertion is "no bypass
    known", not "inference still pending".
    """
    if not signals:
        return BypassMechanism.NO_BYPASS_KNOWN
    ranked = max(signals, key=lambda s: s.strength)
    if ranked.strength <= 0.0:
        return BypassMechanism.NO_BYPASS_KNOWN
    return _MECHANISM_TO_ENUM.get(ranked.mechanism, BypassMechanism.NO_BYPASS_KNOWN)
