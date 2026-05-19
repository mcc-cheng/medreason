"""Pathway topology features for target validation.

Computes the structural features that drive bypass-risk reasoning:

- Paralog count and identity (same gene family) — predicts paralog
  compensation bypass.
- Downstream redundancy index — predicts alternative-pathway bypass.
- Upstream / downstream node counts — proxy for how isolated vs
  hub-like a target is in its pathway.
- Known feedback loops — predicts pharmacologic-rebound bypass.

The functions here read from a pathway-DB cache (Reactome / OmniPath
flat files). v0.1 skeleton uses an in-memory dict; real DB ingest lands
later. The choice to keep this offline-only is deliberate: the network
fetcher lives in evidence_ingest.py and writes to the cache, so this
module stays pure-Python and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


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

    `mechanism` matches the BypassMechanism enum in case.py.
    `strength` ∈ [0, 1] is a soft prior — higher = more concerning.
    `evidence_summary` is short free text injected into the agent prompt.
    """

    mechanism: str
    strength: float
    evidence_summary: str


# ── Cache (in-memory; persistent backing comes later) ─────────────────────────


@dataclass
class PathwayCache:
    """Trivial in-memory pathway DB cache. Tests + scaffolding only.

    Real implementation: a parquet snapshot of Reactome reactions +
    OmniPath family annotations, refreshed quarterly, hashed and pinned
    in PROMPTS_LOCK-style.
    """

    paralogs_by_gene: dict[str, list[str]] = field(default_factory=dict)
    family_by_gene: dict[str, str] = field(default_factory=dict)
    feedback_loops_by_gene: dict[str, list[str]] = field(default_factory=dict)
    downstream_redundancy: dict[str, float] = field(default_factory=dict)

    def lookup_paralogs(self, gene_symbol: str) -> ParalogFeature:
        paralogs = tuple(self.paralogs_by_gene.get(gene_symbol, ()))
        family = self.family_by_gene.get(gene_symbol)
        return ParalogFeature(gene_symbol=gene_symbol, family=family, paralogs=paralogs)

    def lookup_feedback_loops(self, gene_symbol: str) -> list[str]:
        return list(self.feedback_loops_by_gene.get(gene_symbol, ()))

    def lookup_downstream_redundancy(self, gene_symbol: str) -> Optional[float]:
        return self.downstream_redundancy.get(gene_symbol)


# ── Bypass signal derivation ──────────────────────────────────────────────────


_PARALOG_STRENGTH = 0.15  # per paralog, capped at 0.75
_FEEDBACK_STRENGTH = 0.5
_REDUNDANCY_STRENGTH_SCALE = 1.0  # multiplies the [0,1] redundancy index


def derive_bypass_signals(
    gene_symbol: str, cache: PathwayCache
) -> list[BypassSignal]:
    """Compute bypass signals for a single target from the pathway cache.

    Returns one BypassSignal per detected mechanism with non-zero strength.
    Empty list means no signals — NOT "low risk", just "we don't have
    evidence either way."
    """
    signals: list[BypassSignal] = []

    paralog_feature = cache.lookup_paralogs(gene_symbol)
    if paralog_feature.count > 0:
        strength = min(0.75, paralog_feature.count * _PARALOG_STRENGTH)
        signals.append(
            BypassSignal(
                mechanism="paralog_compensation",
                strength=strength,
                evidence_summary=(
                    f"{paralog_feature.count} paralogs in family "
                    f"{paralog_feature.family or 'unknown'}: "
                    f"{', '.join(paralog_feature.paralogs)}"
                ),
            )
        )

    feedback = cache.lookup_feedback_loops(gene_symbol)
    if feedback:
        signals.append(
            BypassSignal(
                mechanism="downstream_feedback",
                strength=_FEEDBACK_STRENGTH,
                evidence_summary=f"Known feedback loops: {', '.join(feedback)}",
            )
        )

    redundancy = cache.lookup_downstream_redundancy(gene_symbol)
    if redundancy is not None and redundancy > 0:
        signals.append(
            BypassSignal(
                mechanism="alternative_pathway",
                strength=min(1.0, redundancy * _REDUNDANCY_STRENGTH_SCALE),
                evidence_summary=(
                    f"Downstream redundancy index {redundancy:.2f}"
                ),
            )
        )

    return signals
