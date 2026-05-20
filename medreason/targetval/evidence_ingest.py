"""Evidence fetchers — fill an EvidenceBundle for a (target, disease) pair.

Each fetcher is a Protocol so we can swap real-network impls in / out and
keep the scaffold offline-testable. Real fetchers (with network) land in
follow-up commits; v0.1 ships only the Protocols + Fake* implementations
for tests. v0.2 adds an EvidencePipeline orchestrator so callers can wire
fakes once and reuse them across an entire campaign.

Data sources targeted (in priority order for MVP):

1. OpenTargets Platform — genetics + association scores. Public REST.
2. DepMap public release — CRISPR/RNAi gene-effect across cell lines.
3. Reactome + OmniPath — pathway topology + paralog families.
4. clinicaltrials.gov + AACT — trial outcomes for prior drugs against
   the target.
5. Customer-internal CSV/Parquet — drops into InternalEvidence as-is.

The orchestrator pattern: ``EvidencePipeline.fetch`` calls every fetcher
that's wired and merges results. Missing data stays missing (not zero).
The ``CachedTopologyFetcher`` stamps the active ``PathwayCache.cache_version``
into ``PathwayTopologyEvidence.reference_pathway`` so every bundle records
which topology snapshot fed it — a precondition for reproducible swarm
runs (per §Builder 1 of the Phase 2 plan).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

from .case import DiseaseContext, TargetID
from .evidence import (
    EvidenceBundle,
    GeneticsEvidence,
    InternalEvidence,
    KnockoutDependenceEvidence,
    PathwayTopologyEvidence,
    PriorTrialEvidence,
)
from .topology import PathwayCache


# ── Fetcher Protocols ─────────────────────────────────────────────────────────


class GeneticsFetcher(Protocol):
    def fetch(self, target: TargetID, disease: DiseaseContext) -> GeneticsEvidence:
        ...


class KnockoutFetcher(Protocol):
    def fetch(
        self, target: TargetID, disease: DiseaseContext
    ) -> KnockoutDependenceEvidence:
        ...


class TopologyFetcher(Protocol):
    def fetch(
        self, target: TargetID, disease: DiseaseContext
    ) -> PathwayTopologyEvidence:
        ...


class TrialsFetcher(Protocol):
    def fetch(
        self, target: TargetID, disease: DiseaseContext
    ) -> PriorTrialEvidence:
        ...


# ── Fake / cache-backed implementations (offline-testable) ────────────────────


class FakeGeneticsFetcher:
    def __init__(self, by_target: Optional[dict[str, GeneticsEvidence]] = None):
        self._table = by_target or {}

    def fetch(self, target: TargetID, disease: DiseaseContext) -> GeneticsEvidence:
        return self._table.get(target.gene_symbol, GeneticsEvidence())


class FakeKnockoutFetcher:
    def __init__(
        self, by_target: Optional[dict[str, KnockoutDependenceEvidence]] = None
    ):
        self._table = by_target or {}

    def fetch(
        self, target: TargetID, disease: DiseaseContext
    ) -> KnockoutDependenceEvidence:
        return self._table.get(target.gene_symbol, KnockoutDependenceEvidence())


class CachedTopologyFetcher:
    """Topology fetcher backed by a PathwayCache.

    The cache itself is populated by an offline Reactome/OmniPath ingest
    pipeline. Every bundle stamps the cache's ``cache_version`` into
    ``PathwayTopologyEvidence.reference_pathway`` so downstream consumers
    can trace which topology snapshot produced the evidence.
    """

    def __init__(self, cache: PathwayCache):
        self._cache = cache

    def fetch(
        self, target: TargetID, disease: DiseaseContext
    ) -> PathwayTopologyEvidence:
        gene = target.gene_symbol
        paralogs = self._cache.lookup_paralogs(gene)
        feedback = self._cache.lookup_feedback_loops(gene)
        redundancy = self._cache.lookup_downstream_redundancy(gene)
        upstream = self._cache.lookup_upstream_nodes(gene)
        downstream = self._cache.lookup_downstream_nodes(gene)
        return PathwayTopologyEvidence(
            paralog_count=paralogs.count if paralogs.count else None,
            paralogs=list(paralogs.paralogs),
            downstream_redundancy_index=redundancy,
            upstream_node_count=len(upstream) if upstream else None,
            downstream_node_count=len(downstream) if downstream else None,
            known_feedback_loops=feedback,
            reference_pathway=self._cache.cache_version,
        )


class FakeTrialsFetcher:
    def __init__(self, by_target: Optional[dict[str, PriorTrialEvidence]] = None):
        self._table = by_target or {}

    def fetch(self, target: TargetID, disease: DiseaseContext) -> PriorTrialEvidence:
        return self._table.get(target.gene_symbol, PriorTrialEvidence())


# ── Orchestrator ──────────────────────────────────────────────────────────────


@dataclass
class EvidencePipeline:
    """Holds wired fetchers. One pipeline serves a whole campaign.

    Construct once (typically via :func:`build_default_pipeline`) and call
    :meth:`fetch` per (target, disease) pair. Missing fetchers leave the
    corresponding sub-bundle at its empty default — they don't raise.

    The pipeline is intentionally stateless except for the fetchers it
    holds; any caching belongs inside the fetcher itself (e.g. the
    ``PathwayCache`` referenced by ``CachedTopologyFetcher``).
    """

    genetics: Optional[GeneticsFetcher] = None
    knockout: Optional[KnockoutFetcher] = None
    topology: Optional[TopologyFetcher] = None
    trials: Optional[TrialsFetcher] = None

    def fetch(
        self,
        target: TargetID,
        disease: DiseaseContext,
        *,
        internal: Optional[InternalEvidence] = None,
    ) -> EvidenceBundle:
        """Assemble an :class:`EvidenceBundle` by calling every wired fetcher."""
        return EvidenceBundle(
            genetics=(
                self.genetics.fetch(target, disease)
                if self.genetics
                else GeneticsEvidence()
            ),
            knockout=(
                self.knockout.fetch(target, disease)
                if self.knockout
                else KnockoutDependenceEvidence()
            ),
            topology=(
                self.topology.fetch(target, disease)
                if self.topology
                else PathwayTopologyEvidence()
            ),
            prior_trials=(
                self.trials.fetch(target, disease)
                if self.trials
                else PriorTrialEvidence()
            ),
            internal=internal or InternalEvidence(),
        )


def build_default_pipeline(
    cache: PathwayCache,
    *,
    genetics_table: Optional[dict[str, GeneticsEvidence]] = None,
    knockout_table: Optional[dict[str, KnockoutDependenceEvidence]] = None,
    trials_table: Optional[dict[str, PriorTrialEvidence]] = None,
) -> EvidencePipeline:
    """Wire the standard offline pipeline.

    - ``CachedTopologyFetcher`` always wraps the supplied ``cache`` so
      topology evidence stamps the active ``cache_version``.
    - The three lookup tables default to empty dicts, in which case the
      respective Fake fetcher returns the empty pydantic default.

    Tests and the offline CLI use this factory to assemble a deterministic
    pipeline without touching the network.
    """
    return EvidencePipeline(
        genetics=FakeGeneticsFetcher(genetics_table),
        knockout=FakeKnockoutFetcher(knockout_table),
        topology=CachedTopologyFetcher(cache),
        trials=FakeTrialsFetcher(trials_table),
    )


# ── Backwards-compatible orchestrator (free function) ─────────────────────────


def build_evidence_bundle(
    target: TargetID,
    disease: DiseaseContext,
    *,
    genetics: Optional[GeneticsFetcher] = None,
    knockout: Optional[KnockoutFetcher] = None,
    topology: Optional[TopologyFetcher] = None,
    trials: Optional[TrialsFetcher] = None,
    internal: Optional[InternalEvidence] = None,
) -> EvidenceBundle:
    """Free-function orchestrator preserved for callers that don't keep a
    pipeline around. Equivalent to instantiating an :class:`EvidencePipeline`
    inline and calling ``.fetch``.
    """
    pipeline = EvidencePipeline(
        genetics=genetics, knockout=knockout, topology=topology, trials=trials
    )
    return pipeline.fetch(target, disease, internal=internal)
