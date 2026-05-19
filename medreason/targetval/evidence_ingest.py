"""Evidence fetchers — fill an EvidenceBundle for a (target, disease) pair.

Each fetcher is a Protocol so we can swap real-network impls in / out and
keep the scaffold offline-testable. Real fetchers (with network) land in
follow-up commits; v0.1 ships only the Protocols + a FakeEvidenceFetcher
for tests.

Data sources targeted (in priority order for MVP):

1. OpenTargets Platform — genetics + association scores. Public REST.
2. DepMap public release — CRISPR/RNAi gene-effect across cell lines.
3. Reactome + OmniPath — pathway topology + paralog families.
4. clinicaltrials.gov + AACT — trial outcomes for prior drugs against
   the target.
5. Customer-internal CSV/Parquet — drops into InternalEvidence as-is.

The orchestrator pattern: build_evidence_bundle() calls every fetcher
that's wired and merges results. Missing data stays missing (not zero).
"""

from __future__ import annotations

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
    """Topology fetcher backed by a PathwayCache. The cache itself is
    populated by an offline Reactome/OmniPath ingest pipeline.
    """

    def __init__(self, cache: PathwayCache):
        self._cache = cache

    def fetch(
        self, target: TargetID, disease: DiseaseContext
    ) -> PathwayTopologyEvidence:
        paralogs = self._cache.lookup_paralogs(target.gene_symbol)
        feedback = self._cache.lookup_feedback_loops(target.gene_symbol)
        redundancy = self._cache.lookup_downstream_redundancy(target.gene_symbol)
        return PathwayTopologyEvidence(
            paralog_count=paralogs.count if paralogs.count else None,
            paralogs=list(paralogs.paralogs),
            downstream_redundancy_index=redundancy,
            known_feedback_loops=feedback,
        )


class FakeTrialsFetcher:
    def __init__(self, by_target: Optional[dict[str, PriorTrialEvidence]] = None):
        self._table = by_target or {}

    def fetch(self, target: TargetID, disease: DiseaseContext) -> PriorTrialEvidence:
        return self._table.get(target.gene_symbol, PriorTrialEvidence())


# ── Orchestrator ──────────────────────────────────────────────────────────────


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
    """Assemble an EvidenceBundle by calling every wired fetcher.

    Missing fetchers leave the corresponding sub-bundle empty (default).
    InternalEvidence comes in pre-built from the customer ingest path.
    """
    return EvidenceBundle(
        genetics=genetics.fetch(target, disease) if genetics else GeneticsEvidence(),
        knockout=(
            knockout.fetch(target, disease)
            if knockout
            else KnockoutDependenceEvidence()
        ),
        topology=(
            topology.fetch(target, disease) if topology else PathwayTopologyEvidence()
        ),
        prior_trials=trials.fetch(target, disease) if trials else PriorTrialEvidence(),
        internal=internal or InternalEvidence(),
    )
