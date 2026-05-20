"""medreason.targetval — target validation domain on top of the MedReason core.

The target-validation task: given a disease + a list of candidate proteins,
return a prioritized short-list with a decision memo per target covering
supporting evidence, disconfirming evidence, bypass-mechanism risk, and the
cheapest experiments to prove/disprove.

This package is layered ON TOP of medreason/ primitives:

- ReasoningRule + RuleStore             → reused unchanged
- 3-tier retrieval (ontology/dense/rerank) → reused, with new structural
  trigger fields (gene/family/pathway/disease) wrapped via TargetRuleTrigger
- failure_analyzer + rule_abstractor    → reused, plus a cross-agent variant
  in cross_agent_analyzer.py that extracts patterns when the WHOLE swarm
  misses a class of bypass
- MemoryRunner                          → reused as the per-target agent;
  swarm.SwarmRunner runs N of them in parallel

The architectural addition is the 3-layer memory split:

- UniversalLayer   — mechanism rules trained only on public retrospectives
- DiseaseLayer     — disease/pathway-scoped rules; selected by disease tag
- CampaignLayer    — per-customer, never leaves the customer tenant

The layer split is enforced by LayerRouter + a per-layer LeakGuard policy.
Universal rules cannot reference customer-derived case_ids; campaign rules
cannot reference other customers' case_ids.

Status: SKELETON. No LLM calls wired. Heavy logic (LLM-driven extraction,
real evidence ingest, pathway topology computation) is left as
NotImplementedError stubs to be filled in after the architecture is
reviewed.
"""

from .case import (
    DiseaseContext,
    GroundTruthOutcome,
    Modality,
    TargetID,
    TargetValidationCase,
)
from .evidence import (
    EvidenceBundle,
    GeneticsEvidence,
    InternalEvidence,
    KnockoutDependenceEvidence,
    PathwayTopologyEvidence,
    PriorTrialEvidence,
)
from .layers import Layer, LayerPolicy, LayerRouter
from .topology import BypassSignal, ParalogFeature

__all__ = [
    "BypassSignal",
    "DiseaseContext",
    "EvidenceBundle",
    "GeneticsEvidence",
    "GroundTruthOutcome",
    "InternalEvidence",
    "KnockoutDependenceEvidence",
    "Layer",
    "LayerPolicy",
    "LayerRouter",
    "Modality",
    "ParalogFeature",
    "PathwayTopologyEvidence",
    "PriorTrialEvidence",
    "TargetID",
    "TargetValidationCase",
]
