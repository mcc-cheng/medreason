"""Phase 5 extraction — critic verification, rule proposal, generalization gate.

These modules turn a successful agent run into a set of validated
`ReasoningRule` objects that can be stored in the memory pipeline.

Flow:

    agent.run(case) → if correct:
        critic.run_critic(case, agent_determination) → CriticResult
        if critic agrees:
            store verified trace
            rule_proposer.propose_rules(trace, policy) → candidates
            for cand in candidates:
                generalization_gate.validate(cand) → ACTIVE or CANDIDATE or DEPRECATED

The generalization gate (Phase 5 commit 2) is the single load-bearing
quality lever: it re-tests each candidate rule against held-out train/dev
cases before promoting it to ACTIVE. Without the gate, the pipeline
degenerates to "store whatever the proposer suggested" — the old
pipeline's failure mode.
"""

from .critic import CriticResult, run_critic
from .failure_analyzer import analyze_failure
from .generalization_gate import (
    DEFAULT_THRESHOLDS,
    GateResult,
    GateThresholds,
    GeneralizationGate,
)
from .rule_abstractor import abstract_rule
from .rule_proposer import (
    PatientIdentifierError,
    PolicyCitationError,
    ProposalRejection,
    ProposalResult,
    propose_rules,
)

__all__ = [
    "CriticResult",
    "run_critic",
    "ProposalResult",
    "ProposalRejection",
    "PolicyCitationError",
    "PatientIdentifierError",
    "propose_rules",
    "analyze_failure",
    "abstract_rule",
    "GeneralizationGate",
    "GateResult",
    "GateThresholds",
    "DEFAULT_THRESHOLDS",
]
