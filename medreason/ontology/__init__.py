"""medreason.ontology — domain models for the memory pipeline.

This package replaces the legacy single-file `medreason/ontology.py`. Every
name the pre-rework modules used to import is re-exported here so existing
extractor / generator / agent / store / injector / benchmark / local_cases
keep working unchanged until later phases replace them.

New code should import the specific submodule (e.g. `from
medreason.ontology.rule import ReasoningRule`) but importing from the
package root also works.
"""

from .codes import (
    CPTFamily,
    ICD10Chapter,
    cpt_families,
    cpt_family,
    icd10_chapter,
    icd10_chapters,
)
from .case import (
    BenchmarkCase,
    DenialReason,
    Difficulty,
    FacilityType,
    Outcome,
    Payer,
    PriorAuthTaskConfig,
)
from .trace import (
    ArgumentFraming,
    ArgumentStructure,
    ReasoningPattern,  # legacy shim — see trace.py
    ReasoningStep,
    ReasoningTrace,
)
from .rule import (
    ReasoningRule,
    RuleEvidence,
    RuleStatus,
    RuleTrigger,
)
from .result import (
    AgentResult,
    AppliedRule,
    BenchmarkResult,
)

__all__ = [
    # codes
    "CPTFamily",
    "ICD10Chapter",
    "cpt_family",
    "cpt_families",
    "icd10_chapter",
    "icd10_chapters",
    # case
    "Payer",
    "FacilityType",
    "DenialReason",
    "Outcome",
    "Difficulty",
    "PriorAuthTaskConfig",
    "BenchmarkCase",
    # trace
    "ArgumentFraming",
    "ReasoningStep",
    "ArgumentStructure",
    "ReasoningTrace",
    "ReasoningPattern",  # legacy
    # rule (new)
    "RuleStatus",
    "RuleTrigger",
    "RuleEvidence",
    "ReasoningRule",
    # result
    "AppliedRule",
    "AgentResult",
    "BenchmarkResult",
]
