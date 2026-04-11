"""Phase 5 retrieval — ontology lookup + dense embedding + reranking,
plus the structured rule injector that turns a list of retrieved rules
into a `system_extra` block for the AgentRunner Protocol.

The old `medreason/injector.py` is kept untouched (the pre-rework
agent.py/benchmark.py still import from it). Everything new lives here.
"""

from .injector import (
    ACTION_MAX_WORDS,
    APPLIED_RULES_FIELD,
    build_rule_checklist,
    missing_rule_ids,
    parse_applied_rules,
)

__all__ = [
    "ACTION_MAX_WORDS",
    "APPLIED_RULES_FIELD",
    "build_rule_checklist",
    "parse_applied_rules",
    "missing_rule_ids",
]
