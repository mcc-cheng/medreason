"""Rule abstractor — rewrites case-specific rules into concept-level rules.

Extracted rules often reference specific drugs, diagnoses, or procedures.
This module asks an LLM to rewrite the action, semantic_predicate, and
rationale so the rule describes the GENERAL PRINCIPLE, not the specific
case. The trigger enums, citation, and polarity are preserved.

This sits between extraction (propose_rules / analyze_failure) and
storage (rule_store.put). The caller gets back a new ReasoningRule with
the same rule_id but abstracted text fields.
"""

from __future__ import annotations

import copy
from typing import Optional
from uuid import uuid4

from ..llm.base import LLMClient
from ..ontology.rule import ReasoningRule
from ..prompts import load_prompt
from ..runners._prompting import ResponseParseError, parse_json_response


ABSTRACTOR_PROMPT_FILE = "rule_abstractor.txt"


def abstract_rule(
    rule: ReasoningRule,
    llm_client: LLMClient,
    *,
    max_tokens: int = 1024,
    seed: int = 0,
) -> ReasoningRule:
    """Rewrite a rule's text fields to be concept-level.

    Returns a new ReasoningRule with the same rule_id, trigger enums,
    evidence, and polarity, but with abstracted action,
    semantic_predicate, and rationale. If the LLM call fails or returns
    unparseable output, returns the original rule unchanged.
    """
    system_prompt = load_prompt(ABSTRACTOR_PROMPT_FILE)

    user_parts = [
        "## Original rule",
        f"action: {rule.action}",
        f"semantic_predicate: {rule.trigger.semantic_predicate}",
        f"rationale: {rule.rationale}",
        f"polarity: {rule.polarity}",
        f"source_policy_citation: {rule.evidence.source_policy_citation}",
        "",
        "## Trigger context",
        f"cpt_families: {[f.value for f in rule.trigger.cpt_families]}",
        f"icd10_chapters: {[c.value for c in rule.trigger.icd10_chapters]}",
        f"payers: {[p.value for p in rule.trigger.payers]}",
    ]

    user_msg = "\n".join(user_parts)

    try:
        response = llm_client.complete(
            system=system_prompt,
            user=user_msg,
            max_tokens=max_tokens,
            seed=seed,
        )
    except Exception:
        return rule

    raw = response.text or ""
    try:
        data = parse_json_response(raw)
    except ResponseParseError:
        return rule

    new_action = str(data.get("abstracted_action", "")).strip()
    new_predicate = str(data.get("abstracted_semantic_predicate", "")).strip()
    new_rationale = str(data.get("abstracted_rationale", "")).strip()

    if not new_action or not new_predicate:
        return rule

    abstracted = copy.deepcopy(rule)
    abstracted.action = new_action
    abstracted.trigger.semantic_predicate = new_predicate
    abstracted.rationale = new_rationale

    return abstracted
