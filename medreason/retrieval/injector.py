"""Structured rule injector — turns retrieved ReasoningRules into a
`system_extra` block for the AgentRunner Protocol.

This REPLACES the decorative markdown of the pre-rework
`medreason/injector.py`. Key differences:

- Each rule is presented with a STABLE `rule_id` so the agent can
  reference it by name.
- The injection extends the output schema with an `applied_rules`
  contract: for every retrieved rule_id, the agent must emit
  `{rule_id, applied: bool, rationale}`. Missing rule_ids trigger a
  re-prompt in the memory wrapper (Phase 5i).
- The injection block is placed ABOVE the frozen `system_pa.txt`, so
  the adversarial reviewer instructions still apply at the base layer
  while memory sets the stage.
- There is no softening language like "use as guidance, do not copy
  verbatim". The agent is told to apply the rules explicitly.

The parse_applied_rules() function is the symmetric reader — given the
agent's raw JSON response, it extracts and validates the applied_rules
field. missing_rule_ids() is the set-difference helper used by the
wrapper to decide whether to re-prompt.
"""

from __future__ import annotations

import json
from typing import Iterable

from ..ontology.result import AppliedRule
from ..ontology.rule import ReasoningRule

# NOTE: we deliberately do NOT import from medreason.runners at module
# top level. medreason.runners.__init__ imports memory_wrapper which
# imports this very module, so a top-level import here would create a
# circular dependency that crashes whichever side gets imported first.
# parse_applied_rules() lazy-imports the JSON helpers when called.


# Mirrors medreason.extraction.rule_proposer.ACTION_MAX_WORDS for
# convenience when higher layers want to reference the constant from
# a retrieval-side import path. Keeping both definitions DRY by not
# cross-importing: the constant is a contract, not a runtime dependency.
ACTION_MAX_WORDS = 25

# The field name the agent must emit for per-rule application.
APPLIED_RULES_FIELD = "applied_rules"


# ── Checklist builder ────────────────────────────────────────────────────────


def build_rule_checklist(
    rules: list[ReasoningRule],
    *,
    compact: bool = True,
) -> str:
    """Build the structured injection block. Returns "" when rules is empty.

    Two formats:

    - **compact (default)**: one line per rule (`[rule_id] ACTION`),
      minimal output contract that asks only for `{rule_id, applied}`
      with rationale optional. Designed to keep the injection token
      cost under control. ~15 tokens per rule + ~80 token footer.

    - **verbose** (compact=False): the original multi-line format with
      trigger summary, posterior, polarity, rationale, citation, and
      semantic predicate. Useful for debugging and audit but ~60+
      tokens per rule. The output contract requires a `rationale` field
      per applied rule.

    The returned string is meant to be passed to
    `AgentRunner.run(case, system_extra=<this>)`. Empty input returns an
    empty string, which the runner treats as "no injection" — so
    retrieval returning zero results naturally degrades to zero-shot.
    """
    if not rules:
        return ""

    if compact:
        return _build_compact(rules)
    return _build_verbose(rules)


def _build_compact(rules: list[ReasoningRule]) -> str:
    lines = [
        "=== REASONING MEMORY ===",
        f"{len(rules)} rule(s) retrieved from prior cases. Apply each one "
        "to this case. The clinical notes either satisfy a rule or they "
        "do not — do not paraphrase a rule away.",
        "",
    ]
    for rule in rules:
        lines.append(f"[{rule.rule_id}] {rule.action}")
    lines.extend([
        "",
        "OUTPUT CONTRACT: Add this field to your standard determination JSON:",
        f'"{APPLIED_RULES_FIELD}": [{{"rule_id":"...","applied":true}},'
        ' ...]',
        "Emit one entry per rule_id above. `applied`=true means the rule's "
        "criterion is satisfied by the notes.",
        "=== END MEMORY ===",
    ])
    return "\n".join(lines)


def _build_verbose(rules: list[ReasoningRule]) -> str:
    header = [
        "=== INSTITUTIONAL REASONING MEMORY ===",
        f"You have been given {len(rules)} reasoning rule(s) retrieved from",
        "institutional memory for cases like this one. Apply each rule "
        "explicitly.",
        "For every rule, cite whether the clinical notes satisfy the "
        "check — do not paraphrase a rule away.",
        "",
    ]

    rule_blocks: list[str] = []
    for rule in rules:
        rule_blocks.extend(_format_rule_verbose(rule))
        rule_blocks.append("")

    footer = [
        "=== OUTPUT CONTRACT ===",
        "In addition to the standard determination JSON schema in the "
        "system prompt below, you MUST also include an "
        f'"{APPLIED_RULES_FIELD}" field:',
        "",
        f'"{APPLIED_RULES_FIELD}": [',
        '  {"rule_id": "<exact rule_id from above>",',
        '   "applied": true | false,',
        '   "rationale": "short explanation"},',
        "  ...",
        "]",
        "",
        f"Emit exactly ONE entry per rule_id above. Missing rule_ids "
        "will fail validation and trigger a re-prompt.",
        "=== END INSTITUTIONAL MEMORY ===",
    ]

    return "\n".join(header + rule_blocks + footer)


def _format_rule_verbose(rule: ReasoningRule) -> list[str]:
    # Display rule: use .value for Payer (title-case brand names) and
    # .name.lower() for CPTFamily / ICD10Chapter / FacilityType — the
    # latter two have unnatural values ("M00-M99", "ASC") so the
    # member name gives a more readable, prompt-friendly label.
    trigger_parts: list[str] = []
    if rule.trigger.payers:
        trigger_parts.append(
            "payer=" + ", ".join(p.value for p in rule.trigger.payers)
        )
    if rule.trigger.cpt_families:
        trigger_parts.append(
            "cpt_family=" + ", ".join(
                f.name.lower() for f in rule.trigger.cpt_families
            )
        )
    if rule.trigger.cpt_codes:
        trigger_parts.append(
            "cpt=" + ", ".join(rule.trigger.cpt_codes)
        )
    if rule.trigger.icd10_chapters:
        trigger_parts.append(
            "icd_chapter=" + ", ".join(
                c.name.lower() for c in rule.trigger.icd10_chapters
            )
        )
    if rule.trigger.icd10_prefixes:
        trigger_parts.append(
            "icd_prefix=" + ", ".join(rule.trigger.icd10_prefixes)
        )
    if rule.trigger.facility_types:
        trigger_parts.append(
            "facility=" + ", ".join(
                f.name.lower() for f in rule.trigger.facility_types
            )
        )
    trigger_str = " | ".join(trigger_parts) if trigger_parts else "any"

    lines = [
        f"[{rule.rule_id}] (posterior={rule.posterior_mean:.2f}, n={rule.trials})",
        f"  TRIGGER:  {trigger_str}",
        f"  ACTION:   {rule.action}",
        f"  POLARITY: {rule.polarity}",
    ]
    if rule.rationale:
        lines.append(f"  RATIONALE: {rule.rationale}")
    if rule.evidence.source_policy_citation:
        lines.append(f"  CITATION: {rule.evidence.source_policy_citation}")
    if rule.trigger.semantic_predicate:
        lines.append(f"  APPLIES WHEN: {rule.trigger.semantic_predicate}")
    return lines


# Backward-compat alias used in retrieval/dense.py
_format_rule = _format_rule_verbose


# ── Response-side parsing ────────────────────────────────────────────────────


def parse_applied_rules(raw_response_text: str) -> list[AppliedRule]:
    """Extract an `applied_rules` list from an agent response.

    Returns an empty list if:
    - the response has no parseable JSON at all,
    - the JSON does not contain an `applied_rules` key,
    - `applied_rules` is not a list.

    Individual malformed entries inside the list are silently dropped
    so a partially-valid response still yields partial data. The
    memory wrapper uses `missing_rule_ids()` on the returned list to
    decide whether to re-prompt.
    """
    # Lazy import to avoid the retrieval ↔ runners cycle. See module docstring.
    from ..runners._prompting import (  # noqa: PLC0415
        ResponseParseError,
        parse_json_response,
    )

    if not raw_response_text:
        return []
    try:
        data = parse_json_response(raw_response_text)
    except ResponseParseError:
        return []

    raw_list = data.get(APPLIED_RULES_FIELD)
    if not isinstance(raw_list, list):
        return []

    out: list[AppliedRule] = []
    for entry in raw_list:
        if not isinstance(entry, dict):
            continue
        rule_id = entry.get("rule_id")
        applied = entry.get("applied")
        if not isinstance(rule_id, str) or not rule_id:
            continue
        if not isinstance(applied, bool):
            # Tolerate "true"/"false" strings from sloppy models.
            if isinstance(applied, str):
                if applied.strip().lower() == "true":
                    applied = True
                elif applied.strip().lower() == "false":
                    applied = False
                else:
                    continue
            else:
                continue
        rationale = entry.get("rationale", "")
        if not isinstance(rationale, str):
            rationale = str(rationale)
        out.append(AppliedRule(
            rule_id=rule_id,
            applied=applied,
            rationale=rationale,
        ))
    return out


def missing_rule_ids(
    retrieved_rule_ids: Iterable[str],
    applied: list[AppliedRule],
) -> list[str]:
    """Return the retrieved rule_ids the agent failed to address.

    Used by the memory wrapper to decide whether to re-prompt. The
    returned list preserves the original retrieval order so error
    messages are deterministic.
    """
    present = {a.rule_id for a in applied}
    out: list[str] = []
    for rid in retrieved_rule_ids:
        if rid not in present:
            out.append(rid)
    return out
