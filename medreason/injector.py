"""Injection middleware — retrieves and formats patterns for agent context."""

from __future__ import annotations

from .ontology import PriorAuthTaskConfig, ReasoningPattern
from .store import PatternStore


def build_injection_context(
    patterns: list[ReasoningPattern],
) -> str:
    """Build the institutional memory injection block."""
    if not patterns:
        return ""

    lines = [
        "=== INSTITUTIONAL REASONING MEMORY ===",
        f"Found {len(patterns)} relevant resolution patterns from prior cases.",
        "",
    ]

    for i, p in enumerate(patterns, 1):
        total = p.success_count + p.failure_count
        rate = p.success_rate
        lines.append(f"--- Pattern {i} (success rate: {rate:.0%}, n={total}) ---")
        lines.append(f"Payer: {p.task_config.payer.value} | CPT: {p.task_config.cpt_code}")
        lines.append(f"Framing: {p.argument_structure.framing.value}")
        lines.append(
            f"Clinical evidence: {', '.join(p.argument_structure.clinical_evidence_elements)}"
        )
        lines.append(
            f"Policy hooks: {', '.join(p.argument_structure.payer_policy_hooks)}"
        )
        lines.append("Reasoning approach:")
        for j, step in enumerate(p.key_reasoning_steps, 1):
            lines.append(f"  {j}. {step}")
        lines.append("")

    lines.extend([
        "Use these patterns as guidance. Adapt to the specific case — do not copy verbatim.",
        "If patterns conflict with current clinical evidence, prioritize the evidence.",
        "=== END INSTITUTIONAL MEMORY ===",
    ])

    return "\n".join(lines)


def retrieve_and_inject(
    store: PatternStore,
    task_config: PriorAuthTaskConfig,
    top_k: int = 3,
) -> tuple[str, list[str]]:
    """Retrieve patterns and build injection context.

    Returns (injection_text, list_of_pattern_ids).
    """
    patterns = store.retrieve_patterns(task_config, top_k=top_k)
    injection = build_injection_context(patterns)
    pattern_ids = [p.pattern_id for p in patterns]

    # Log retrieval
    if patterns:
        store._log_event(
            "retrieve_and_inject",
            injected_ids=pattern_ids,
            details={
                "config_hash": task_config.config_hash,
                "num_patterns": len(patterns),
            },
        )

    return injection, pattern_ids
