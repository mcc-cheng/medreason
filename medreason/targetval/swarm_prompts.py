"""Prompt builders for the targetval swarm.

Why this is its own module:
- Keeps ``swarm.py`` under the 500-line cap (per the Phase 2 plan).
- Lets tests inspect the rendered prompt without spinning up an agent.
- Centralises the 3-layer policy guardrails so every agent sees the
  same boilerplate.

The system prompt is intentionally **inline** in Python (not loaded via
``medreason.prompts.load_prompt``) so iterating on it doesn't require
touching ``PROMPTS_LOCK.json``. The leaderboard-scored prompts still
live under ``medreason/prompts/`` with the lock guarding them; this is
a Phase-2 swarm prompt, not a leaderboard prompt.
"""

from __future__ import annotations

from typing import Optional

from ..ontology.rule import ReasoningRule
from .case import TargetValidationCase
from .topology import BypassSignal


# ── Constants ────────────────────────────────────────────────────────────────


SYSTEM_PROMPT_TARGETVAL = (
    "You are a target-validation reviewer in a multi-agent swarm. You "
    "see one (target, disease) pair and the structured evidence about "
    "it. Produce a strict-JSON memo with these fields and nothing else:\n"
    "\n"
    "{\n"
    '  "priority_score": float in [0, 1],\n'
    '  "bypass_risk_score": float in [0, 1],\n'
    '  "predicted_bypass": one of '
    '"no_bypass_known" | "paralog_compensation" | "downstream_feedback" |'
    ' "alternative_pathway" | "microenvironment_rescue" | '
    '"resistance_mutation" | "pharmacokinetic_escape" | '
    '"other_validated" | "unknown",\n'
    '  "supporting_evidence": [string, ...],\n'
    '  "weakening_evidence": [string, ...],\n'
    '  "proposed_experiments": [string, ...],\n'
    '  "rationale": string (2-4 sentences),\n'
    '  "applied_rule_ids": [string, ...]  # subset of the retrieved rule '
    "ids you actually used\n"
    "}\n"
    "\n"
    "Hard rules:\n"
    "- Output JSON only. No prose, no markdown fences, no preamble.\n"
    "- priority_score is *target* quality (high = good candidate). "
    "bypass_risk_score is the probability the target will be bypassed "
    "in the disease context (high = bad).\n"
    "- If the bypass signals listed in the user message strongly "
    "suggest a mechanism, pick it as predicted_bypass and reflect it in "
    "bypass_risk_score. Do NOT default to 'unknown' when the signals "
    "are explicit.\n"
    "- Treat retrieved rules as advisory, not authoritative. Cite the "
    "rule ids you actually used in applied_rule_ids; omit ones you "
    "rejected.\n"
    "- 3-layer policy guardrails: never quote customer/internal evidence "
    "verbatim, never name a tenant, never invent paralog families or "
    "feedback loops that aren't in the bypass-signal list."
)


# ── User-prompt builder ──────────────────────────────────────────────────────


def build_user_prompt(
    case: TargetValidationCase,
    *,
    retrieved_rules: list[ReasoningRule],
    bypass_signals: list[BypassSignal],
) -> str:
    """Render the per-target user prompt.

    Sections (in order):
      1. Target identification
      2. Disease context
      3. Bypass signals (from the pathway cache, modality-filtered)
      4. Evidence summary (numbers, not raw blobs)
      5. Retrieved rules (advisory)
      6. Output schema reminder

    Notes:
    - InternalEvidence.readouts content is **not** rendered as keys/
      values; only the *fact* that internal data exists is mentioned.
      Rule extraction never sees customer payloads.
    - The bypass-signal section is the load-bearing one for the
      cross-agent analyzer's "missed bypass" detection — it lets the
      analyzer assert "the swarm SAW this signal and still scored bypass
      risk low".
    """
    parts: list[str] = []

    # ── 1. Target ──────────────────────────────────────────────────────────
    target = case.target
    parts.append("## Target")
    parts.append(f"- gene_symbol: {target.gene_symbol}")
    if target.family:
        parts.append(f"- family: {target.family}")
    if target.uniprot_accession:
        parts.append(f"- uniprot: {target.uniprot_accession}")
    if target.aliases:
        parts.append(f"- aliases: {', '.join(target.aliases)}")
    parts.append(f"- modality: {case.modality.value}")
    parts.append("")

    # ── 2. Disease ─────────────────────────────────────────────────────────
    disease = case.disease
    parts.append("## Disease context")
    parts.append(f"- label: {disease.disease_label}")
    if disease.therapeutic_area:
        parts.append(f"- therapeutic_area: {disease.therapeutic_area}")
    if disease.biomarker_context:
        parts.append(f"- biomarker: {disease.biomarker_context}")
    if disease.ontology_code:
        parts.append(f"- ontology: {disease.ontology_code}")
    parts.append("")

    # ── 3. Bypass signals ──────────────────────────────────────────────────
    parts.append("## Bypass signals (pathway-cache derived)")
    if not bypass_signals:
        parts.append(
            "- (none detected for this modality — absence of signal is "
            "NOT evidence of safety)"
        )
    else:
        for sig in bypass_signals:
            features = (
                f" features=[{', '.join(sig.contributing_features)}]"
                if sig.contributing_features
                else ""
            )
            parts.append(
                f"- mechanism={sig.mechanism}  strength={sig.strength:.2f}"
                f"  evidence: {sig.evidence_summary}{features}"
            )
    parts.append("")

    # ── 4. Evidence summary ────────────────────────────────────────────────
    ev = case.evidence
    parts.append("## Evidence summary")
    if ev.genetics.overall_score is not None:
        parts.append(
            f"- genetics.overall_score: {ev.genetics.overall_score:.3f}"
        )
    if ev.knockout.mean_dependency_score is not None:
        parts.append(
            "- knockout.mean_dependency_score: "
            f"{ev.knockout.mean_dependency_score:.3f}"
        )
        if ev.knockout.fraction_dependent_lines is not None:
            parts.append(
                "- knockout.fraction_dependent_lines: "
                f"{ev.knockout.fraction_dependent_lines:.3f}"
            )
        if ev.knockout.cell_line_context:
            parts.append(
                f"- knockout.context: {ev.knockout.cell_line_context}"
            )
    topo = ev.topology
    if topo.paralog_count is not None:
        parts.append(f"- topology.paralog_count: {topo.paralog_count}")
    if topo.paralogs:
        parts.append(f"- topology.paralogs: {', '.join(topo.paralogs)}")
    if topo.downstream_redundancy_index is not None:
        parts.append(
            "- topology.downstream_redundancy_index: "
            f"{topo.downstream_redundancy_index:.2f}"
        )
    if topo.known_feedback_loops:
        parts.append(
            "- topology.known_feedback_loops: "
            f"{', '.join(topo.known_feedback_loops)}"
        )
    if topo.reference_pathway:
        # cache_version snapshot id (see PathwayCache.cache_version)
        parts.append(
            f"- topology.snapshot: {topo.reference_pathway}"
        )
    pt = ev.prior_trials
    if pt.n_trials_prior or pt.n_phase2_failures_efficacy or pt.n_approvals:
        parts.append(
            "- prior_trials: "
            f"n_total={pt.n_trials_prior}, "
            f"n_p2_efficacy_fail={pt.n_phase2_failures_efficacy}, "
            f"n_p2_safety_fail={pt.n_phase2_safety_failures}, "
            f"n_approvals={pt.n_approvals}"
        )
    if pt.summary:
        parts.append(f"- prior_trials.summary: {pt.summary}")
    if ev.has_internal_data():
        # Mention *fact* of internal data, never its content.
        parts.append(
            "- internal_data_present: true (customer evidence redacted "
            "by 3-layer policy; do NOT echo its contents)"
        )
    if ev.additional_notes:
        parts.append(f"- notes: {ev.additional_notes}")
    parts.append("")

    # ── 5. Retrieved rules ─────────────────────────────────────────────────
    parts.append("## Retrieved rules (advisory)")
    if not retrieved_rules:
        parts.append("- (no rules retrieved)")
    else:
        for rule in retrieved_rules:
            polarity = rule.polarity
            predicate = rule.trigger.semantic_predicate or "(no predicate)"
            parts.append(
                f"- [{rule.rule_id}] polarity={polarity}  "
                f"predicate: {predicate}  action: {rule.action}"
            )
    parts.append("")

    # ── 6. Output schema reminder ──────────────────────────────────────────
    parts.append("## Output")
    parts.append(
        "Return ONLY the JSON memo described in the system message. "
        "No prose, no fences."
    )
    return "\n".join(parts)


# ── Optional: repair-prompt builder (kept here in case retry lands) ─────────


def build_repair_user_prompt(
    original_user: str,
    *,
    raw_response: str,
    parse_error: str,
) -> Optional[str]:
    """Currently unused — kept for a future retry-on-malformed path.

    The architect's spec for v0.2 says the parser is **tolerant** — it
    only raises ``MemoParseError`` if no JSON can be extracted at all,
    and otherwise builds a memo with ``parse_warnings``. So a repair
    prompt is not on the critical path. This helper exists so a future
    builder can add a single retry without re-deriving the prompt
    shape.
    """
    return (
        f"{original_user}\n\n"
        "## Previous attempt\n"
        f"Your last response could not be parsed: {parse_error}\n"
        f"Raw response was:\n{raw_response[:500]}\n\n"
        "Re-emit the memo as strict JSON only. No fences, no prose."
    )
