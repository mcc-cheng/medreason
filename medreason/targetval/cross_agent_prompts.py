"""Prompt builder for the cross-agent corrective-rule proposer.

The cross-agent analyzer detects SYSTEMATIC reasoning failures across
many independent SwarmAgents (one agent per target). When it sees a
miss-pattern (e.g. swarm under-scored bypass risk on 7/8 cases with
ground-truth paralog compensation), it asks an LLM what UNIVERSAL-layer
rule would have prevented that miss-pattern.

This module owns the prompt for that LLM call. Three invariants live here:

1. **De-identification.** The user message lists only opaque
   ``case_id`` strings — no gene symbols, no customer tags, no notes.
   The corrective rule is meant for the UNIVERSAL layer, so the LLM
   must not be primed with any case-specific identifiers it could
   regurgitate into ``action`` / ``rationale`` text.

2. **No ``InternalEvidence.readouts`` ever.** The plan's hard
   constraint: even when called from a retro-fixture path that has no
   customer data, the prompt builder must defensively skip the
   ``InternalEvidence`` bundle. The leak-guard for the universal layer
   rejects any rule whose provenance touches a tenant; the prompt-
   builder is the upstream filter that keeps tenant strings out of the
   LLM context window in the first place.

3. **Structural cross-case framing.** The system prompt asks for
   structural predicates (``"kinase target with paralog_count >= 2"``)
   rather than specific gene symbols. The downstream
   ``generalization_gate`` checks the rule against held-out targets;
   a too-specific predicate fails that gate by construction.
"""

from __future__ import annotations

from typing import Iterable, Optional

from .case import TargetValidationCase
from .swarm import TargetMemo


SYSTEM_PROMPT_CROSS_AGENT = """You are the cross-agent metacognitive analyzer for a target-validation swarm.

A swarm of independent SwarmAgents each reasoned about one (target, disease)
pair and emitted a TargetMemo. The cross-agent analyzer grouped their outputs
against retrospective ground truth and identified a SYSTEMATIC reasoning
miss: a pattern the swarm gets wrong across many independent cases.

Your job: propose UNIVERSAL-layer corrective rule(s) that, if applied at
inference time, would have prevented that miss-pattern.

OUTPUT FORMAT (strict JSON, no prose):
{
  "rules": [
    {
      "semantic_predicate": "structural condition that triggers the rule",
      "action": "≤25 words, one atomic check or score adjustment",
      "rationale": "1-2 sentences for human auditors",
      "polarity": "supports_approval" | "supports_denial" | "requires_check"
    }
  ]
}

HARD RULES:
- "action" MUST be ≤25 words. The injector renders it verbatim.
- NEVER mention patient identifiers (ages, dates, "Mr./Mrs./Dr. X",
  "patient John", "58 y/o", ISO dates). The candidate is rejected if
  any of these strings appear in any field.
- PREFER structural predicates ("kinase target with paralog_count >= 2")
  over specific gene symbols ("KRAS"). Specific symbols fail the
  downstream generalization gate.
- "rules" MAY be an empty list if the miss-pattern is too noisy to
  generalise. Returning an empty list is preferred over emitting a
  bad rule.
- Return ONLY the JSON object. No markdown fences, no commentary.
"""


REPAIR_SUFFIX = (
    "\n\nThe previous response was not valid JSON matching the required "
    "schema. Re-emit ONLY the JSON object: "
    '{"rules": [{"semantic_predicate": "...", "action": "...", '
    '"rationale": "...", "polarity": "requires_check"}]}. '
    "No prose, no markdown fences."
)


def _safe_mean(values: Iterable[float]) -> Optional[float]:
    xs = list(values)
    if not xs:
        return None
    return sum(xs) / len(xs)


def build_proposer_prompt(
    error,  # SystematicError — kept untyped to avoid an import cycle
    cases: list[TargetValidationCase],
    memos: Optional[list[TargetMemo]] = None,
) -> str:
    """Render the de-identified user message for one SystematicError.

    Parameters
    ----------
    error
        The SystematicError to propose a rule for. Only its
        ``error_kind``, ``severity``, ``description`` and
        ``affected_case_ids`` are surfaced — never the underlying
        ``InternalEvidence``.
    cases
        Full case list. Used ONLY to look up the affected case_ids by
        identity. The function reads ``case.case_id`` and the structural
        ground-truth fields (``ground_truth_bypass``,
        ``ground_truth_outcome``). It explicitly does NOT touch
        ``case.evidence.internal.readouts`` or ``case.evidence.internal.notes``.
    memos
        Optional swarm memos for the affected cases. If provided, the
        prompt includes the swarm's average ``bypass_risk_score`` plus
        any ``bypass_signals_seen`` the swarm noted-but-ignored — a
        powerful pointer for the LLM ("you saw the signal yet ranked
        the case low"). Fallback memos (those whose ``parse_warnings``
        contain a ``memo_parse_error`` marker) are skipped to keep a
        single misbehaving LLM call from biasing rule extraction.
    """
    affected_ids = set(error.affected_case_ids)
    affected_cases = [c for c in cases if c.case_id in affected_ids]

    parts: list[str] = []
    parts.append("## Systematic reasoning miss")
    parts.append(f"error_kind: {error.error_kind}")
    parts.append(f"severity: {error.severity:.3f}")
    parts.append(f"n_affected_cases: {len(error.affected_case_ids)}")
    if error.description:
        parts.append(f"description: {error.description}")
    parts.append("")

    parts.append("## Affected cases (opaque IDs)")
    for case in affected_cases:
        bypass = case.ground_truth_bypass.value
        outcome = case.ground_truth_outcome.value
        # Structural ground-truth fields only — NEVER evidence.internal.
        parts.append(
            f"- case_id={case.case_id} "
            f"ground_truth_bypass={bypass} "
            f"ground_truth_outcome={outcome}"
        )
    if not affected_cases:
        # Defensive: if cases lookup misses, still list opaque ids so the
        # LLM can count them.
        for cid in error.affected_case_ids:
            parts.append(f"- case_id={cid}")
    parts.append("")

    # ── Swarm-side context (memos), filtered to drop fallback memos. ─────
    if memos is not None:
        clean_memos = [
            m
            for m in memos
            if m.case_id in affected_ids
            and not _is_fallback_memo(m)
        ]
        if clean_memos:
            avg_bypass = _safe_mean(m.bypass_risk_score for m in clean_memos)
            avg_priority = _safe_mean(m.priority_score for m in clean_memos)
            parts.append("## Swarm behaviour on affected cases")
            if avg_bypass is not None:
                parts.append(f"avg_bypass_risk_score: {avg_bypass:.3f}")
            if avg_priority is not None:
                parts.append(f"avg_priority_score: {avg_priority:.3f}")

            # Aggregate the signals the swarm saw but failed to act on.
            seen_signals: set[str] = set()
            for m in clean_memos:
                for sig in m.bypass_signals_seen:
                    seen_signals.add(sig)
            if seen_signals:
                parts.append(
                    "signals_seen_but_underweighted: "
                    + ", ".join(sorted(seen_signals))
                )
            parts.append("")

    parts.append("## Task")
    parts.append(
        "Propose UNIVERSAL-layer corrective rule(s) that would have "
        "flipped the swarm's scoring on these cases. Return the strict "
        "JSON shape from the system prompt. If the miss-pattern is too "
        "noisy to generalise, return {\"rules\": []}."
    )

    return "\n".join(parts)


def _is_fallback_memo(memo: TargetMemo) -> bool:
    """True if a memo is the synthetic fallback emitted when the per-agent
    LLM call returned unparseable output. We skip these so a single
    misbehaving call can't drive cross-case rule extraction.
    """
    for w in memo.parse_warnings:
        if isinstance(w, str) and w.startswith("memo_parse_error"):
            return True
    return False
