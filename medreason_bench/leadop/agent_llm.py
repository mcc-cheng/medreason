"""LLM-backed lead-op decision agent.

`propose_llm(ctx, use_memory, llm, seed)` builds a system + user prompt,
calls the Claude LLM, parses a JSON response into a `LeadOpDecision`.
Two configurations:

- OFF (baseline RAG): system prompt frames the agent as a lead-op
  advisor; user prompt includes campaign context + ChEMBL target-activity
  RAG corpus + SAR paper abstract. No memory_rules visible.
- ON (memory-augmented): same as OFF, plus a "Prior decisions in this
  campaign" block listing outcome-labeled prior DP rationales.

The LLM output schema is strict JSON:
  {"direction_ranking": [..4 strings..], "compound_ranking": [..cids..],
   "rationale": "short text"}
Parser is forgiving: accepts JSON wrapped in markdown code fences, tries
to repair missing directions by padding from ALL_DIRECTIONS, errors hard
only if no valid JSON object can be extracted.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from medreason.llm.base import LLMClient

from .harness import (
    ALL_DIRECTIONS,
    Direction,
    DpRationale,
    LeadOpContext,
    LeadOpDecision,
)
from .schema import CompoundRow


SYSTEM_PROMPT = """\
You are a lead-optimization advisor reviewing a SAR campaign
retrospectively. At each decision point, choose the next round's
direction by identifying THE BIGGEST REMAINING BOTTLENECK across the
four dimensions:

- potency:      on-target IC50 still unacceptable (typically >100 nM)
- selectivity:  on-target potency good but selectivity_ratio vs related
                targets is low (<10x) or off-target engagement is bad
- ADMET:        potency+selectivity acceptable but metabolism (high CLint,
                CYP3A4 TDI), safety (hERG IC50 < 10 uM), or permeability
                (Caco-2 Papp < 3) is blocking further progression
- scaffold_hop: no progress on any dimension is achievable in the current
                chemotype; a core scaffold change is required

Rules of thumb:
- Do NOT keep picking the same direction once the assays show it's
  already solved — pick the next bottleneck instead.
- If prior decisions in this campaign succeeded at direction X, treat
  X as LESS urgent, not more. Move to what's still broken.
- ADMET becomes the bottleneck late, after potency and selectivity are
  both in-range, which is when visible compounds show high CLint or
  CYP3A4 TDI flags.

Output ranks:
1. direction_ranking — all four directions, most → least urgent.
2. compound_ranking — every compound_id from the candidate pool exactly
   once, best synthesis-priority → worst, given your chosen direction.
3. rationale — one sentence naming the bottleneck you identified.

Return strict JSON and nothing else. Schema:
{"direction_ranking": ["<dir>", "<dir>", "<dir>", "<dir>"],
 "compound_ranking": ["<cid>", ...],
 "rationale": "<one sentence>"}
"""


def _serialize_compound(c: CompoundRow, *, include_outcome: bool) -> str:
    parts = [
        f"{c.compound_id} (round {c.round_index})",
        f"SMILES: {c.smiles}",
        f"MW={c.mw:.1f} cLogP={c.clogp:.2f} TPSA={c.tpsa:.1f} HBD={c.hbd} HBA={c.hba}"
        if c.mw is not None
        else "descriptors: n/a",
    ]
    if c.proposed_modification:
        parts.append(f"modification: {c.proposed_modification}")
    if c.assay_readouts:
        parts.append(f"assays: {json.dumps(c.assay_readouts, sort_keys=True)}")
    if include_outcome and c.outcome_label:
        parts.append(f"outcome: {c.outcome_label}")
    return " | ".join(parts)


def _serialize_rationale(r: DpRationale) -> str:
    note = f" — {r.note}" if r.note else ""
    return (
        f"{r.decision_point_id} (round {r.round_index}): chose "
        f"{r.direction_chosen} → outcome={r.outcome}{note}"
    )


# Thresholds the digest uses to label each dimension "IN RANGE" or
# "STILL SHORT". Matches the targets named in SYSTEM_PROMPT.
_POTENCY_TARGET_NM = 100.0
_SELECTIVITY_TARGET = 10.0
_HERG_TARGET_UM = 10.0
_CACO2_TARGET = 3.0
_CLINT_TARGET = 50.0  # uL/min/mg


def _memory_digest(ctx: LeadOpContext) -> str:
    """Unmet-targets digest: compressed state across the four direction
    axes, derived from visible compound outcomes. This IS the
    metacognitive memory — a baseline agent seeing raw compounds must
    re-derive this state at every DP; a memory-augmented agent gets it
    pre-digested from prior rounds' assay trajectories."""
    def _vals(key):
        return [
            c.assay_readouts.get(key)
            for c in ctx.visible_compounds
            if c.assay_readouts.get(key) is not None
        ]

    lines: list[str] = []

    ic50s = [v for v in _vals("ic50_nm") if isinstance(v, (int, float))]
    if ic50s:
        best = min(ic50s)
        status = "IN RANGE" if best <= _POTENCY_TARGET_NM else "STILL SHORT"
        lines.append(
            f"- potency:     best IC50 so far = {best:.1f} nM (target <{_POTENCY_TARGET_NM:.0f}) [{status}]"
        )

    sels = [v for v in _vals("selectivity_ratio") if isinstance(v, (int, float))]
    if sels:
        best = max(sels)
        status = "IN RANGE" if best >= _SELECTIVITY_TARGET else "STILL SHORT"
        lines.append(
            f"- selectivity: best ratio so far = {best:.1f}x (target >{_SELECTIVITY_TARGET:.0f}x) [{status}]"
        )

    hergs = [v for v in _vals("herg_ic50_um") if isinstance(v, (int, float))]
    clints = [v for v in _vals("clint_ul_min_mg") if isinstance(v, (int, float))]
    tdis = [v for v in _vals("cyp3a4_tdi") if isinstance(v, bool)]
    if hergs or clints or tdis:
        bits = []
        if hergs:
            worst_h = min(hergs)
            h_stat = "ok" if worst_h >= _HERG_TARGET_UM else "liability"
            bits.append(f"hERG worst={worst_h:.1f} uM ({h_stat})")
        if clints:
            worst_c = max(clints)
            c_stat = "ok" if worst_c <= _CLINT_TARGET else "liability"
            bits.append(f"CLint worst={worst_c:.1f} ({c_stat})")
        if tdis:
            n_tdi = sum(1 for t in tdis if t)
            bits.append(f"CYP3A4 TDI flagged={n_tdi}/{len(tdis)}")
        admet_short = (
            (hergs and min(hergs) < _HERG_TARGET_UM)
            or (clints and max(clints) > _CLINT_TARGET)
            or any(tdis)
        )
        status = "STILL SHORT" if admet_short else "IN RANGE"
        lines.append("- ADMET:       " + "; ".join(bits) + f" [{status}]")

    # Prior-DP direction history
    if ctx.visible_rationales:
        tried = ", ".join(
            f"{r.direction_chosen}({r.outcome})" for r in ctx.visible_rationales
        )
        lines.append(f"- directions already tried: {tried}")

    return "\n".join(lines) if lines else "(no memory digest yet)"


def build_user_prompt(
    ctx: LeadOpContext,
    *,
    use_memory: bool,
    rag_corpus: str | None = None,
) -> str:
    lines: list[str] = []
    lines.append(
        f"CAMPAIGN: {ctx.campaign_id}   DECISION POINT: {ctx.decision_point_id}"
    )
    lines.append(f"Choosing direction for ROUND {ctx.round_index}.")
    lines.append("")

    if rag_corpus:
        lines.append("=== RAG CORPUS (target & SAR literature) ===")
        lines.append(rag_corpus.strip())
        lines.append("")

    lines.append("=== VISIBLE COMPOUND HISTORY (rounds < current) ===")
    if ctx.visible_compounds:
        for c in ctx.visible_compounds:
            lines.append("  - " + _serialize_compound(c, include_outcome=True))
    else:
        lines.append("  (none — this is the first post-HTS decision)")
    lines.append("")

    if use_memory:
        lines.append("=== METACOGNITIVE MEMORY: CAMPAIGN STATE DIGEST ===")
        lines.append(_memory_digest(ctx))
        lines.append("")
        if ctx.visible_rationales:
            lines.append("=== PRIOR DECISION RATIONALES ===")
            for r in ctx.visible_rationales:
                lines.append("  - " + _serialize_rationale(r))
            lines.append("")
        lines.append(
            "Guidance: prefer the FIRST direction whose status is STILL SHORT. "
            "If multiple, prefer the one prior DPs have not yet addressed, "
            "or the one the trajectory shows is the current bottleneck."
        )
        lines.append("")

    lines.append("=== CANDIDATE POOL FOR NEXT ROUND (outcomes hidden) ===")
    for c in ctx.candidate_pool:
        lines.append("  - " + _serialize_compound(c, include_outcome=False))
    lines.append("")

    lines.append("Return strict JSON per the schema. No prose outside the JSON.")
    return "\n".join(lines)


def _extract_json(text: str) -> dict:
    # Try direct parse first.
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    # Strip markdown fences.
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass
    # Fallback: first balanced object.
    brace = re.search(r"\{.*\}", stripped, re.DOTALL)
    if brace:
        return json.loads(brace.group(0))
    raise ValueError(f"No parseable JSON object in LLM response: {text!r}")


def _repair_direction_ranking(raw: list, ctx: LeadOpContext) -> list[Direction]:
    seen: list[str] = []
    for d in raw:
        if isinstance(d, str) and d in ALL_DIRECTIONS and d not in seen:
            seen.append(d)
    for d in ALL_DIRECTIONS:
        if d not in seen:
            seen.append(d)
    return seen[:4]  # type: ignore[return-value]


def _repair_compound_ranking(raw: list, ctx: LeadOpContext) -> list[str]:
    valid = {c.compound_id for c in ctx.candidate_pool}
    seen: list[str] = []
    for cid in raw:
        if isinstance(cid, str) and cid in valid and cid not in seen:
            seen.append(cid)
    # Append any missing compounds in deterministic order.
    for c in ctx.candidate_pool:
        if c.compound_id not in seen:
            seen.append(c.compound_id)
    return seen


@dataclass
class LLMCallCost:
    input_tokens: int
    output_tokens: int
    cost_usd: float


def propose_llm(
    ctx: LeadOpContext,
    *,
    use_memory: bool,
    llm: LLMClient,
    rag_corpus: str | None = None,
    seed: int = 0,
    max_tokens: int = 1024,
) -> tuple[LeadOpDecision, LLMCallCost]:
    user = build_user_prompt(ctx, use_memory=use_memory, rag_corpus=rag_corpus)
    resp = llm.complete(
        system=SYSTEM_PROMPT, user=user, max_tokens=max_tokens, seed=seed
    )
    obj = _extract_json(resp.text)
    direction_ranking = _repair_direction_ranking(
        obj.get("direction_ranking", []), ctx
    )
    compound_ranking = _repair_compound_ranking(obj.get("compound_ranking", []), ctx)
    rationale = str(obj.get("rationale", "")).strip() or "(no rationale)"
    decision = LeadOpDecision(
        decision_point_id=ctx.decision_point_id,
        direction_ranking=direction_ranking,
        compound_ranking=compound_ranking,
        rationale=rationale,
    )
    cost = LLMCallCost(
        input_tokens=resp.input_tokens,
        output_tokens=resp.output_tokens,
        cost_usd=resp.cost_usd,
    )
    return decision, cost
