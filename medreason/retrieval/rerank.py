"""Retrieval Tier 3 — LLM reranker.

Given a list of candidate rules (typically the top 25 from Tier 2), a
single LLM call scores each one on applicability + expected utility.
The final retrieval order combines these with the rule's posterior
mean:

    final_score = (applicability / 10) * (expected_utility / 10) * posterior_mean

Rules with `applicability < min_applicability` are dropped entirely.
This is the Phase 5 primary filter for "relevant but off-topic" rules
that survived the Tier 1 ontology filter and the Tier 2 cosine match.

Failure modes and recovery:

- Reranker returns malformed JSON → fall back to Tier 2 ordering with
  posterior weighting, drop no rules. The pipeline still produces a
  reasonable ordering; we just lose the applicability filter.
- Reranker excludes every candidate → return empty list. This is
  CORRECT behavior: it says "no rule in memory is useful for this
  case" and the memory wrapper will fall back to zero-shot. That is
  NOT a bug.
- Reranker emits fewer entries than the candidates → score missing
  rules with applicability=0 so they get dropped below the threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..llm.base import LLMClient
from ..ontology import BenchmarkCase, ReasoningRule
from ..prompts import load_prompt
from ..runners._prompting import ResponseParseError, parse_json_response


RERANK_PROMPT_FILE = "rerank.txt"


@dataclass
class RerankScore:
    rule_id: str
    applicability: float  # 0..10
    expected_utility: float  # 0..10
    conflict_with: list[str]

    @property
    def raw_score(self) -> float:
        """(applicability/10) * (expected_utility/10), in [0, 1]."""
        return (
            max(0.0, min(10.0, self.applicability))
            * max(0.0, min(10.0, self.expected_utility))
            / 100.0
        )


def _format_rerank_user(
    case: BenchmarkCase,
    candidates: list[ReasoningRule],
    max_notes_chars: int = 1500,
) -> str:
    """Build the user message. Mirrors the field set the reranker prompt
    describes so the model has every signal it needs to score accurately.
    """
    tc = case.task_config
    parts: list[str] = [
        "## Case",
        f"Payer: {tc.payer.value}",
        f"CPT: {tc.cpt_code}",
        f"ICD-10: {', '.join(tc.icd10_codes) if tc.icd10_codes else 'none'}",
        f"Facility: {tc.facility_type.value}",
    ]
    notes = (case.clinical_notes or "").strip()[:max_notes_chars]
    if notes:
        parts.append("")
        parts.append("Clinical notes:")
        parts.append(notes)

    parts.append("")
    parts.append("## Candidate rules")
    for i, rule in enumerate(candidates, start=1):
        trig = rule.trigger
        trigger_parts: list[str] = []
        if trig.payers:
            trigger_parts.append("payer=" + ",".join(p.value for p in trig.payers))
        if trig.cpt_families:
            trigger_parts.append(
                "cpt_family="
                + ",".join(f.name.lower() for f in trig.cpt_families)
            )
        if trig.cpt_codes:
            trigger_parts.append("cpt=" + ",".join(trig.cpt_codes))
        if trig.icd10_chapters:
            trigger_parts.append(
                "icd_chapter="
                + ",".join(c.name.lower() for c in trig.icd10_chapters)
            )
        trig_summary = " | ".join(trigger_parts) or "any"

        parts.append(
            f"{i}. [rule_id={rule.rule_id}] (posterior={rule.posterior_mean:.2f}, "
            f"n={rule.trials})"
        )
        parts.append(f"   TRIGGER: {trig_summary}")
        parts.append(f"   ACTION:  {rule.action}")
        parts.append(f"   POLARITY: {rule.polarity}")
        if trig.semantic_predicate:
            parts.append(f"   APPLIES WHEN: {trig.semantic_predicate}")

    return "\n".join(parts)


def _parse_rerank_response(raw: str) -> list[RerankScore]:
    """Parse the LLM output into a list of RerankScore objects.

    Returns an empty list on any parse failure so the pipeline can
    fall back to Tier 2 ordering.
    """
    try:
        data = parse_json_response(raw)
    except ResponseParseError:
        return []

    raw_list = data.get("scores")
    if not isinstance(raw_list, list):
        return []

    out: list[RerankScore] = []
    for entry in raw_list:
        if not isinstance(entry, dict):
            continue
        rid = entry.get("rule_id")
        if not isinstance(rid, str) or not rid:
            continue
        try:
            app = float(entry.get("applicability", 0))
            util = float(entry.get("expected_utility", 0))
        except (TypeError, ValueError):
            continue
        conflict_raw = entry.get("conflict_with", [])
        conflicts = (
            [str(c) for c in conflict_raw if isinstance(c, str)]
            if isinstance(conflict_raw, list)
            else []
        )
        out.append(RerankScore(
            rule_id=rid,
            applicability=app,
            expected_utility=util,
            conflict_with=conflicts,
        ))
    return out


def rerank(
    case: BenchmarkCase,
    candidates: list[ReasoningRule],
    llm_client: LLMClient,
    *,
    top_k: int = 6,
    min_applicability: float = 4.0,
    max_tokens: int = 2048,
    seed: int = 0,
) -> list[ReasoningRule]:
    """Rerank candidates via an LLM and return the top `top_k`.

    Final score = raw_score (applicability × expected_utility / 100)
                  × posterior_mean

    Rules with applicability below `min_applicability` are filtered
    entirely (returning empty = "no useful memory" is valid).

    On any LLM or parse error, falls back to Tier 2 ordering (the
    `candidates` list is assumed to arrive in Tier 2 order) weighted
    by posterior_mean.
    """
    if not candidates:
        return []

    system = load_prompt(RERANK_PROMPT_FILE)
    user = _format_rerank_user(case, candidates)

    try:
        response = llm_client.complete(
            system=system, user=user, max_tokens=max_tokens, seed=seed
        )
    except Exception:
        return _fallback(candidates, top_k)

    scores = _parse_rerank_response(response.text or "")
    if not scores:
        return _fallback(candidates, top_k)

    scores_by_id: dict[str, RerankScore] = {s.rule_id: s for s in scores}

    # Apply the applicability filter and compute final scores. Missing
    # scores (reranker skipped a rule) default to applicability=0 and
    # are therefore dropped.
    kept: list[tuple[ReasoningRule, float]] = []
    for rule in candidates:
        sc = scores_by_id.get(rule.rule_id)
        if sc is None or sc.applicability < min_applicability:
            continue
        final = sc.raw_score * rule.posterior_mean
        kept.append((rule, final))

    kept.sort(key=lambda pair: (-pair[1], -pair[0].posterior_mean, pair[0].rule_id))
    return [r for r, _ in kept[:top_k]]


def _fallback(
    candidates: list[ReasoningRule],
    top_k: int,
) -> list[ReasoningRule]:
    """Reranker-free ordering by posterior_mean. Used on LLM/parse
    failure. Does NOT apply the applicability filter because we have
    no applicability signal — Tier 2 already pre-selected these."""
    ordered = sorted(
        candidates,
        key=lambda r: (-r.posterior_mean, -r.trials, r.rule_id),
    )
    return ordered[:top_k]
