"""Retrieval Tier 2 — dense embedding cosine match.

Given a list of candidate rules (from Tier 1 or unfiltered) and a
case, rank rules by cosine similarity between:

- the case's structured representation embedding (per `_case_repr`), and
- each rule's cached `trigger.semantic_embedding`.

Rules that lack a cached embedding get embedded on demand. The caller
is responsible for persisting the embedding back to the RuleStore
(this module is pure — it does not write).

The `_case_repr()` function builds a deterministic case view that
mirrors what the proposer would see for a matching case, so that rule
triggers extracted from one case's semantic_predicate line up
naturally with the representation of a similar case.

Embedding model pinning: every rule carries `trigger.embedding_model`
alongside its vector. If the embedder's model_id doesn't match, the
rule's cached vector is ignored and a re-embed is forced — an
embedder model bump must not silently return stale cosine scores.
"""

from __future__ import annotations

from typing import Iterable, Optional

from medreason.ontology import BenchmarkCase, ReasoningRule

from .embedder import Embedder, cosine_similarity


def _case_repr(case: BenchmarkCase, max_notes_chars: int = 1500) -> str:
    """Structured case representation for embedding.

    Fields are tagged (e.g. `[PAYER]`) so the embedder sees a
    consistent schema across cases. Clinical notes are truncated to
    keep the token budget predictable — embeddings concentrate on the
    structural prefix anyway.
    """
    tc = case.task_config
    parts = [
        f"[PAYER] {tc.payer.value}",
        f"[CPT] {tc.cpt_code}",
        f"[ICD] {', '.join(tc.icd10_codes) if tc.icd10_codes else 'none'}",
        f"[FACILITY] {tc.facility_type.value}",
    ]
    if tc.modifiers:
        parts.append(f"[MODIFIERS] {', '.join(tc.modifiers)}")
    notes = (case.clinical_notes or "").strip()[:max_notes_chars]
    if notes:
        parts.append(f"[CLINICAL] {notes}")
    return "\n".join(parts)


def _rule_repr(rule: ReasoningRule) -> str:
    """Deterministic rule representation used when we need to embed a
    rule on demand. Mirrors the structural fields a case view would
    carry, plus the semantic predicate — so case embeddings and rule
    embeddings sit in a comparable subspace.
    """
    trig = rule.trigger
    parts: list[str] = []
    if trig.payers:
        parts.append("[PAYER] " + ", ".join(p.value for p in trig.payers))
    if trig.cpt_codes:
        parts.append("[CPT] " + ", ".join(trig.cpt_codes))
    if trig.cpt_families:
        parts.append(
            "[CPT_FAMILY] "
            + ", ".join(f.name.lower() for f in trig.cpt_families)
        )
    if trig.icd10_chapters:
        parts.append(
            "[ICD_CHAPTER] "
            + ", ".join(c.name.lower() for c in trig.icd10_chapters)
        )
    if trig.icd10_prefixes:
        parts.append("[ICD_PREFIX] " + ", ".join(trig.icd10_prefixes))
    if trig.facility_types:
        parts.append(
            "[FACILITY] "
            + ", ".join(f.name.lower() for f in trig.facility_types)
        )
    if trig.semantic_predicate:
        parts.append(f"[SEMANTIC] {trig.semantic_predicate}")
    parts.append(f"[ACTION] {rule.action}")
    if rule.rationale:
        parts.append(f"[RATIONALE] {rule.rationale}")
    return "\n".join(parts)


def _resolve_rule_embedding(
    rule: ReasoningRule,
    embedder: Embedder,
) -> list[float]:
    """Return the rule's embedding, computing it if missing or stale.

    "Stale" = `trigger.embedding_model` doesn't match `embedder.model_id`.
    This module does NOT persist the computed embedding — the caller
    can decide whether to write it back to the store (e.g. during a
    warm-up pass) or treat it as ephemeral (e.g. during an ablation).
    """
    cached = rule.trigger.semantic_embedding
    cached_model = rule.trigger.embedding_model
    if (
        cached
        and len(cached) == embedder.dimension
        and cached_model == embedder.model_id
    ):
        return cached
    text = _rule_repr(rule)
    return embedder.embed(text)


def dense_rerank(
    case: BenchmarkCase,
    candidates: Iterable[ReasoningRule],
    embedder: Embedder,
    *,
    top_k: int = 25,
    min_similarity: float = 0.0,
) -> list[tuple[ReasoningRule, float]]:
    """Rank candidate rules by cosine similarity against the case.

    Args:
        case: The case being resolved.
        candidates: Rules to rank (typically Tier 1's output).
        embedder: The Embedder used to vectorize the case and any rules
            that lack a cached embedding. Must match the model id that
            cached rule embeddings were written with — stale cache
            entries are silently bypassed.
        top_k: Maximum results returned.
        min_similarity: Similarity floor. Results below this are dropped.
            0.0 is the safe default; set higher when you trust the
            embedder's absolute scale.

    Returns:
        A list of (rule, cosine_similarity) pairs sorted by similarity
        desc. Tied similarities broken by posterior_mean desc.
    """
    case_vec = embedder.embed(_case_repr(case))
    scored: list[tuple[ReasoningRule, float]] = []
    for rule in candidates:
        rule_vec = _resolve_rule_embedding(rule, embedder)
        sim = cosine_similarity(case_vec, rule_vec)
        if sim < min_similarity:
            continue
        scored.append((rule, sim))
    scored.sort(key=lambda pair: (-pair[1], -pair[0].posterior_mean, pair[0].rule_id))
    return scored[:top_k]
