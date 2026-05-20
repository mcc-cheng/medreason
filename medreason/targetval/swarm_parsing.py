"""Strict-JSON parser for SwarmAgent LLM outputs.

Contract:

- ``extract_memo_json(text)`` extracts the first JSON object from an LLM
  response. Strategy mirrors ``medreason_bench/leadop/agent_llm.py:
  _extract_json``:

      1. ``json.loads(text.strip())`` — pure JSON happy path.
      2. Fenced triple-backtick ``json`` block regex.
      3. First balanced ``{ … }`` block regex.
      4. Otherwise raise ``MemoParseError``.

  Never uses ``eval``. Never uses any LLM. Pure-Python, deterministic.

- ``parse_memo(text, ...)`` builds a ``TargetMemo`` from the raw text.
  It is **tolerant**: missing fields fall back to schema defaults,
  out-of-range scores clamp into ``[0, 1]`` with a warning, unknown
  bypass-mechanism strings collapse to ``BypassMechanism.UNKNOWN`` with
  a warning. The only fatal failure is "no JSON object found at all",
  which raises ``MemoParseError``. Callers that want a non-raising
  fallback should call ``safe_parse_memo`` instead.

Why tolerant rather than strict?
- Phase 2 swarm prompts are not leaderboard-scored — the priority is
  forward progress, not strict-schema enforcement.
- The downstream cross-agent analyzer (``proposer-builder``) reads
  ``parse_warnings`` to spot systematic prompt-following failures
  across the swarm. Surfacing those at memo time, instead of
  exception time, makes the analyzer's job easier.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .case import BypassMechanism
from .topology import BypassSignal


# ── Error type ──────────────────────────────────────────────────────────────


class MemoParseError(ValueError):
    """No JSON object could be extracted from the LLM response."""


# ── JSON extraction ─────────────────────────────────────────────────────────


_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_BALANCED_BRACE_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_memo_json(text: str) -> dict:
    """Extract the first JSON *object* (dict) from an LLM response.

    See module docstring for the four-step strategy. Raises
    ``MemoParseError`` if no JSON object can be recovered.
    """
    if text is None:
        raise MemoParseError("LLM response text is None")
    stripped = text.strip()
    if not stripped:
        raise MemoParseError("LLM response text is empty")

    # 1. Try a direct parse.
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj
        raise MemoParseError(
            f"top-level JSON is {type(obj).__name__}, expected object"
        )
    except json.JSONDecodeError:
        pass

    # 2. Try fenced ```json``` block.
    fence = _FENCED_JSON_RE.search(stripped)
    if fence:
        try:
            obj = json.loads(fence.group(1))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    # 3. Try first balanced { ... } block. The greedy ".*" with DOTALL
    # picks up the OUTERMOST braces, which is what we want for nested
    # JSON. If that fails, fall through to the typed error.
    brace = _BALANCED_BRACE_RE.search(stripped)
    if brace:
        try:
            obj = json.loads(brace.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    # 4. Nothing worked.
    raise MemoParseError(
        f"No parseable JSON object in LLM response (first 200 chars): "
        f"{stripped[:200]!r}"
    )


# ── Field helpers ───────────────────────────────────────────────────────────


_VALID_BYPASS_VALUES: frozenset[str] = frozenset(b.value for b in BypassMechanism)


def _clamp_score(
    raw: Any,
    *,
    field_name: str,
    warnings: list[str],
) -> float:
    """Coerce ``raw`` to a float in ``[0, 1]``.

    - Non-numeric → 0.0 with a warning.
    - Out-of-range → clamp + warning.
    - Missing → 0.0, no warning (already covered by default).
    """
    if raw is None:
        return 0.0
    try:
        value = float(raw)
    except (TypeError, ValueError):
        warnings.append(
            f"{field_name}: non-numeric value {raw!r}, defaulting to 0.0"
        )
        return 0.0
    if value < 0.0:
        warnings.append(
            f"{field_name}: {value} below 0, clamped to 0.0"
        )
        return 0.0
    if value > 1.0:
        warnings.append(
            f"{field_name}: {value} above 1, clamped to 1.0"
        )
        return 1.0
    return value


def _parse_bypass(
    raw: Any,
    *,
    warnings: list[str],
) -> BypassMechanism:
    """Parse a ``predicted_bypass`` value into a ``BypassMechanism``.

    Accepts the ``.value`` strings (e.g., ``"paralog_compensation"``)
    and also the enum *member* names (e.g., ``"PARALOG_COMPENSATION"``).
    Unknown strings collapse to ``UNKNOWN`` and emit a warning.
    """
    if raw is None:
        return BypassMechanism.UNKNOWN
    if not isinstance(raw, str):
        warnings.append(
            f"predicted_bypass: non-string {raw!r}, set to UNKNOWN"
        )
        return BypassMechanism.UNKNOWN
    candidate = raw.strip()
    if not candidate:
        return BypassMechanism.UNKNOWN
    if candidate in _VALID_BYPASS_VALUES:
        return BypassMechanism(candidate)
    # Try the upper-case member name as a fallback (e.g., "ALTERNATIVE_PATHWAY")
    try:
        return BypassMechanism[candidate.upper()]
    except KeyError:
        warnings.append(
            f"predicted_bypass: unknown mechanism {raw!r}, set to UNKNOWN"
        )
        return BypassMechanism.UNKNOWN


def _str_list(
    raw: Any,
    *,
    field_name: str,
    warnings: list[str],
) -> list[str]:
    """Coerce ``raw`` to a list of strings.

    - ``None`` / missing → ``[]``.
    - A bare string → ``[string]`` with a warning (tolerant, not silent).
    - Non-string entries are skipped with a per-entry warning.
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        warnings.append(
            f"{field_name}: expected list, got string; wrapping in a list"
        )
        return [raw]
    if not isinstance(raw, list):
        warnings.append(
            f"{field_name}: expected list, got {type(raw).__name__}; "
            "defaulting to []"
        )
        return []
    out: list[str] = []
    for i, entry in enumerate(raw):
        if isinstance(entry, str):
            out.append(entry)
        else:
            warnings.append(
                f"{field_name}[{i}]: expected string, got "
                f"{type(entry).__name__}; skipped"
            )
    return out


# ── Public parser ───────────────────────────────────────────────────────────


# Forward reference avoidance: parse_memo returns a TargetMemo defined in
# swarm.py, but swarm.py imports from this module. We import lazily
# inside the function to break the cycle. The dataclass below mirrors the
# return shape so callers and tests can introspect what fields are set.


@dataclass
class _ParsedMemoPayload:
    """Intermediate parse result. Internal — callers see TargetMemo."""

    priority_score: float = 0.0
    bypass_risk_score: float = 0.0
    predicted_bypass: BypassMechanism = BypassMechanism.UNKNOWN
    supporting_evidence: list[str] = field(default_factory=list)
    weakening_evidence: list[str] = field(default_factory=list)
    proposed_experiments: list[str] = field(default_factory=list)
    rationale: str = ""
    applied_rule_ids: list[str] = field(default_factory=list)
    parse_warnings: list[str] = field(default_factory=list)


def _parse_payload(data: dict) -> _ParsedMemoPayload:
    warnings: list[str] = []
    priority = _clamp_score(
        data.get("priority_score"), field_name="priority_score", warnings=warnings
    )
    bypass_risk = _clamp_score(
        data.get("bypass_risk_score"),
        field_name="bypass_risk_score",
        warnings=warnings,
    )
    predicted_bypass = _parse_bypass(
        data.get("predicted_bypass"), warnings=warnings
    )
    supporting = _str_list(
        data.get("supporting_evidence"),
        field_name="supporting_evidence",
        warnings=warnings,
    )
    weakening = _str_list(
        data.get("weakening_evidence"),
        field_name="weakening_evidence",
        warnings=warnings,
    )
    experiments = _str_list(
        data.get("proposed_experiments"),
        field_name="proposed_experiments",
        warnings=warnings,
    )
    rationale_raw = data.get("rationale", "")
    if isinstance(rationale_raw, str):
        rationale = rationale_raw
    else:
        warnings.append(
            f"rationale: expected string, got {type(rationale_raw).__name__}; "
            "defaulting to ''"
        )
        rationale = ""
    applied = _str_list(
        data.get("applied_rule_ids"),
        field_name="applied_rule_ids",
        warnings=warnings,
    )
    return _ParsedMemoPayload(
        priority_score=priority,
        bypass_risk_score=bypass_risk,
        predicted_bypass=predicted_bypass,
        supporting_evidence=supporting,
        weakening_evidence=weakening,
        proposed_experiments=experiments,
        rationale=rationale,
        applied_rule_ids=applied,
        parse_warnings=warnings,
    )


def parse_memo(
    text: str,
    *,
    case_id: str,
    gene_symbol: str,
    retrieved_rule_ids: list[str],
    bypass_signals: list[BypassSignal],
    cost_usd: float,
    seed: int,
):
    """Build a ``TargetMemo`` from raw LLM text.

    Raises ``MemoParseError`` ONLY if the JSON itself can't be extracted
    at all (e.g., the LLM returned a paragraph of prose with no braces).
    All other validation issues (unknown enum value, score out of range,
    wrong type) are recorded as ``parse_warnings`` on the returned memo
    and the offending field falls back to a sane default.
    """
    # Local import breaks the swarm.py ↔ swarm_parsing.py cycle.
    from .swarm import TargetMemo

    data = extract_memo_json(text)
    payload = _parse_payload(data)

    # bypass_signals_seen: copy the mechanism strings from the signals
    # the agent was shown. proposer-builder reads this field to assert
    # "the swarm SAW signal X and still mis-scored bypass_risk".
    bypass_signals_seen = [s.mechanism for s in bypass_signals]

    return TargetMemo(
        case_id=case_id,
        gene_symbol=gene_symbol,
        priority_score=payload.priority_score,
        bypass_risk_score=payload.bypass_risk_score,
        predicted_bypass=payload.predicted_bypass,
        supporting_evidence=payload.supporting_evidence,
        weakening_evidence=payload.weakening_evidence,
        proposed_experiments=payload.proposed_experiments,
        rationale=payload.rationale,
        retrieved_rule_ids=list(retrieved_rule_ids),
        applied_rule_ids=payload.applied_rule_ids,
        bypass_signals_seen=bypass_signals_seen,
        parse_warnings=payload.parse_warnings,
        cost_usd=cost_usd,
        seed=seed,
    )


def safe_parse_memo(
    text: str,
    *,
    case_id: str,
    gene_symbol: str,
    retrieved_rule_ids: list[str],
    bypass_signals: list[BypassSignal],
    cost_usd: float,
    seed: int,
):
    """Like ``parse_memo`` but never raises.

    On total parse failure (no JSON object found at all), returns a
    fallback ``TargetMemo`` with scores at ``0.0`` and a single
    ``parse_warnings`` entry describing the failure. Used by
    ``SwarmAgent.run`` so a single misbehaving LLM call doesn't kill
    the whole swarm.
    """
    from .swarm import TargetMemo

    try:
        return parse_memo(
            text,
            case_id=case_id,
            gene_symbol=gene_symbol,
            retrieved_rule_ids=retrieved_rule_ids,
            bypass_signals=bypass_signals,
            cost_usd=cost_usd,
            seed=seed,
        )
    except MemoParseError as e:
        return TargetMemo(
            case_id=case_id,
            gene_symbol=gene_symbol,
            priority_score=0.0,
            bypass_risk_score=0.0,
            predicted_bypass=BypassMechanism.UNKNOWN,
            rationale="(LLM output unparseable — fallback memo)",
            retrieved_rule_ids=list(retrieved_rule_ids),
            applied_rule_ids=[],
            bypass_signals_seen=[s.mechanism for s in bypass_signals],
            parse_warnings=[f"memo_parse_error: {e}"],
            cost_usd=cost_usd,
            seed=seed,
        )
