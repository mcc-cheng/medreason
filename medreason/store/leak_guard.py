"""Leak guard — prevents test-set contamination of the memory store.

Three protections, enforced at every write that reaches the store:

1. **Write-side case_id check**: if a rule's evidence.supporting_case_ids
   (or a trace's case_id, once we add one) references any case_id in the
   frozen test fingerprint set, raise TestSetLeakError. The rule/trace is
   never written to disk.

2. **Embedding-side similarity check**: if a rule carries a
   trigger.semantic_embedding and the guard was constructed with test-case
   embeddings, the maximum cosine similarity against the test set must stay
   below `embedding_similarity_threshold`. Exceeding it raises
   LikelyTestLeakError — a rule that's semantically a paraphrase of a test
   case should not enter memory even if no case_id is declared.

3. **Read-side test-phase mode**: `enter_test_phase()` puts the guard into
   read-only mode. Any store.put() / store.update_posteriors() attempted
   while in test phase raises TestSetLeakError. This is the tripwire that
   the eval harness flips during scored runs — it catches bugs where the
   store is still being updated mid-eval.

The guard logs every violation to the store's `leak_violations` table via
the `on_violation` callback (injected by RuleStore/TraceStore at wiring
time). Raising an exception is primary; logging is secondary.

The fingerprints file schema is:

    {
      "train": {"train_001": "<sha256>", ...},
      "dev":   {"dev_001":   "<sha256>", ...},
      "test":  {"test_001":  "<sha256>", ...}
    }

Hashes aren't checked here (that's the manifest verifier's job). The guard
only cares about case_id membership.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, Iterable, Optional

from ..ontology.rule import ReasoningRule
from ..ontology.trace import ReasoningTrace


# ── Exceptions ────────────────────────────────────────────────────────────────


class TestSetLeakError(Exception):
    """Raised when a write to the memory store would contaminate the test set.

    Categories:
    - A rule/trace references a test case_id in its provenance.
    - A write is attempted while the store is in read-only test-phase mode.
    """


class LikelyTestLeakError(TestSetLeakError):
    """Raised when a rule's semantic embedding is too similar to a test case.

    This is a soft match: no explicit case_id link, but the embedding cosine
    exceeds the guard's threshold. Treated as fatal because the rule is
    likely a near-paraphrase of test content.
    """


# ── Guard ─────────────────────────────────────────────────────────────────────


OnViolation = Callable[[str, dict], None]
"""Callback signature: (violation_type, details) -> None.
Stores should log violations via this hook before the exception propagates.
"""


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity. Returns 0.0 if either vector has zero norm."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


class LeakGuard:
    """Stateful guard object that every store write must consult.

    A LeakGuard with no test_case_ids and no test_case_embeddings is a
    pass-through: `check_rule_against_test_set` never raises. This is the
    safe default for unit tests and local development.
    """

    def __init__(
        self,
        test_case_ids: Optional[Iterable[str]] = None,
        train_case_ids: Optional[Iterable[str]] = None,
        dev_case_ids: Optional[Iterable[str]] = None,
        test_case_embeddings: Optional[dict[str, list[float]]] = None,
        embedding_similarity_threshold: float = 0.92,
        on_violation: Optional[OnViolation] = None,
    ):
        self.test_case_ids: frozenset[str] = frozenset(test_case_ids or ())
        self.train_case_ids: frozenset[str] = frozenset(train_case_ids or ())
        self.dev_case_ids: frozenset[str] = frozenset(dev_case_ids or ())
        self.test_case_embeddings: dict[str, list[float]] = dict(test_case_embeddings or {})
        self.embedding_similarity_threshold: float = float(embedding_similarity_threshold)
        self._read_only: bool = False
        self._on_violation: Optional[OnViolation] = on_violation

    # ── Fingerprint file loader ──────────────────────────────────────────────

    @classmethod
    def from_fingerprint_file(
        cls,
        path: Path | str,
        test_case_embeddings: Optional[dict[str, list[float]]] = None,
        embedding_similarity_threshold: float = 0.92,
        on_violation: Optional[OnViolation] = None,
    ) -> "LeakGuard":
        """Load case_id sets from a manifest fingerprints.json file.

        The file must be a mapping of split name → {case_id: sha256}. Only
        the case_id keys are read; hashes are ignored (hash verification is
        the manifest verifier's responsibility).
        """
        p = Path(path)
        data = json.loads(p.read_text())
        return cls(
            test_case_ids=(data.get("test") or {}).keys(),
            train_case_ids=(data.get("train") or {}).keys(),
            dev_case_ids=(data.get("dev") or {}).keys(),
            test_case_embeddings=test_case_embeddings,
            embedding_similarity_threshold=embedding_similarity_threshold,
            on_violation=on_violation,
        )

    # ── Violation logging hook ───────────────────────────────────────────────

    def set_violation_logger(self, hook: OnViolation) -> None:
        """Wire up the store's violation logger after construction."""
        self._on_violation = hook

    def _log(self, violation_type: str, details: dict) -> None:
        if self._on_violation is not None:
            try:
                self._on_violation(violation_type, details)
            except Exception:
                # Logging must never mask the primary exception.
                pass

    # ── Read-only (test phase) mode ──────────────────────────────────────────

    def enter_test_phase(self) -> None:
        """Put the guard into read-only mode. Used by the eval harness
        during scored test runs — any further store.put() raises."""
        self._read_only = True

    def exit_test_phase(self) -> None:
        self._read_only = False

    @property
    def read_only(self) -> bool:
        return self._read_only

    def check_write_allowed(self, operation: str = "write") -> None:
        """Raise if the guard is in read-only test-phase mode."""
        if self._read_only:
            details = {"operation": operation}
            self._log("read_only", details)
            raise TestSetLeakError(
                f"LeakGuard is in read-only test-phase mode; {operation} blocked"
            )

    # ── Per-entity checks ────────────────────────────────────────────────────

    def check_rule_against_test_set(self, rule: ReasoningRule) -> None:
        """Write-side protection for a ReasoningRule.

        Raises:
            TestSetLeakError: if the rule's evidence references any test
                case_id.
            LikelyTestLeakError: if the rule's semantic_embedding has
                cosine > threshold against any test case embedding.
        """
        # 1. Case-id protection
        offenders = set(rule.evidence.supporting_case_ids) & self.test_case_ids
        if offenders:
            offenders_list = sorted(offenders)
            details = {
                "rule_id": rule.rule_id,
                "offending_case_ids": offenders_list,
            }
            self._log("case_id", details)
            raise TestSetLeakError(
                f"Rule {rule.rule_id} references test case_id(s): {offenders_list}"
            )

        # 2. Embedding-similarity protection
        emb = rule.trigger.semantic_embedding
        if emb and self.test_case_embeddings:
            worst_case_id = None
            worst_sim = 0.0
            for cid, tvec in self.test_case_embeddings.items():
                sim = _cosine(emb, tvec)
                if sim > worst_sim:
                    worst_sim = sim
                    worst_case_id = cid
            if worst_sim > self.embedding_similarity_threshold:
                details = {
                    "rule_id": rule.rule_id,
                    "max_cosine": round(worst_sim, 6),
                    "threshold": self.embedding_similarity_threshold,
                    "nearest_test_case_id": worst_case_id,
                }
                self._log("embedding", details)
                raise LikelyTestLeakError(
                    f"Rule {rule.rule_id} semantic embedding matches test case "
                    f"{worst_case_id!r} with cosine {worst_sim:.3f} "
                    f"(threshold {self.embedding_similarity_threshold})"
                )

    def check_trace_against_test_set(self, trace: ReasoningTrace) -> None:
        """Write-side protection for a ReasoningTrace.

        Traces don't currently carry case_id provenance on the model, but
        when the critic pipeline lands in Phase 5a we'll add that field
        and this method will check it. For now this is a stub that only
        enforces the read-only test-phase mode via the caller.
        """
        # Reserved: future case_id check when ReasoningTrace grows a
        # `source_case_id` field in Phase 5a. Intentionally no-op for now.
        _ = trace

    # ── Convenience ──────────────────────────────────────────────────────────

    def is_test_case_id(self, case_id: str) -> bool:
        return case_id in self.test_case_ids

    def __repr__(self) -> str:
        return (
            f"LeakGuard(test={len(self.test_case_ids)}, "
            f"train={len(self.train_case_ids)}, dev={len(self.dev_case_ids)}, "
            f"test_emb={len(self.test_case_embeddings)}, "
            f"read_only={self._read_only})"
        )
