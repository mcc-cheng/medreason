"""Train the memory pipeline by walking a training split.

For each training case:
  1. Run the base agent zero-shot. Discard wrong answers.
  2. Run the critic on the case inputs alone. Discard disagreement.
  3. Persist the critic-verified trace.
  4. Run the rule proposer against the verified trace + policy excerpt.
  5. For each candidate rule, run the GeneralizationGate against OTHER
     training cases (NOT dev, NOT test — holdout hygiene for the eval
     that follows).
  6. Promoted rules land in the RuleStore with status=ACTIVE. Candidates
     and deprecated stay off the eval path.

Design notes:

- Held-out pool is "other train cases". Using dev would leak the eval
  target into the gate. Using test is flat-out forbidden (LeakGuard
  would reject any rule trying to reference a test case_id anyway).
- The base runner used for the gate is the SAME model as the training
  runner. This is a first-pass test shortcut; plan risk #1 calls out
  that the critic should ideally be a different vendor. For the cheap
  Haiku experiment we accept the correlation.
- The embedder is FakeEmbedder by default so the train loop works
  without an OpenAI key. Retrieval doesn't run during training — the
  embedder is only needed so we can compute and cache each promoted
  rule's semantic_embedding for downstream eval.
- The policy is loaded from the bundled fixture unless --lcd is
  passed. Phase 3 shipped the fixture; Phase 6 ingestion will stream
  real LCDs once network is available.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from medreason.extraction import (
    DEFAULT_THRESHOLDS as GATE_DEFAULTS,
    GateResult,
    GeneralizationGate,
    propose_rules,
    run_critic,
)
from medreason.llm.base import LLMClient
from medreason.ontology import (
    AgentResult,
    BenchmarkCase,
    ReasoningRule,
    RuleStatus,
)
from medreason.retrieval.dense import _rule_repr  # internal — intentional
from medreason.retrieval.embedder import Embedder, FakeEmbedder
from medreason.runners.base import AgentRunner
from medreason.store import RuleStore, TraceStore

from .data import parse_lcd_xml
from .data.schemas import LCDPolicy
from .splits import load_split


# ── Report dataclass ────────────────────────────────────────────────────────


@dataclass
class TrainingReport:
    version: str
    split: str
    runner_id: str
    started_at: datetime
    finished_at: Optional[datetime] = None

    # Per-stage counts
    n_cases_seen: int = 0
    n_agent_correct: int = 0
    n_critic_invoked: int = 0
    n_critic_agreed: int = 0
    n_traces_stored: int = 0
    n_proposer_invoked: int = 0
    n_rules_proposed: int = 0
    n_rules_rejected: int = 0
    n_rules_gated: int = 0
    n_rules_promoted: int = 0
    n_rules_deprecated: int = 0
    n_rules_deferred: int = 0

    # Accumulated cost (USD)
    cost_agent: float = 0.0
    cost_critic: float = 0.0
    cost_proposer: float = 0.0
    cost_gate: float = 0.0

    # Miscellaneous diagnostics
    rejection_reasons: list[str] = field(default_factory=list)
    gate_results: list[dict] = field(default_factory=list)

    @property
    def cost_total(self) -> float:
        return self.cost_agent + self.cost_critic + self.cost_proposer + self.cost_gate

    @property
    def elapsed_seconds(self) -> float:
        if self.finished_at is None:
            return 0.0
        return (self.finished_at - self.started_at).total_seconds()

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "split": self.split,
            "runner_id": self.runner_id,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "elapsed_seconds": self.elapsed_seconds,
            "n_cases_seen": self.n_cases_seen,
            "n_agent_correct": self.n_agent_correct,
            "n_critic_invoked": self.n_critic_invoked,
            "n_critic_agreed": self.n_critic_agreed,
            "n_traces_stored": self.n_traces_stored,
            "n_proposer_invoked": self.n_proposer_invoked,
            "n_rules_proposed": self.n_rules_proposed,
            "n_rules_rejected": self.n_rules_rejected,
            "n_rules_gated": self.n_rules_gated,
            "n_rules_promoted": self.n_rules_promoted,
            "n_rules_deprecated": self.n_rules_deprecated,
            "n_rules_deferred": self.n_rules_deferred,
            "cost_agent": round(self.cost_agent, 5),
            "cost_critic": round(self.cost_critic, 5),
            "cost_proposer": round(self.cost_proposer, 5),
            "cost_gate": round(self.cost_gate, 5),
            "cost_total": round(self.cost_total, 5),
            "rejection_reasons": self.rejection_reasons[:20],
        }


# ── Config ──────────────────────────────────────────────────────────────────


@dataclass
class TrainingConfig:
    runner: AgentRunner
    critic_llm: LLMClient
    proposer_llm: LLMClient
    store: RuleStore
    trace_store: TraceStore
    policy: LCDPolicy
    train_cases: list[BenchmarkCase]
    version: str = "v0.0"
    split: str = "train"

    # Gate config
    gate_k: int = 3
    gate_seed: int = 0

    # Cache embeddings on promoted rules so retrieval doesn't have to
    # re-embed every inference. FakeEmbedder is fine for the cheap run.
    embedder: Optional[Embedder] = None

    # Progress hook: callable(message: str) -> None
    progress_hook: Optional[Any] = None

    # Seed forwarded to every runner call
    seed: int = 0

    # Early stopping: cap the number of train cases processed (useful
    # for debugging or ultra-cheap runs). None = process all.
    max_cases: Optional[int] = None


# ── The training loop ──────────────────────────────────────────────────────


def _log(hook: Optional[Any], msg: str) -> None:
    if hook is not None:
        hook(msg)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def run_training(config: TrainingConfig) -> TrainingReport:
    """Walk `config.train_cases` and populate the stores.

    Returns a TrainingReport with per-stage counts and accumulated cost.
    The RuleStore and TraceStore passed in on the config are mutated
    in place. Caller owns persistence (commits, closes, etc.).
    """
    report = TrainingReport(
        version=config.version,
        split=config.split,
        runner_id=config.runner.runner_id,
        started_at=_utcnow(),
    )

    embedder = config.embedder or FakeEmbedder()
    cases = config.train_cases
    if config.max_cases is not None:
        cases = cases[: config.max_cases]

    _log(config.progress_hook,
         f"[train] {len(cases)} case(s), runner={config.runner.runner_id}, "
         f"gate_k={config.gate_k}")

    for idx, case in enumerate(cases):
        report.n_cases_seen += 1
        _log(config.progress_hook,
             f"[train] case {idx + 1}/{len(cases)} — {case.case_id} "
             f"({case.task_config.cpt_code}, {case.ground_truth_outcome.value})")

        # ── Stage 1: zero-shot agent ────────────────────────────────────
        agent_result = config.runner.run(case, seed=config.seed)
        report.cost_agent += agent_result.cost_usd
        if not agent_result.correct:
            _log(config.progress_hook, "  agent wrong — skipping")
            continue
        report.n_agent_correct += 1

        # ── Stage 2: critic verification ─────────────────────────────────
        report.n_critic_invoked += 1
        critic_result = run_critic(
            case,
            agent_determination=agent_result.determination,
            llm_client=config.critic_llm,
            seed=config.seed,
        )
        # We can't directly read critic cost from CriticResult — the
        # critic internally calls llm_client.complete which returns
        # LLMResponse.cost_usd, but run_critic discards it. For the
        # first cheap run we estimate via the runner's est cost.
        # TODO: thread cost_usd through CriticResult in a follow-up.
        report.cost_critic += config.runner.estimated_cost_per_call()

        if not critic_result.agrees:
            _log(config.progress_hook,
                 f"  critic {critic_result.reason} — skipping")
            continue
        report.n_critic_agreed += 1

        # Persist the verified trace
        assert critic_result.trace is not None
        config.trace_store.put(critic_result.trace)
        report.n_traces_stored += 1

        # ── Stage 3: rule proposer ───────────────────────────────────────
        report.n_proposer_invoked += 1
        proposal = propose_rules(
            critic_result.trace,
            config.policy,
            config.proposer_llm,
            supporting_case_ids=[case.case_id],
            seed=config.seed,
        )
        report.cost_proposer += config.runner.estimated_cost_per_call()
        report.n_rules_proposed += proposal.n_candidates
        report.n_rules_rejected += proposal.n_rejected
        if proposal.n_rejected:
            for _, reason in proposal.rejected[:3]:
                report.rejection_reasons.append(reason)

        if not proposal.candidates:
            _log(config.progress_hook, "  proposer returned no candidates")
            continue

        _log(config.progress_hook,
             f"  proposer: {proposal.n_candidates} candidate(s), "
             f"{proposal.n_rejected} rejected")

        # ── Stage 4: generalization gate ─────────────────────────────────
        # Held-out pool: every other train case. This is the entire
        # train split minus the current case, capped by the gate's
        # internal k-limit.
        held_out = [c for c in cases if c.case_id != case.case_id]
        gate = GeneralizationGate(
            held_out,
            config.runner,
            k=config.gate_k,
            seed=config.gate_seed,
        )

        for cand in proposal.candidates:
            gate_result = gate.validate(cand)
            report.n_rules_gated += 1
            report.gate_results.append({
                "rule_id": cand.rule_id,
                "action": cand.action[:80],
                "status": gate_result.new_status.value,
                "score": round(gate_result.score, 3),
                "trials": gate_result.total_trials,
                "reason": gate_result.reason,
            })
            # Track gate cost: each rule that actually ran the gate
            # incurred `total_trials` runner calls. (Deferred rules
            # — below min_total_for_evaluation — didn't call the
            # runner at all.)
            report.cost_gate += (
                gate_result.total_trials
                * config.runner.estimated_cost_per_call()
            )

            if gate_result.new_status is RuleStatus.ACTIVE:
                cand.status = RuleStatus.ACTIVE
                # Cache the rule's embedding so downstream retrieval
                # doesn't pay for it at inference time. Use the same
                # _rule_repr used by dense_rerank for consistency.
                emb = embedder.embed(_rule_repr(cand))
                cand.trigger.semantic_embedding = emb
                cand.trigger.embedding_model = embedder.model_id
                config.store.put(cand)
                report.n_rules_promoted += 1
                _log(config.progress_hook,
                     f"  promoted {cand.rule_id} "
                     f"(score={gate_result.score:.2f}, n={gate_result.total_trials})")
            elif gate_result.new_status is RuleStatus.DEPRECATED:
                report.n_rules_deprecated += 1
                _log(config.progress_hook,
                     f"  deprecated {cand.rule_id} "
                     f"(score={gate_result.score:.2f})")
            else:
                report.n_rules_deferred += 1
                _log(config.progress_hook,
                     f"  deferred {cand.rule_id} "
                     f"(reason={gate_result.reason[:60]})")

    report.finished_at = _utcnow()
    _log(
        config.progress_hook,
        f"[train] done in {report.elapsed_seconds:.1f}s, "
        f"total cost ${report.cost_total:.4f}, "
        f"{report.n_rules_promoted} rules promoted",
    )
    return report
