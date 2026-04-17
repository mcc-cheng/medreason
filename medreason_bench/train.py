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
    abstract_rule,
    analyze_failure,
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

    # Phase 51 — failure-driven extractor counters. Set when
    # TrainingConfig.include_failures is True and the agent reaches a
    # wrong determination, which triggers analyze_failure instead of
    # the normal critic→propose path.
    n_agent_wrong: int = 0
    n_failure_analyzer_invoked: int = 0
    n_failure_rules_proposed: int = 0
    n_failure_rules_rejected: int = 0
    n_failure_rules_promoted: int = 0

    # Accumulated cost (USD)
    cost_agent: float = 0.0
    cost_critic: float = 0.0
    cost_proposer: float = 0.0
    cost_gate: float = 0.0
    cost_failure_analyzer: float = 0.0

    # Miscellaneous diagnostics
    rejection_reasons: list[str] = field(default_factory=list)
    gate_results: list[dict] = field(default_factory=list)

    @property
    def cost_total(self) -> float:
        return (
            self.cost_agent
            + self.cost_critic
            + self.cost_proposer
            + self.cost_gate
            + self.cost_failure_analyzer
        )

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
            "n_agent_wrong": self.n_agent_wrong,
            "n_failure_analyzer_invoked": self.n_failure_analyzer_invoked,
            "n_failure_rules_proposed": self.n_failure_rules_proposed,
            "n_failure_rules_rejected": self.n_failure_rules_rejected,
            "n_failure_rules_promoted": self.n_failure_rules_promoted,
            "cost_agent": round(self.cost_agent, 5),
            "cost_critic": round(self.cost_critic, 5),
            "cost_proposer": round(self.cost_proposer, 5),
            "cost_gate": round(self.cost_gate, 5),
            "cost_failure_analyzer": round(self.cost_failure_analyzer, 5),
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
    # When None, the proposer reads each case's policy_excerpt directly
    # and runs in citation-wildcard mode (no structured indications/
    # limitations to validate against). Use this for multi-policy
    # fixtures like v0.1 adversarial.
    policy: Optional[LCDPolicy]
    train_cases: list[BenchmarkCase]
    version: str = "v0.0"
    split: str = "train"

    # Gate config
    gate_k: int = 3
    gate_seed: int = 0
    # Bypass the generalization gate entirely. Used for sparse multi-
    # policy fixtures (v0.1 adversarial) where every case has a
    # structurally unique trigger and the gate can't find held-out
    # matching cases. Critic verification remains the only quality
    # filter when this is True. Loses the "rules must generalize"
    # signal but enables training on small heterogeneous corpora.
    skip_gate: bool = False

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

    # Phase 51 — failure-driven extraction. When True, cases where the
    # agent reaches the WRONG determination are routed to
    # `analyze_failure` instead of being discarded. This is the
    # architectural fix for the chicken-and-egg identified in Phase 6:
    # the hardest cases (adv_004, adv_006, adv_014, adv_015 in v0.2)
    # never produced a rule because the agent always got them wrong,
    # so the store had zero rules to retrieve for them at eval time.
    include_failures: bool = False

    # LLM client used for failure analysis. Defaults to proposer_llm
    # when None. Could be a different vendor for cross-vendor
    # corroboration (Phase 7 concern).
    failure_analyzer_llm: Optional[LLMClient] = None

    # Metacognitive rule proposer switch. When True, the rule_proposer
    # uses medreason/prompts/metacognitive_rule_proposer.txt instead
    # of the default prompt, emitting PROTECT-THE-DEFAULT /
    # OVERRIDE-THE-DEFAULT style rules that describe the AI's default
    # reasoning pattern + the specific override/protection. This is
    # the v0.3 experiment hypothesis: metacognitive rules transfer
    # better than additive checks because they tell future retrievals
    # how the AI fails, not just what the policy says. Relaxes the
    # action word cap to 30 words (metacognitive rules need both the
    # default and the override in one sentence).
    use_metacognitive_proposer: bool = False

    # Rule abstraction pass. When True, every extracted rule (both
    # normal and failure-derived) is run through the rule_abstractor
    # before promotion. The abstractor rewrites case-specific text
    # (drug names, diagnoses, procedures) into concept-level language
    # so rules transfer across cases with different specific entities
    # but the same structural pattern.
    abstract_rules: bool = False

    # LLM client for rule abstraction. Defaults to proposer_llm.
    abstractor_llm: Optional[LLMClient] = None


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

        # Per-case state that the shared promotion block below uses to
        # route counters and logs to either the normal path or the
        # failure-analyzer path.
        proposal = None
        is_failure_path = False

        if agent_result.correct:
            # ── Normal path: critic → trace → propose ──────────────────
            report.n_agent_correct += 1

            # Stage 2: critic verification
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

            # Stage 3: rule proposer
            # If running on a multi-policy fixture (config.policy is None),
            # pass the case's own policy_excerpt as the source text. The
            # proposer's citation validator becomes a wildcard in that
            # mode — it accepts any well-formed §X.Y citation without
            # requiring a structured policy lookup.
            report.n_proposer_invoked += 1
            # Metacognitive mode: swap the prompt file and relax the
            # action word cap so rules can encode both the default
            # reasoning and the override in one sentence.
            proposer_prompt = (
                "metacognitive_rule_proposer.txt"
                if config.use_metacognitive_proposer
                else "rule_proposer.txt"
            )
            proposer_action_max_words = (
                30 if config.use_metacognitive_proposer else 25
            )
            if config.policy is None:
                proposal = propose_rules(
                    critic_result.trace,
                    None,
                    config.proposer_llm,
                    supporting_case_ids=[case.case_id],
                    seed=config.seed,
                    policy_excerpt_text=case.policy_excerpt,
                    prompt_file=proposer_prompt,
                    action_max_words=proposer_action_max_words,
                )
            else:
                proposal = propose_rules(
                    critic_result.trace,
                    config.policy,
                    config.proposer_llm,
                    supporting_case_ids=[case.case_id],
                    seed=config.seed,
                    prompt_file=proposer_prompt,
                    action_max_words=proposer_action_max_words,
                )
            report.cost_proposer += config.runner.estimated_cost_per_call()
            report.n_rules_proposed += proposal.n_candidates
            report.n_rules_rejected += proposal.n_rejected
        else:
            # ── Phase 51: failure-driven extraction ────────────────────
            # Agent got this case wrong. Under the default config we
            # discard the case (chicken-and-egg: the hardest cases
            # never produce a rule). Under include_failures=True we
            # route the case to analyze_failure instead, which reasons
            # through the policy to find a corrective rule that would
            # have led a reviewer to the ground-truth outcome.
            report.n_agent_wrong += 1
            if not config.include_failures:
                _log(config.progress_hook, "  agent wrong — skipping")
                continue

            _log(config.progress_hook,
                 f"  agent wrong (pred={agent_result.determination.value}, "
                 f"gt={case.ground_truth_outcome.value}) — running failure analyzer")
            report.n_failure_analyzer_invoked += 1
            is_failure_path = True

            analyzer_llm = config.failure_analyzer_llm or config.proposer_llm
            proposal = analyze_failure(
                case,
                agent_determination=agent_result.determination,
                ground_truth_outcome=case.ground_truth_outcome,
                llm_client=analyzer_llm,
                policy=config.policy,
                supporting_case_ids=[case.case_id],
                seed=config.seed,
            )
            report.cost_failure_analyzer += config.runner.estimated_cost_per_call()
            report.n_failure_rules_proposed += proposal.n_candidates
            report.n_failure_rules_rejected += proposal.n_rejected

        # ── Shared rejection/empty handling ──────────────────────────────
        if proposal.n_rejected:
            for _, reason in proposal.rejected[:3]:
                report.rejection_reasons.append(reason)

        if not proposal.candidates:
            _log(config.progress_hook,
                 "  proposer returned no candidates" if not is_failure_path
                 else "  failure analyzer returned no candidates")
            continue

        _log(config.progress_hook,
             f"  {'failure analyzer' if is_failure_path else 'proposer'}: "
             f"{proposal.n_candidates} candidate(s), "
             f"{proposal.n_rejected} rejected")

        # ── Stage 4: generalization gate (shared promotion block) ───────
        # Held-out pool: every other train case. This is the entire
        # train split minus the current case, capped by the gate's
        # internal k-limit.
        #
        # When config.skip_gate is True (used for sparse multi-policy
        # fixtures where every case has a unique trigger and the gate
        # finds 0 matching cases), we promote every verified candidate
        # directly. For normal rules, critic verification is the only
        # quality filter in this mode. For failure-derived rules, the
        # only quality filter is the failure analyzer's citation +
        # PII + action-length validators.
        if config.skip_gate:
            for cand in proposal.candidates:
                report.n_rules_gated += 1
                if config.abstract_rules:
                    abs_llm = config.abstractor_llm or config.proposer_llm
                    old_action = cand.action
                    cand = abstract_rule(cand, abs_llm, seed=config.seed)
                    if cand.action != old_action:
                        _log(config.progress_hook,
                             f"  abstracted: {old_action[:60]} -> {cand.action[:60]}")
                report.gate_results.append({
                    "rule_id": cand.rule_id,
                    "action": cand.action[:80],
                    "status": "active",
                    "score": 1.0,
                    "trials": 0,
                    "reason": (
                        "skip_gate=True (failure-derived, validators only)"
                        if is_failure_path
                        else "skip_gate=True (critic-verified only)"
                    ),
                })
                cand.status = RuleStatus.ACTIVE
                emb = embedder.embed(_rule_repr(cand))
                cand.trigger.semantic_embedding = emb
                cand.trigger.embedding_model = embedder.model_id
                config.store.put(cand)
                if is_failure_path:
                    report.n_failure_rules_promoted += 1
                else:
                    report.n_rules_promoted += 1
                _log(config.progress_hook,
                     f"  promoted {cand.rule_id} (skip_gate, "
                     f"{'failure-derived' if is_failure_path else 'critic-verified'})")
            continue

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
                if config.abstract_rules:
                    abs_llm = config.abstractor_llm or config.proposer_llm
                    old_action = cand.action
                    cand = abstract_rule(cand, abs_llm, seed=config.seed)
                    if cand.action != old_action:
                        _log(config.progress_hook,
                             f"  abstracted: {old_action[:60]} -> {cand.action[:60]}")
                cand.status = RuleStatus.ACTIVE
                emb = embedder.embed(_rule_repr(cand))
                cand.trigger.semantic_embedding = emb
                cand.trigger.embedding_model = embedder.model_id
                config.store.put(cand)
                if is_failure_path:
                    report.n_failure_rules_promoted += 1
                else:
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
