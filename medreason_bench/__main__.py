"""medreason-bench CLI.

Subcommands:
    data build       — parse an LCD and build a stratified manifest
    splits verify    — re-hash a manifest and confirm LeakGuard compat
    train            — populate the memory store from a training split
    eval             — run an AgentRunner (optionally memory-wrapped)
                       against a split and emit a LeaderboardEntry

The train + eval subcommands share a persistent SQLite store per
manifest version: medreason_bench/data/stores/<version>.db. Running
`train` overwrites it; running `eval --memory` reads from it.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

from .data import parse_lcd_xml
from .data.adversarial_cases import build_adversarial_cases
from .data.aetna_lumbar_mri_cases import build_aetna_lumbar_mri_cases
from .data.case_builder import build_cases_from_lcd
from .data.lcd_edge_cases import build_lcd_edge_cases
from .eval.harness import EvalConfig, run_eval
from .leaderboard.build import build_entry, save_entry
from .splits import (
    SplitRatios,
    load_split,
    stratify,
    verify_manifest,
    write_manifest,
)


# Default fixture LCD — used when --lcd is not provided.
_DEFAULT_LCD = Path(__file__).parent / "data" / "fixtures" / "sample_lcd.xml"
# Default splits root.
_SPLITS_ROOT = Path(__file__).parent / "data" / "splits"
# Leaderboard output root.
_LEADERBOARD_ROOT = Path(__file__).parent / "leaderboard" / "entries"
# Persistent memory store root. Each manifest version gets its own sqlite file.
_STORES_ROOT = Path(__file__).parent / "data" / "stores"
# Training report output root.
_REPORTS_ROOT = Path(__file__).parent / "data" / "training_reports"


def _store_path(version: str) -> Path:
    return _STORES_ROOT / f"{version}.db"


def _build_claude_runner(
    model_alias: str,
    *,
    include_policy: bool = False,
    policy_max_chars: int | None = None,
):
    """Build a ClaudeRunner with a CLI-alias-resolved model."""
    from medreason.runners import ClaudeRunner, resolve_claude_model
    pinned = resolve_claude_model(model_alias)
    return ClaudeRunner(
        model=pinned,
        include_policy=include_policy,
        policy_max_chars=policy_max_chars,
    )


def _build_claude_llm_client(model_alias: str):
    from medreason.llm import ClaudeLLMClient
    from medreason.runners import resolve_claude_model
    return ClaudeLLMClient(model=resolve_claude_model(model_alias))


def _cmd_data_build(args: argparse.Namespace) -> int:
    if args.source == "adversarial":
        print(f"[data build] using adversarial v0.1 fixture (hand-authored)")
        cases = build_adversarial_cases()
        print(f"  loaded {len(cases)} adversarial cases")
    elif args.source == "lcd_edge":
        print(f"[data build] using LCD edge-case fixture (xlsx-derived)")
        cases = build_lcd_edge_cases()
        print(f"  loaded {len(cases)} LCD edge cases")
    elif args.source == "aetna_lumbar":
        print(f"[data build] using Aetna lumbar MRI within-domain fixture (30 cases)")
        cases = build_aetna_lumbar_mri_cases()
        print(f"  loaded {len(cases)} Aetna lumbar MRI cases")
    elif args.source == "combined":
        print(f"[data build] combining v0.1 adversarial + LCD edge fixture")
        cases = build_adversarial_cases() + build_lcd_edge_cases()
        oc = Counter(c.ground_truth_outcome.value for c in cases)
        print(f"  loaded {len(cases)} cases  (combined)")
        print(f"  outcomes: {dict(oc)}")
    else:
        lcd_path = Path(args.lcd) if args.lcd else _DEFAULT_LCD
        if not lcd_path.exists():
            print(f"error: LCD file not found: {lcd_path}", file=sys.stderr)
            return 2

        print(f"[data build] reading LCD from {lcd_path}")
        policy = parse_lcd_xml(lcd_path)
        print(
            f"  {policy.document_id}: {policy.title}  "
            f"({len(policy.cpt_codes)} CPTs, {len(policy.indications)} criteria, "
            f"{len(policy.limitations)} limitations)"
        )

        print(f"[data build] generating {args.target} cases (seed={args.seed})")
        cases = build_cases_from_lcd(
            policy, target_count=args.target, seed=args.seed
        )
    outcome_counts = Counter(c.ground_truth_outcome.value for c in cases)
    diff_counts = Counter(c.difficulty.value for c in cases)
    print(
        f"  outcomes: {dict(outcome_counts)}\n"
        f"  difficulty: {dict(diff_counts)}"
    )

    ratios = SplitRatios(train=0.6, dev=0.2, test=0.2)
    print(f"[data build] stratifying (ratios={ratios.as_tuple()})")
    splits = stratify(cases, ratios=ratios, seed=args.seed)
    for name in ("train", "dev", "test"):
        oc = Counter(c.ground_truth_outcome.value for c in splits[name])
        print(f"  {name}: {len(splits[name])} cases  {dict(oc)}")

    out_dir = _SPLITS_ROOT / args.version
    print(f"[data build] writing manifest to {out_dir}")
    fingerprints = write_manifest(splits, out_dir)
    total = sum(len(v) for v in fingerprints.values())
    print(
        f"  wrote train.jsonl dev.jsonl test.jsonl fingerprints.json "
        f"({total} case fingerprints)"
    )

    verify_manifest(out_dir)
    print(f"[data build] verify OK")
    return 0


def _cmd_splits_verify(args: argparse.Namespace) -> int:
    out_dir = _SPLITS_ROOT / args.version
    if not out_dir.exists():
        print(f"error: no splits directory at {out_dir}", file=sys.stderr)
        return 2
    print(f"[splits verify] re-hashing manifest at {out_dir}")
    fps = verify_manifest(out_dir)
    for split_name, items in fps.items():
        print(f"  {split_name}: {len(items)} case(s) OK")

    # Smoke-load the LeakGuard against the freshly verified manifest. If
    # this fails, the fingerprints file is structurally incompatible with
    # the store layer — that would be a Phase 1 regression.
    from medreason.store import LeakGuard
    lg = LeakGuard.from_fingerprint_file(out_dir / "fingerprints.json")
    print(
        f"  LeakGuard loaded: train={len(lg.train_case_ids)} "
        f"dev={len(lg.dev_case_ids)} test={len(lg.test_case_ids)}"
    )
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    """Populate the memory store from a training split.

    Walks the train split with the configured base runner, runs critic
    verification → rule proposer → generalization gate on each correct
    case, and persists promoted rules to medreason_bench/data/stores/<version>.db.
    """
    from medreason.retrieval.embedder import FakeEmbedder
    from medreason.store import RuleStore, TraceStore

    from .train import TrainingConfig, run_training

    # Load the training split first so we can detect adversarial fixtures
    # by checking the version tag (v0.1 = adversarial; everything else
    # uses the bundled LCD).
    split_dir = _SPLITS_ROOT / args.version
    if not split_dir.exists():
        print(f"error: no splits directory at {split_dir}", file=sys.stderr)
        return 2
    train_cases = load_split(split_dir, args.split)
    if args.max_cases:
        train_cases = train_cases[: args.max_cases]

    # Detect multi-policy mode: --multi-policy flag, OR no --lcd given
    # AND the case_ids look adversarial (start with 'adv_'). In multi-
    # policy mode the proposer reads each case's policy_excerpt and the
    # citation validator is wildcarded.
    is_multi_policy = (
        args.multi_policy
        or (args.lcd is None and any(c.case_id.startswith("adv_")
                                     for c in train_cases))
    )

    # Build runner + LLM clients (same model for all three slots in the
    # cheap run — cross-vendor critic is Phase 7)
    runner = _build_claude_runner(args.model)
    critic_llm = _build_claude_llm_client(args.critic_model or args.model)
    proposer_llm = _build_claude_llm_client(args.proposer_model or args.model)

    # Fresh store (overwrite previous training pass)
    _STORES_ROOT.mkdir(parents=True, exist_ok=True)
    db_path = _store_path(args.version)
    if db_path.exists() and not args.append:
        db_path.unlink()
    conn = sqlite3.connect(str(db_path))
    rule_store = RuleStore(conn)
    trace_store = TraceStore(conn)

    policy = None
    if is_multi_policy:
        print("[train] multi-policy mode: proposer will read each case's "
              "policy_excerpt directly (citation validator wildcarded)")
    else:
        lcd_path = Path(args.lcd) if args.lcd else _DEFAULT_LCD
        if not lcd_path.exists():
            print(f"error: LCD file not found: {lcd_path}", file=sys.stderr)
            return 2
        policy = parse_lcd_xml(lcd_path)
        print(
            f"[train] policy {policy.document_id}: {policy.title}  "
            f"({len(policy.cpt_codes)} CPTs, {len(policy.indications)} criteria)"
        )
    print(
        f"[train] runner={runner.runner_id}  critic={critic_llm.model_version}  "
        f"proposer={proposer_llm.model_version}"
    )
    print(
        f"[train] {len(train_cases)} training case(s), gate_k={args.gate_k}, "
        f"store={db_path}"
    )

    config = TrainingConfig(
        runner=runner,
        critic_llm=critic_llm,
        proposer_llm=proposer_llm,
        store=rule_store,
        trace_store=trace_store,
        policy=policy,
        train_cases=train_cases,
        version=args.version,
        split=args.split,
        gate_k=args.gate_k,
        gate_seed=args.seed,
        skip_gate=args.skip_gate,
        embedder=FakeEmbedder(),
        progress_hook=lambda msg: print(msg),
        seed=args.seed,
        include_failures=args.include_failures,
    )

    report = run_training(config)
    conn.commit()
    conn.close()

    print()
    print("=== training report ===")
    print(f"  cases seen              : {report.n_cases_seen}")
    print(f"  agent correct           : {report.n_agent_correct}")
    print(f"  critic agreed           : {report.n_critic_agreed}")
    print(f"  traces stored           : {report.n_traces_stored}")
    print(f"  rules proposed          : {report.n_rules_proposed}")
    print(f"  rules rejected (proposer): {report.n_rules_rejected}")
    print(f"  rules gated             : {report.n_rules_gated}")
    print(f"  rules PROMOTED (ACTIVE) : {report.n_rules_promoted}")
    print(f"  rules deprecated        : {report.n_rules_deprecated}")
    print(f"  rules deferred          : {report.n_rules_deferred}")
    if args.include_failures:
        print()
        print(f"  agent wrong (seen)      : {report.n_agent_wrong}")
        print(f"  failure analyzer runs   : {report.n_failure_analyzer_invoked}")
        print(f"  failure rules proposed  : {report.n_failure_rules_proposed}")
        print(f"  failure rules rejected  : {report.n_failure_rules_rejected}")
        print(f"  failure rules PROMOTED  : {report.n_failure_rules_promoted}")
    print()
    print(f"  cost (agent/critic/proposer/gate/failure): "
          f"${report.cost_agent:.4f} / ${report.cost_critic:.4f} / "
          f"${report.cost_proposer:.4f} / ${report.cost_gate:.4f} / "
          f"${report.cost_failure_analyzer:.4f}")
    print(f"  TOTAL COST              : ${report.cost_total:.4f}")
    print(f"  elapsed                 : {report.elapsed_seconds:.1f}s")

    # Persist the training report JSON
    _REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    report_path = _REPORTS_ROOT / f"{args.version}_{args.model}.json"
    report_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[train] report saved to {report_path}")
    print(f"[train] store saved to {db_path}")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from medreason.retrieval.embedder import FakeEmbedder
    from medreason.runners import MemoryRunner
    from medreason.store import RuleStore

    # Build the base runner. Claude is the only fully-wired option;
    # OpenAI/Gemini are Phase 7 skeletons.
    if args.runner == "claude":
        base_runner = _build_claude_runner(
            args.model,
            include_policy=args.include_policy,
            policy_max_chars=args.policy_max_chars,
        )
    elif args.runner in ("gpt4", "openai"):
        from medreason.runners import OpenAIRunner
        base_runner = OpenAIRunner()
    elif args.runner == "gemini":
        from medreason.runners import GeminiRunner
        base_runner = GeminiRunner()
    else:
        print(f"error: unknown runner {args.runner!r}", file=sys.stderr)
        return 2

    # Memory wrapping (if --memory)
    runner_for_harness = base_runner
    store_conn = None
    memory_note = ""
    if args.memory:
        db_path = _store_path(args.version)
        if not db_path.exists():
            print(
                f"error: memory store not found at {db_path}. "
                f"Run `medreason-bench train --version {args.version}` first.",
                file=sys.stderr,
            )
            return 2
        store_conn = sqlite3.connect(str(db_path))
        rule_store = RuleStore(store_conn)

        # Reranker LLM: off by default for the cheap run, wired
        # only if --rerank is explicitly passed.
        reranker_llm = None
        if args.rerank:
            reranker_llm = _build_claude_llm_client(args.model)

        runner_for_harness = MemoryRunner(
            base_runner=base_runner,
            store=rule_store,
            embedder=FakeEmbedder(),
            reranker_llm=reranker_llm,
            tier3_top_k=args.top_k,
        )
        memory_note = f" [memory from {db_path.name}]"

    if args.split == "test" and not args.memory:
        print(
            "warning: running zero-shot on the TEST split is allowed but "
            "note that test-phase tripwires still enforce prompts-lock "
            "verification.",
            file=sys.stderr,
        )

    config = EvalConfig(
        runner=runner_for_harness,
        splits_root=_SPLITS_ROOT,
        version=args.version,
        split=args.split,
        seeds=list(args.seeds),
        quick=args.quick,
        progress_hook=lambda msg: print(msg),
    )

    print(
        f"[eval] runner={runner_for_harness.runner_id}  version={args.version}  "
        f"split={args.split}  seeds={config.seeds}"
        + ("  (quick)" if args.quick else "")
        + memory_note
    )
    run = run_eval(config)

    cases_by_id = {c.case_id: c for c in run.cases}
    entry, metrics = build_entry(
        run, cases_by_id,
        submitter=args.submitter,
        code_revision=args.revision,
    )

    if store_conn is not None:
        store_conn.close()

    print()
    print(f"  n_cases              : {entry.n_cases}")
    print(f"  total calls          : {run.total_calls}")
    print(f"  accuracy (mean, CI)  : {entry.accuracy_mean:.3f}  "
          f"[{entry.accuracy_ci_low:.3f}, {entry.accuracy_ci_high:.3f}]")
    print(f"  macro F1             : {entry.macro_f1:.3f}")
    print(f"  Brier / ECE          : {entry.brier:.3f} / {entry.ece:.3f}")
    print(f"  avg total tokens     : {entry.avg_total_tokens:.1f}")
    print(f"  latency p50 / p95    : {entry.p50_latency_ms:.0f}ms / {entry.p95_latency_ms:.0f}ms")
    print(f"  cost per case        : ${entry.cost_per_case_usd:.5f}")
    print(f"  total cost           : ${entry.total_cost_usd:.4f}")
    print(f"  pattern utilization  : {entry.pattern_utilization}")

    out_path = save_entry(entry, _LEADERBOARD_ROOT)
    print(f"[eval] leaderboard entry saved to {out_path}")

    # Also save the flat per-case results JSON so the dashboard can
    # render per-case breakdowns without re-running the eval.
    per_case_path = _LEADERBOARD_ROOT / f"{entry.runner_id.replace(':', '_')}__{args.version}__{args.split}__cases.json"
    per_case_payload = {
        "runner_id": entry.runner_id,
        "version": args.version,
        "split": args.split,
        "mode": "memory" if args.memory else "zero_shot",
        "seeds": list(args.seeds),
        "cases": [
            {
                "case_id": r.case_id,
                "determination": r.determination.value,
                "ground_truth": cases_by_id[r.case_id].ground_truth_outcome.value,
                "correct": r.correct,
                "confidence": r.confidence,
                "reasoning_chain": r.reasoning_chain[:500],
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "cost_usd": r.cost_usd,
                "latency_ms": r.latency_ms,
                "seed": r.seed,
                "retrieved_rule_ids": list(r.retrieved_rule_ids),
                "applied_rules": [
                    {"rule_id": a.rule_id, "applied": a.applied,
                     "rationale": a.rationale}
                    for a in r.applied_rules
                ],
            }
            for r in run.flat_results()
        ],
    }
    per_case_path.write_text(
        json.dumps(per_case_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[eval] per-case results saved to {per_case_path}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="medreason-bench",
        description="MedReason-Bench CLI (Phase 3: data build + splits verify).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # data
    data_p = sub.add_parser("data", help="Data ingestion + case building")
    data_sub = data_p.add_subparsers(dest="data_command", required=True)
    data_build = data_sub.add_parser(
        "build", help="Parse an LCD and build a stratified manifest"
    )
    data_build.add_argument(
        "--source", type=str, default="lcd",
        choices=["lcd", "adversarial", "lcd_edge", "aetna_lumbar", "combined"],
        help="Case source: 'lcd' (template-expanded from an LCD XML, "
             "default), 'adversarial' (v0.1 hand-authored fixture), "
             "'lcd_edge' (xlsx-derived 30 LCD edge cases), "
             "'aetna_lumbar' (v0.3 within-domain Aetna lumbar MRI "
             "fixture, 30 hand-authored cases), or "
             "'combined' (v0.1 adversarial + xlsx LCD edge — 50 cases "
             "with balanced approve/deny ratio)",
    )
    data_build.add_argument(
        "--lcd", type=str, default=None,
        help="Path to an LCD XML file (only used when --source=lcd)",
    )
    data_build.add_argument(
        "--target", type=int, default=50,
        help="Number of cases to generate (only used when --source=lcd)",
    )
    data_build.add_argument(
        "--version", type=str, default="v0.0",
        help="Manifest version tag (default: v0.0)",
    )
    data_build.add_argument(
        "--seed", type=int, default=42,
        help="Deterministic seed (default: 42)",
    )
    data_build.set_defaults(func=_cmd_data_build)

    # splits
    splits_p = sub.add_parser("splits", help="Manifest verification")
    splits_sub = splits_p.add_subparsers(dest="splits_command", required=True)
    splits_verify = splits_sub.add_parser(
        "verify", help="Re-hash a manifest and confirm LeakGuard compatibility"
    )
    splits_verify.add_argument(
        "--version", type=str, default="v0.0",
        help="Manifest version tag to verify (default: v0.0)",
    )
    splits_verify.set_defaults(func=_cmd_splits_verify)

    # train
    train_p = sub.add_parser(
        "train", help="Populate the memory store from a training split"
    )
    train_p.add_argument(
        "--version", type=str, default="v0.0",
        help="Manifest version to load the training split from",
    )
    train_p.add_argument(
        "--split", type=str, default="train",
        choices=["train", "dev"],
        help="Which split to train on (default: train)",
    )
    train_p.add_argument(
        "--model", type=str, default="haiku",
        help="Claude model alias or pinned id (default: haiku)",
    )
    train_p.add_argument(
        "--critic-model", type=str, default=None,
        help="Optional: different model for the critic (default: same as --model)",
    )
    train_p.add_argument(
        "--proposer-model", type=str, default=None,
        help="Optional: different model for the proposer (default: same as --model)",
    )
    train_p.add_argument(
        "--gate-k", type=int, default=5,
        help="Held-out cases per candidate rule in the gen gate. "
             "Must be >= 5 to meet the default promotion threshold "
             "(min_total_for_promote=5). Lower values leave rules as "
             "CANDIDATE instead of promoting to ACTIVE.",
    )
    train_p.add_argument(
        "--lcd", type=str, default=None,
        help="Path to the LCD XML policy (default: bundled fixture). "
             "Ignored when --multi-policy is set or when training cases "
             "are detected as adversarial (case_ids starting with 'adv_').",
    )
    train_p.add_argument(
        "--multi-policy", action="store_true",
        help="Multi-policy mode: each case's policy_excerpt is the source "
             "the proposer reads from (instead of a single LCD XML). "
             "Citation validator becomes a wildcard.",
    )
    train_p.add_argument(
        "--skip-gate", action="store_true",
        help="Bypass the generalization gate. Promotes every critic-"
             "verified candidate directly. Used for sparse fixtures "
             "where the gate's structural-trigger matching can't find "
             "held-out cases. Critic verification remains the only "
             "quality filter.",
    )
    train_p.add_argument(
        "--seed", type=int, default=0,
        help="Reproducibility seed forwarded to every runner call",
    )
    train_p.add_argument(
        "--max-cases", type=int, default=None,
        help="Cap the number of training cases processed (for cheap runs)",
    )
    train_p.add_argument(
        "--append", action="store_true",
        help="Append to an existing store instead of overwriting",
    )
    train_p.add_argument(
        "--include-failures", action="store_true",
        help="Phase 51: route wrong-answer cases through the failure "
             "analyzer instead of discarding them. Extracts corrective "
             "rules from the cases the agent couldn't solve, breaking "
             "the chicken-and-egg where hard cases never produced a rule.",
    )
    train_p.set_defaults(func=_cmd_train)

    # eval
    eval_p = sub.add_parser("eval", help="Run an AgentRunner against a split")
    eval_p.add_argument(
        "--runner", type=str, default="claude",
        choices=["claude", "gpt4", "openai", "gemini"],
        help="Which base runner to use (Phase 4: claude is the only wired adapter)",
    )
    eval_p.add_argument(
        "--model", type=str, default="sonnet",
        help="Claude model alias or pinned id (default: sonnet)",
    )
    eval_p.add_argument(
        "--memory", action="store_true",
        help="Wrap the base runner in the memory pipeline",
    )
    eval_p.add_argument(
        "--no-memory", dest="memory", action="store_false",
        help="Run zero-shot (default)",
    )
    eval_p.set_defaults(memory=False)
    eval_p.add_argument(
        "--include-policy", action="store_true",
        help="RAG mode: include the case's policy_excerpt in the agent's "
             "user message. Mirrors what real prior auth companies have "
             "(retrieved policy chunk in context). The runner_id gets "
             "':rag' appended so leaderboard entries differentiate.",
    )
    eval_p.add_argument(
        "--policy-max-chars", type=int, default=None,
        help="Sparse-RAG mode: truncate the policy excerpt to the first "
             "N chars before injecting. Simulates real-world RAG that "
             "retrieves a chunk header but misses footnotes / "
             "operational nuance. The runner_id gets ':sparseN' appended. "
             "Only effective when --include-policy is also set.",
    )
    eval_p.add_argument(
        "--rerank", action="store_true",
        help="Enable Tier 3 LLM reranker (extra cost per memory call)",
    )
    eval_p.add_argument(
        "--no-rerank", dest="rerank", action="store_false",
        help="Skip the reranker (default — cheaper, uses Tier 2 ordering)",
    )
    eval_p.set_defaults(rerank=False)
    eval_p.add_argument(
        "--top-k", type=int, default=6,
        help="Max retrieved rules per case (default: 6)",
    )
    eval_p.add_argument(
        "--split", type=str, default="dev",
        choices=["train", "dev", "test"],
        help="Which split to evaluate against",
    )
    eval_p.add_argument(
        "--version", type=str, default="v0.0",
        help="Manifest version to load",
    )
    eval_p.add_argument(
        "--seeds", type=int, nargs="+", default=[11, 17, 23, 29, 31],
        help="Seed set for multi-seed eval (default: 5 primes)",
    )
    eval_p.add_argument(
        "--quick", action="store_true",
        help="Sample 10 stratified cases for cheap dev iteration",
    )
    eval_p.add_argument(
        "--submitter", type=str, default="local",
        help="Leaderboard submitter id",
    )
    eval_p.add_argument(
        "--revision", type=str, default="",
        help="Code revision stamp (e.g., git sha)",
    )
    eval_p.set_defaults(func=_cmd_eval)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
