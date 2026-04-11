"""medreason-bench CLI — Phase 3 entrypoint.

Usage:
    python -m medreason_bench data build [--lcd <path>] [--target N]
                                          [--version v0.0] [--seed 42]
    python -m medreason_bench splits verify [--version v0.0]

The full CLI surface (planned for later phases) is listed in the top-level
rework plan. Phase 3 ships only the two subcommands needed to satisfy the
"done when" criterion: producing a real stratified manifest on disk and
verifying it with the LeakGuard-compatible fingerprints file.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from .data import parse_lcd_xml
from .data.case_builder import build_cases_from_lcd
from .eval.harness import EvalConfig, run_eval
from .leaderboard.build import build_entry, save_entry
from .splits import (
    SplitRatios,
    load_split,
    stratify,
    verify_manifest,
    write_manifest,
)


# Default fixture LCD — used when --lcd is not provided. Phase 3 only.
_DEFAULT_LCD = Path(__file__).parent / "data" / "fixtures" / "sample_lcd.xml"
# Default splits root.
_SPLITS_ROOT = Path(__file__).parent / "data" / "splits"
# Leaderboard output root.
_LEADERBOARD_ROOT = Path(__file__).parent / "leaderboard" / "entries"


def _cmd_data_build(args: argparse.Namespace) -> int:
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


def _cmd_eval(args: argparse.Namespace) -> int:
    # Runner selection. Only Claude is wired in Phase 4 (OpenAI/Gemini
    # are skeletons that raise NotImplementedError from .run()).
    if args.runner == "claude":
        from medreason.runners import ClaudeRunner
        runner = ClaudeRunner()
    elif args.runner in ("gpt4", "openai"):
        from medreason.runners import OpenAIRunner
        runner = OpenAIRunner()
    elif args.runner == "gemini":
        from medreason.runners import GeminiRunner
        runner = GeminiRunner()
    else:
        print(f"error: unknown runner {args.runner!r}", file=sys.stderr)
        return 2

    if args.split == "test" and args.memory is False:
        print(
            "warning: running zero-shot on the TEST split before the memory "
            "pipeline is wired is a smell. Prefer --split dev until Phase 5.",
            file=sys.stderr,
        )

    config = EvalConfig(
        runner=runner,
        splits_root=_SPLITS_ROOT,
        version=args.version,
        split=args.split,
        seeds=list(args.seeds),
        quick=args.quick,
        progress_hook=lambda msg: print(msg),
    )

    print(
        f"[eval] runner={runner.runner_id}  version={args.version}  "
        f"split={args.split}  seeds={config.seeds}"
        + ("  (quick)" if args.quick else "")
    )
    run = run_eval(config)

    cases_by_id = {c.case_id: c for c in run.cases}
    entry, metrics = build_entry(
        run, cases_by_id,
        submitter=args.submitter,
        code_revision=args.revision,
    )

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
        "--lcd", type=str, default=None,
        help="Path to an LCD XML file (defaults to the bundled fixture)",
    )
    data_build.add_argument(
        "--target", type=int, default=50,
        help="Number of cases to generate (default: 50)",
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

    # eval
    eval_p = sub.add_parser("eval", help="Run an AgentRunner against a split")
    eval_p.add_argument(
        "--runner", type=str, default="claude",
        choices=["claude", "gpt4", "openai", "gemini"],
        help="Which base runner to use (Phase 4: claude is the only wired adapter)",
    )
    eval_p.add_argument(
        "--memory", action="store_true",
        help="Wrap the runner in the memory pipeline (Phase 5+)",
    )
    eval_p.add_argument(
        "--no-memory", dest="memory", action="store_false",
        help="Run zero-shot (default)",
    )
    eval_p.set_defaults(memory=False)
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
