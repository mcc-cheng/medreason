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
from .splits import (
    SplitRatios,
    stratify,
    verify_manifest,
    write_manifest,
)


# Default fixture LCD — used when --lcd is not provided. Phase 3 only.
_DEFAULT_LCD = Path(__file__).parent / "data" / "fixtures" / "sample_lcd.xml"
# Default splits root.
_SPLITS_ROOT = Path(__file__).parent / "data" / "splits"


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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
