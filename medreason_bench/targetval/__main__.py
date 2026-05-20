"""CLI for the target-validation benchmark.

Mirrors the medreason_bench top-level CLI pattern. Subcommands:

    data  build-mapk-retro --version v0.1 --out <dir>
    data  ingest-customer  --customer <tag> --targets <csv> --outcomes <csv>
    swarm dryrun           --campaign synthetic --seeds 11 17 --no-memory
    card  write            --out deliverables/

Status: SKELETON. The CLI parses commands and prints what it would do.
Real wiring lands once the rest of the pipeline (swarm.run with LLM,
LayerRouter store wiring) is implemented.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cmd_data_build_mapk_retro(args: argparse.Namespace) -> int:
    from .mapk_retro import build_mapk_retro_seed

    cases = build_mapk_retro_seed()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "mapk_retro_seed.json"
    path.write_text(json.dumps([c.model_dump(mode="json") for c in cases], indent=2))
    print(f"Wrote {len(cases)} MAPK seed cases to {path}")
    return 0


def _cmd_data_ingest_customer(args: argparse.Namespace) -> int:
    from .recursion_ingest import (
        load_customer_targets,
        parse_outcomes_csv,
        parse_targets_csv,
    )

    targets = parse_targets_csv(args.targets)
    outcomes = parse_outcomes_csv(args.outcomes) if args.outcomes else None
    report = load_customer_targets(
        targets, customer_tag=args.customer, outcomes_iter=outcomes
    )
    print(
        f"Loaded {report.n_cases} customer cases for tenant {args.customer!r} "
        f"({report.n_rejected} rejected)"
    )
    for row, reason in report.rejected:
        print(f"  rejected: {row.get('case_id', '<no-case_id>')}: {reason}")
    return 0


def _cmd_swarm_dryrun(args: argparse.Namespace) -> int:
    from medreason.llm.base import FakeLLMClient
    from medreason.targetval.layers import LayerRouter, LayerStorePaths
    from medreason.targetval.swarm import SwarmRunner

    if args.campaign == "synthetic":
        from .synthetic import SYNTHETIC_CAMPAIGN_ID, build_synthetic_targets

        cases = build_synthetic_targets()
        campaign_id = SYNTHETIC_CAMPAIGN_ID
    else:
        print(f"Unknown campaign {args.campaign!r}", file=sys.stderr)
        return 2

    llm = FakeLLMClient(default_text="{}")
    router = LayerRouter(
        LayerStorePaths(universal=Path("/dev/null"), disease={}, campaign={})
    )
    runner = SwarmRunner(llm, router, max_workers=1)
    report = runner.run(cases, campaign_id=campaign_id, seed=args.seeds[0])
    print(json.dumps(
        {
            "campaign_id": report.campaign_id,
            "n_targets": report.n_targets,
            "ranking": report.ranking,
            "total_cost_usd": report.total_cost_usd,
        },
        indent=2,
    ))
    return 0


def _cmd_card_write(args: argparse.Namespace) -> int:
    from .prediction_card import default_v0_1_card, write_prediction_card

    card = default_v0_1_card()
    path = write_prediction_card(card, args.out)
    print(f"Wrote prediction card to {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="medreason_bench.targetval")
    sub = p.add_subparsers(dest="cmd")

    data = sub.add_parser("data")
    data_sub = data.add_subparsers(dest="data_cmd")

    build_mapk = data_sub.add_parser("build-mapk-retro")
    build_mapk.add_argument("--version", default="v0.1")
    build_mapk.add_argument("--out", required=True)
    build_mapk.set_defaults(func=_cmd_data_build_mapk_retro)

    ingest = data_sub.add_parser("ingest-customer")
    ingest.add_argument("--customer", required=True)
    ingest.add_argument("--targets", required=True)
    ingest.add_argument("--outcomes", default=None)
    ingest.set_defaults(func=_cmd_data_ingest_customer)

    swarm = sub.add_parser("swarm")
    swarm_sub = swarm.add_subparsers(dest="swarm_cmd")
    dryrun = swarm_sub.add_parser("dryrun")
    dryrun.add_argument("--campaign", default="synthetic")
    dryrun.add_argument("--seeds", nargs="+", type=int, default=[11])
    dryrun.add_argument("--no-memory", action="store_true")
    dryrun.set_defaults(func=_cmd_swarm_dryrun)

    card = sub.add_parser("card")
    card_sub = card.add_subparsers(dest="card_cmd")
    card_write = card_sub.add_parser("write")
    card_write.add_argument("--out", required=True)
    card_write.set_defaults(func=_cmd_card_write)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
