"""CLI entry point for Veridicus."""

from __future__ import annotations

import click

from .benchmark import compute_metrics, print_report, run_benchmark, save_results
from .generator import generate_local, generate_with_api, load_cases, save_cases
from .store import PatternStore


@click.group()
def cli():
    """Veridicus — Institutional reasoning memory for healthcare AI agents."""
    pass


@cli.command()
@click.option("--n", default=30, help="Number of cases to generate")
@click.option("--seed", default=42, help="Random seed")
@click.option("--local", is_flag=True, help="Use pre-written cases (no API needed)")
@click.option("--output", default="benchmark_cases.json", help="Output filename")
def generate(n: int, seed: int, local: bool, output: str):
    """Generate benchmark cases."""
    click.echo(f"Generating {n} benchmark cases (local={local}, seed={seed})...")

    if local:
        cases = generate_local(n, seed=seed)
    else:
        cases = generate_with_api(n, seed=seed)

    path = save_cases(cases, output)
    click.echo(f"Saved {len(cases)} cases to {path}")

    # Summary
    from collections import Counter
    outcomes = Counter(c.ground_truth_outcome.value for c in cases)
    difficulties = Counter(c.difficulty.value for c in cases)
    click.echo(f"  Outcomes: {dict(outcomes)}")
    click.echo(f"  Difficulty: {dict(difficulties)}")


@cli.command()
@click.option("--mode", type=click.Choice(["zero_shot", "memory", "both"]), default="both")
@click.option("--n", default=None, type=int, help="Number of cases to run (default: all)")
@click.option("--input", "input_file", default="benchmark_cases.json", help="Input cases file")
@click.option("--format", "fmt", type=click.Choice(["table", "json"]), default="table")
def run(mode: str, n: int | None, input_file: str, fmt: str):
    """Run benchmark evaluation."""
    click.echo(f"Loading cases from {input_file}...")
    cases = load_cases(input_file)

    if n is not None:
        cases = cases[:n]

    click.echo(f"Running benchmark: {len(cases)} cases, mode={mode}")
    store = PatternStore()

    try:
        result = run_benchmark(cases, mode=mode, store=store)
        metrics = compute_metrics(result)
        print_report(metrics, fmt=fmt)
        save_results(result, metrics)
    finally:
        store.close()


@cli.command()
@click.option("--format", "fmt", type=click.Choice(["table", "json"]), default="table")
def report(fmt: str):
    """View latest benchmark results."""
    import json
    from .config import RESULTS_DIR

    dash_path = RESULTS_DIR / "latest_dashboard.json"
    if not dash_path.exists():
        click.echo("No results found. Run a benchmark first: veridicus run")
        return

    data = json.loads(dash_path.read_text())
    metrics = data.get("metrics", {})
    print_report(metrics, fmt=fmt)


@cli.command()
def stats():
    """Show pattern store statistics."""
    store = PatternStore()
    try:
        s = store.get_stats()
        click.echo(f"Pattern Store Stats:")
        click.echo(f"  Total active patterns: {s['total_patterns']}")
        click.echo(f"  Avg success rate: {s['avg_success_rate']:.1%}")
        click.echo(f"  Config coverage: {s['coverage_configs']} unique configs")
    finally:
        store.close()


if __name__ == "__main__":
    cli()
