#!/usr/bin/env bash
# Fire all four v0.1 eval combinations after training has populated
# medreason_bench/data/stores/v0.1.db. Each subcommand writes its
# leaderboard entry + per-case JSON. The dashboard build_results.py
# script picks the right pair based on --version + --split.
#
# Usage:
#     bash mvp_dashboard/run_v0_1_evals.sh
#
# What this produces:
#     - zero-shot eval on v0.1 dev
#     - zero-shot eval on v0.1 train (the harder split — 58.3% baseline)
#     - memory eval on v0.1 dev
#     - memory eval on v0.1 train (where the rules came from — fair smoke
#       test for "did the pipeline encode the trick clauses correctly")
#     - results.json for the dashboard
set -euo pipefail

cd "$(dirname "$0")/.."

echo "[1/4] zero-shot dev (already done if cached)"
py -u -m medreason_bench eval \
    --model haiku --no-memory \
    --version v0.1 --split dev --seeds 11

echo
echo "[2/4] zero-shot train (already done if cached)"
py -u -m medreason_bench eval \
    --model haiku --no-memory \
    --version v0.1 --split train --seeds 11

echo
echo "[3/4] memory dev"
py -u -m medreason_bench eval \
    --model haiku --memory --no-rerank --top-k 3 \
    --version v0.1 --split dev --seeds 11

echo
echo "[4/4] memory train"
py -u -m medreason_bench eval \
    --model haiku --memory --no-rerank --top-k 3 \
    --version v0.1 --split train --seeds 11

echo
echo "Done. Now run:"
echo "  py mvp_dashboard/build_results.py --version v0.1 --split train --model haiku"
echo "and refresh http://localhost:3000."
