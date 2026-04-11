#!/usr/bin/env bash
# Run zero-shot + memory eval back-to-back, then build the dashboard JSON.
# Assumes `train` has already populated medreason_bench/data/stores/v0.0.db
# and medreason_bench/data/training_reports/v0.0_haiku.json.
#
# Usage:
#     bash mvp_dashboard/run_eval_stages.sh
#
# Output:
#     - medreason_bench/leaderboard/entries/*__v0.0__dev.json (x2)
#     - medreason_bench/leaderboard/entries/*__v0.0__dev__cases.json (x2)
#     - mvp_dashboard/results.json
set -euo pipefail

cd "$(dirname "$0")/.."

echo "[1/3] zero-shot eval ..."
py -u -m medreason_bench eval \
    --model haiku \
    --no-memory \
    --version v0.0 \
    --split dev \
    --seeds 11

echo
echo "[2/3] memory-augmented eval ..."
py -u -m medreason_bench eval \
    --model haiku \
    --memory \
    --no-rerank \
    --version v0.0 \
    --split dev \
    --seeds 11

echo
echo "[3/3] build dashboard data ..."
py mvp_dashboard/build_results.py --version v0.0 --split dev --model haiku

echo
echo "Done. Refresh http://localhost:3000 to see the result."
