"""Render the Veridicus prediction card PDF from the v3 dry-run report.

Usage:
    py mvp_dashboard/render_prediction_card.py

Output: deliverables/prediction_card_veridicus.pdf
        deliverables/prediction_card_veridicus.values.json
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from medreason_bench.leadop.prediction_card import derive_from_dryrun, render_pdf


def _git_commit_hash() -> str | None:
    try:
        h = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        return h if h else None
    except Exception:
        return None


def main() -> int:
    report = ROOT / "mvp_dashboard" / "dryrun_crizotinib_llm_v3.json"
    out_dir = ROOT / "deliverables"
    out_dir.mkdir(exist_ok=True)
    pdf_path = out_dir / "prediction_card_veridicus.pdf"
    json_path = out_dir / "prediction_card_veridicus.values.json"

    values = derive_from_dryrun(report)
    json_path.write_text(json.dumps(values.to_dict(), indent=2))
    render_pdf(values, pdf_path, git_commit_hash=_git_commit_hash())

    print(f"rendered {pdf_path}")
    print(f"values -> {json_path}")
    print(f"top-1 delta point: {values.delta_point_pp:+.2f}pp  interval [{values.delta_lo_pp:+.2f}, {values.delta_hi_pp:+.2f}]pp")
    print(f"cycle-waste delta: {values.cw_point_pp:+.2f}pp  interval [{values.cw_lo_pp:+.2f}, {values.cw_hi_pp:+.2f}]pp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
