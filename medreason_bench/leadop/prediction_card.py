"""Pre-registered prediction card derivation + single-page PDF renderer.

Derivation per approved design doc (krish-master-design-20260414,
lines 270-288):

  Delta_pred = alpha * Delta_chembl + (1 - alpha) * (Delta_medauth * tau)

  - alpha in [0.5, 0.8]  — weight toward in-domain (ChEMBL) evidence
  - tau   in [0.3, 0.7]  — cross-domain transfer degradation factor on
                           the medical-auth prior
  - Delta_medauth = 4.9pp (v0.5: 93.5% -> 98.4%)
  - Delta_chembl  = observed ChEMBL dry-run top-1 lift
  - Point estimate uses alpha=0.7, tau=0.5
  - Interval: sweep (alpha, tau) across the corners, take min/max, cap
    the full interval width at 15pp (+/- 7.5pp around the point)

Predicted baseline B on Recursion data is the ChEMBL dry-run baseline
unchanged (no prior to combine).

Cycle-waste delta is reported with the same interval construction; it
is supporting color, not headline (reviewer concern 5).
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path


MEDAUTH_DELTA_PP = 4.9  # v0.5: 93.5% -> 98.4%

ALPHA_POINT = 0.7
TAU_POINT = 0.5
ALPHA_RANGE = (0.5, 0.8)
TAU_RANGE = (0.3, 0.7)
INTERVAL_CAP_PP = 15.0  # full interval width cap -> +/- 7.5pp


def _predict_delta(
    delta_chembl_pp: float,
    *,
    alpha: float,
    tau: float,
) -> float:
    return alpha * delta_chembl_pp + (1 - alpha) * MEDAUTH_DELTA_PP * tau


def _cap_interval(point: float, lo: float, hi: float) -> tuple[float, float]:
    width = hi - lo
    if width <= INTERVAL_CAP_PP:
        return lo, hi
    half = INTERVAL_CAP_PP / 2
    return point - half, point + half


@dataclass
class PredictionCardValues:
    delta_chembl_pp: float
    delta_chembl_ci_lo_pp: float
    delta_chembl_ci_hi_pp: float
    baseline_rate_pct: float
    memory_rate_pct_point: float
    delta_point_pp: float
    delta_lo_pp: float
    delta_hi_pp: float
    cw_chembl_pp: float
    cw_point_pp: float
    cw_lo_pp: float
    cw_hi_pp: float
    n_decision_points: int
    n_seeds: int
    alpha_point: float
    tau_point: float
    medauth_delta_pp: float
    gate_passed: bool
    gate_reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def derive_from_dryrun(report_path: str | Path) -> PredictionCardValues:
    report = json.loads(Path(report_path).read_text())
    delta_chembl = report["top1_lift_pp"]
    ci_lo = report["top1_lift_ci_90_lo"]
    ci_hi = report["top1_lift_ci_90_hi"]
    memoff = report["top1_memoff_rates"]
    memon = report["top1_memon_rates"]
    baseline_pct = 100 * statistics.mean(memoff) if memoff else 0.0
    memory_pct_point = baseline_pct + _predict_delta(
        delta_chembl, alpha=ALPHA_POINT, tau=TAU_POINT
    )
    point = _predict_delta(delta_chembl, alpha=ALPHA_POINT, tau=TAU_POINT)

    corner_values = [
        _predict_delta(delta_chembl, alpha=a, tau=t)
        for a in ALPHA_RANGE
        for t in TAU_RANGE
    ]
    lo_raw = min(corner_values)
    hi_raw = max(corner_values)
    lo, hi = _cap_interval(point, lo_raw, hi_raw)

    cw_chembl = report.get("cw_lift_pp", 0.0)
    cw_point = _predict_delta(cw_chembl, alpha=ALPHA_POINT, tau=TAU_POINT)
    cw_corners = [
        _predict_delta(cw_chembl, alpha=a, tau=t)
        for a in ALPHA_RANGE
        for t in TAU_RANGE
    ]
    cw_lo_raw = min(cw_corners)
    cw_hi_raw = max(cw_corners)
    cw_lo, cw_hi = _cap_interval(cw_point, cw_lo_raw, cw_hi_raw)

    return PredictionCardValues(
        delta_chembl_pp=delta_chembl,
        delta_chembl_ci_lo_pp=ci_lo,
        delta_chembl_ci_hi_pp=ci_hi,
        baseline_rate_pct=baseline_pct,
        memory_rate_pct_point=memory_pct_point,
        delta_point_pp=point,
        delta_lo_pp=lo,
        delta_hi_pp=hi,
        cw_chembl_pp=cw_chembl,
        cw_point_pp=cw_point,
        cw_lo_pp=cw_lo,
        cw_hi_pp=cw_hi,
        n_decision_points=report["n_decision_points"],
        n_seeds=len(report["seeds"]),
        alpha_point=ALPHA_POINT,
        tau_point=TAU_POINT,
        medauth_delta_pp=MEDAUTH_DELTA_PP,
        gate_passed=report["gate_passed"],
        gate_reason=report["gate_reason"],
    )


def render_pdf(
    values: PredictionCardValues,
    out_path: str | Path,
    *,
    git_commit_hash: str | None = None,
    scientist_name: str = "Staff Scientist, Recursion Pharmaceuticals",
    campaign_label: str = "Campaign X (neurodrug program)",
) -> Path:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
    from reportlab.lib import colors

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=letter,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        title="Veridicus Pre-Registered Prediction Card",
    )
    styles = getSampleStyleSheet()
    h1 = styles["Heading1"]
    h1.fontSize = 14
    h2 = styles["Heading2"]
    h2.fontSize = 11
    body = ParagraphStyle(
        "body", parent=styles["BodyText"], fontSize=9, leading=12
    )
    small = ParagraphStyle(
        "small", parent=styles["BodyText"], fontSize=8, leading=10,
        textColor=colors.HexColor("#444444"),
    )
    mono = ParagraphStyle(
        "mono", parent=styles["BodyText"], fontName="Courier",
        fontSize=8, leading=10,
    )

    flow: list = []
    flow.append(Paragraph("Veridicus Pre-Registered Prediction Card", h1))
    flow.append(Paragraph(
        f"For: {scientist_name}  |  Target: {campaign_label}", small
    ))
    flow.append(Spacer(1, 6))

    # Predicted numbers
    half = INTERVAL_CAP_PP / 2
    flow.append(Paragraph("Predicted lift (committed BEFORE customer data arrives):", h2))
    table_data = [
        ["Metric", "Baseline B", "Memory-augmented M", "Delta (M - B)"],
        [
            "Top-1 direction-hit rate",
            f"{values.baseline_rate_pct:.1f}%",
            f"{values.memory_rate_pct_point:.1f}% (point)",
            f"{values.delta_point_pp:+.1f} pp  "
            f"[{values.delta_lo_pp:+.1f}, {values.delta_hi_pp:+.1f}] pp",
        ],
        [
            "Cycle-waste delta (K=3)",
            "n/a",
            "n/a",
            f"{values.cw_point_pp:+.1f} pp  "
            f"[{values.cw_lo_pp:+.1f}, {values.cw_hi_pp:+.1f}] pp  (supporting)",
        ],
    ]
    table = Table(table_data, colWidths=[1.9 * inch, 1.2 * inch, 1.6 * inch, 2.3 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#222222")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    flow.append(table)
    flow.append(Spacer(1, 8))

    # Derivation
    flow.append(Paragraph("Derivation formula (verbatim):", h2))
    flow.append(Paragraph(
        "Delta_predicted = alpha * Delta_chembl + (1 - alpha) * (Delta_medauth * tau)",
        mono,
    ))
    flow.append(Paragraph(
        f"alpha in [{ALPHA_RANGE[0]:.1f}, {ALPHA_RANGE[1]:.1f}] "
        f"(weight toward in-domain ChEMBL evidence); point estimate alpha = {values.alpha_point:.2f}. "
        f"tau in [{TAU_RANGE[0]:.1f}, {TAU_RANGE[1]:.1f}] "
        f"(cross-domain transfer degradation on the medical-auth prior); point estimate tau = {values.tau_point:.2f}. "
        f"Delta_medauth = {values.medauth_delta_pp:.1f}pp (v0.5 medical-auth benchmark: 93.5% → 98.4% on 50 train → 148 held-out, 3 seeds ±0.4%). "
        f"Delta_chembl = {values.delta_chembl_pp:.1f}pp observed on a {values.n_decision_points}-DP ChEMBL-grounded dry-run "
        f"(scaffolding; not a customer-deliverable result); 90% bootstrap CI on the lift "
        f"[{values.delta_chembl_ci_lo_pp:+.1f}, {values.delta_chembl_ci_hi_pp:+.1f}]pp across {values.n_seeds} seeds. "
        f"Interval is constructed by sweeping alpha and tau across the four corner cases, taking min/max, "
        f"and capping the full interval width at {INTERVAL_CAP_PP:.0f}pp (+/- {half:.1f}pp) to remain "
        "falsifiable without being vacuous.",
        body,
    ))
    flow.append(Spacer(1, 6))

    # What counts as a successful prediction
    flow.append(Paragraph("Scoring criterion:", h2))
    flow.append(Paragraph(
        f"The prediction succeeds if, on your de-identified campaign data with "
        f"decision points marked, the observed memory-augmented top-1 direction-hit rate "
        f"minus baseline falls within [{values.delta_lo_pp:+.1f}, {values.delta_hi_pp:+.1f}] pp. "
        "Outside that interval in either direction = miss; the miss direction and "
        "magnitude are reported honestly. Cycle-waste delta is reported alongside "
        "but does not affect pass/fail.",
        body,
    ))
    flow.append(Spacer(1, 6))

    # Commitments
    flow.append(Paragraph("Pre-commitment mechanism:", h2))
    git_line = (
        f"git commit hash: {git_commit_hash}"
        if git_commit_hash
        else "git commit hash: (filled at commit time)"
    )
    flow.append(Paragraph(
        f"This PDF is committed to the Veridicus git repository. {git_line}. "
        "The commit is timestamped via OpenTimestamps (bitcoin-anchored public timestamp); "
        "the .ots receipt accompanying this PDF is the falsifiable pre-commitment artifact. "
        "If the .ots receipt is in 'pending' state at delivery time (OTS attestations take "
        "~1-6 hours to finalize), the finalized receipt will be forwarded post-hoc; "
        "the commit-hash alone is sufficient for third-party verification.",
        body,
    ))
    flow.append(Spacer(1, 6))

    # Scope
    flow.append(Paragraph("Scope & caveats:", h2))
    flow.append(Paragraph(
        "(1) The generalization claim rests on the metacognitive rule proposer path "
        "(v0.5 medical-auth 93.5% -> 98.4% on 148 held-out, 3 seeds ±0.4%). A separate "
        "failure-driven extraction path exists in the production system but is "
        "intentionally EXCLUDED from this retrospective because Experiment E "
        "(LOO-validated) showed it memorizes rather than generalizes. "
        "(2) The ChEMBL dry-run is scaffolding, not a published number; it validates "
        "the harness and the +3pp lift gate but does not constitute a drug-discovery "
        "generalization result. (3) Cross-domain transfer from medical-auth to pharma "
        "lead-op reasoning is an open empirical question — the tau parameter in the "
        "formula models this explicitly. (4) Top-1 direction-hit is the headline metric; "
        "cycle-waste delta is supporting and may be withdrawn if noisy on your data.",
        body,
    ))
    flow.append(Spacer(1, 6))

    # Gate reason footer
    flow.append(Paragraph(
        f"Harness gate check: {values.gate_reason}",
        small,
    ))

    doc.build(flow)
    return out_path
