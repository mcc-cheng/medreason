"""Pre-registered prediction card for the target-validation retro.

Mirrors medreason_bench/leadop/prediction_card.py: a JSON-serialisable
envelope that commits to specific lift intervals BEFORE the data arrives,
stamped via OpenTimestamps so the predictions can't be revised post-hoc.

The card commits to:

- top1_lift_pp: minimum percentage-point improvement of
  (swarm + universal-layer memory) over (swarm alone) on top-K target
  hit, with a 90% CI lower bound strictly above 0.
- bypass_recall_lift_pp: minimum percentage-point improvement on
  bypass-mechanism recall (the metric that captures the "metacognitive
  memory catches systematic miss" claim).
- n_min_eligible: minimum number of eligible cases the gate is
  evaluated on (avoids a 3-case lucky bootstrap).

Status: SKELETON. The TargetValPredictionCard dataclass is final; the
ots_stamp / render_pdf helpers are deferred to follow-up commits — for
v0.1 we just write the values JSON and pin it via git commit.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


PREDICTION_CARD_VERSION = "targetval-v0.1"


@dataclass(frozen=True)
class TargetValPredictionCard:
    """The commitments. All values in percentage points; n_min in cases.

    Attributes
    ----------
    top1_lift_min_pp
        Minimum mean lift on top-K target hit. Below this, the run is
        considered NEGATIVE evidence for the mechanism.
    top1_lift_ci_lo_min_pp
        Minimum 90% CI lower bound on the lift. Required > 0 for a
        successful pre-registration.
    bypass_recall_lift_min_pp
        Minimum mean lift on bypass-mechanism recall.
    n_min_eligible
        Minimum number of retro cases required to score the run.
    seeds
        The list of seeds the retro will run with.
    k_for_top_k
        The "K" in top-K target hit.
    """

    top1_lift_min_pp: float
    top1_lift_ci_lo_min_pp: float
    bypass_recall_lift_min_pp: float
    n_min_eligible: int
    seeds: tuple[int, ...]
    k_for_top_k: int = 5
    version: str = PREDICTION_CARD_VERSION

    def to_json(self) -> str:
        d = asdict(self)
        d["seeds"] = list(d["seeds"])
        return json.dumps(d, indent=2, sort_keys=True)


@dataclass(frozen=True)
class TargetValPredictionEnvelope:
    """The full envelope ledger pinned by OpenTimestamps. Includes the
    card + a SHA-256 over a deterministic JSON projection of it.
    """

    card: TargetValPredictionCard
    sha256: str = ""
    ots_proof_path: str = ""

    @classmethod
    def from_card(cls, card: TargetValPredictionCard) -> "TargetValPredictionEnvelope":
        import hashlib

        digest = hashlib.sha256(card.to_json().encode("utf-8")).hexdigest()
        return cls(card=card, sha256=digest)


def write_prediction_card(card: TargetValPredictionCard, out_dir: str | Path) -> Path:
    """Write the card JSON to disk. Caller is responsible for git-commit +
    OpenTimestamps stamping (same pattern as leadop/ots_stamp.py).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "prediction_card_targetval.values.json"
    path.write_text(card.to_json())
    return path


def default_v0_1_card() -> TargetValPredictionCard:
    """The v0.1 pre-registration values.

    These numbers are conservative placeholders — the user committed to
    actual values BEFORE running the MAPK retro. Replace the
    pre-registration with the user-chosen numbers in a separate commit
    that is then OpenTimestamps-stamped.
    """
    return TargetValPredictionCard(
        top1_lift_min_pp=8.0,
        top1_lift_ci_lo_min_pp=0.5,
        bypass_recall_lift_min_pp=10.0,
        n_min_eligible=15,
        seeds=(11, 17, 23, 31, 41, 53, 67),
        k_for_top_k=5,
    )
