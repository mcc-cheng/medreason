"""Tests for the target-validation prediction card."""

from __future__ import annotations

import json
from pathlib import Path

from medreason_bench.targetval.prediction_card import (
    PREDICTION_CARD_VERSION,
    TargetValPredictionCard,
    TargetValPredictionEnvelope,
    default_v0_1_card,
    write_prediction_card,
)


def test_default_card_has_all_required_fields():
    card = default_v0_1_card()
    assert card.top1_lift_min_pp > 0
    assert card.top1_lift_ci_lo_min_pp >= 0
    assert card.bypass_recall_lift_min_pp > 0
    assert card.n_min_eligible >= 3
    assert len(card.seeds) >= 3
    assert card.version == PREDICTION_CARD_VERSION


def test_card_to_json_is_deterministic():
    card = default_v0_1_card()
    a = card.to_json()
    b = card.to_json()
    assert a == b
    # And is parseable
    data = json.loads(a)
    assert data["version"] == PREDICTION_CARD_VERSION


def test_envelope_sha256_matches_card():
    import hashlib

    card = default_v0_1_card()
    env = TargetValPredictionEnvelope.from_card(card)
    expected = hashlib.sha256(card.to_json().encode("utf-8")).hexdigest()
    assert env.sha256 == expected


def test_write_prediction_card_creates_file(tmp_path: Path):
    card = default_v0_1_card()
    path = write_prediction_card(card, tmp_path)
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["n_min_eligible"] == card.n_min_eligible
