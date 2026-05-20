"""Tests for medreason_bench.splits — stratify + manifest + leak guard
integration. Phase 3 done-when: a rule whose provenance references a
test case_id drawn from the written manifest must be refused by a
LeakGuard loaded from that manifest's fingerprints.json.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path

import pytest

from medreason.ontology import (
    BenchmarkCase, Difficulty, Outcome, RuleEvidence, RuleTrigger,
    ReasoningRule,
)
from medreason_bench.data import parse_lcd_xml
from medreason_bench.data.case_builder import build_cases_from_lcd
from medreason_bench.splits import (
    MANIFEST_SPLITS,
    ManifestError,
    SplitRatios,
    canonical_case_fingerprint,
    load_split,
    stratify,
    verify_manifest,
    write_manifest,
)


FIXTURE = Path(__file__).parent.parent / "medreason_bench" / "data" / "fixtures" / "sample_lcd.xml"


@pytest.fixture(scope="module")
def cases() -> list[BenchmarkCase]:
    return build_cases_from_lcd(
        parse_lcd_xml(FIXTURE), target_count=50, seed=42
    )


# ── SplitRatios ──────────────────────────────────────────────────────────────


def test_split_ratios_default_sum_to_one():
    r = SplitRatios()
    assert r.as_tuple() == (0.6, 0.2, 0.2)
    assert sum(r.as_tuple()) == pytest.approx(1.0)


def test_split_ratios_rejects_non_unit_sum():
    with pytest.raises(ValueError):
        SplitRatios(train=0.5, dev=0.2, test=0.2)


def test_split_ratios_rejects_negative():
    with pytest.raises(ValueError):
        SplitRatios(train=1.2, dev=-0.1, test=-0.1)


# ── stratify ────────────────────────────────────────────────────────────────


def test_stratify_preserves_total_count(cases):
    splits = stratify(cases, seed=42)
    total = sum(len(v) for v in splits.values())
    assert total == len(cases)
    assert set(splits.keys()) == set(MANIFEST_SPLITS)


def test_stratify_is_deterministic(cases):
    a = stratify(cases, seed=42)
    b = stratify(cases, seed=42)
    for split in MANIFEST_SPLITS:
        assert [c.case_id for c in a[split]] == [c.case_id for c in b[split]]


def test_stratify_no_case_appears_in_two_splits(cases):
    splits = stratify(cases, seed=42)
    all_ids: list[str] = []
    for split in MANIFEST_SPLITS:
        all_ids.extend(c.case_id for c in splits[split])
    assert len(all_ids) == len(set(all_ids))


def test_stratify_rough_ratios(cases):
    splits = stratify(cases, seed=42)
    n = len(cases)
    assert abs(len(splits["train"]) - 0.6 * n) <= 3
    assert abs(len(splits["dev"]) - 0.2 * n) <= 3
    assert abs(len(splits["test"]) - 0.2 * n) <= 3


def test_stratify_every_outcome_appears_in_every_split(cases):
    splits = stratify(cases, seed=42)
    for split in MANIFEST_SPLITS:
        outcomes = {c.ground_truth_outcome for c in splits[split]}
        # With 50 cases this should hold; if it ever fails we need to
        # raise the stratification resolution.
        assert len(outcomes) >= 2


def test_stratify_sparse_stratum_falls_into_train(cases):
    """A stratum with fewer than 3 cases should dump entirely into
    train so eval splits aren't starved."""
    # Take only 2 cases: both hit the same (outcome, difficulty) cell.
    tiny = [
        c for c in cases
        if c.ground_truth_outcome == Outcome.OVERTURNED_ON_APPEAL
        and c.difficulty == Difficulty.HARD
    ][:2]
    splits = stratify(tiny, seed=42)
    assert len(splits["train"]) == 2
    assert len(splits["dev"]) == 0
    assert len(splits["test"]) == 0


def test_stratify_output_sorted_by_case_id(cases):
    splits = stratify(cases, seed=42)
    for split in MANIFEST_SPLITS:
        ids = [c.case_id for c in splits[split]]
        assert ids == sorted(ids)


# ── Manifest writer / verifier ──────────────────────────────────────────────


def test_canonical_fingerprint_is_stable(cases):
    case = cases[0]
    h1 = canonical_case_fingerprint(case)
    h2 = canonical_case_fingerprint(case)
    assert h1 == h2
    assert len(h1) == 64


def test_write_manifest_creates_all_files(cases, tmp_path):
    splits = stratify(cases, seed=42)
    out = tmp_path / "v_test"
    fps = write_manifest(splits, out)
    for name in MANIFEST_SPLITS:
        assert (out / f"{name}.jsonl").exists()
    assert (out / "fingerprints.json").exists()
    total = sum(len(v) for v in fps.values())
    assert total == len(cases)


def test_write_manifest_is_byte_stable(cases, tmp_path):
    splits = stratify(cases, seed=42)
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    write_manifest(splits, out_a)
    write_manifest(splits, out_b)
    for fname in ("train.jsonl", "dev.jsonl", "test.jsonl", "fingerprints.json"):
        assert (out_a / fname).read_bytes() == (out_b / fname).read_bytes(), (
            f"{fname} differs across runs — manifest is not byte-stable"
        )


def test_write_manifest_rejects_unknown_split_name(cases, tmp_path):
    with pytest.raises(ManifestError):
        write_manifest({"train": cases, "bogus": []}, tmp_path / "x")


def test_write_manifest_rejects_case_in_two_splits(cases, tmp_path):
    c0 = cases[0]
    with pytest.raises(ManifestError) as exc:
        write_manifest({"train": [c0], "test": [c0]}, tmp_path / "x")
    assert c0.case_id in str(exc.value)


def test_verify_manifest_round_trip(cases, tmp_path):
    splits = stratify(cases, seed=42)
    out = tmp_path / "rt"
    write_manifest(splits, out)
    verified = verify_manifest(out)
    for name in MANIFEST_SPLITS:
        assert set(verified[name].keys()) == {c.case_id for c in splits[name]}


def test_verify_manifest_detects_drift(cases, tmp_path):
    splits = stratify(cases, seed=42)
    out = tmp_path / "drift"
    write_manifest(splits, out)

    # Corrupt one line of test.jsonl
    test_path = out / "test.jsonl"
    lines = test_path.read_bytes().split(b"\n")
    if lines and lines[0]:
        # Make a subtle modification inside a field that still yields
        # valid JSON but different bytes.
        lines[0] = lines[0].replace(b'"approved"', b'"denied"', 1)
    test_path.write_bytes(b"\n".join(lines))

    with pytest.raises(ManifestError) as exc:
        verify_manifest(out)
    assert "drift" in str(exc.value).lower() or "missing" in str(exc.value).lower()


def test_verify_manifest_missing_fingerprints_raises(tmp_path):
    with pytest.raises(ManifestError):
        verify_manifest(tmp_path)


def test_load_split_round_trip(cases, tmp_path):
    splits = stratify(cases, seed=42)
    out = tmp_path / "load"
    write_manifest(splits, out)
    loaded = load_split(out, "test")
    assert len(loaded) == len(splits["test"])
    assert {c.case_id for c in loaded} == {c.case_id for c in splits["test"]}
    # Round-tripped cases must be identical to the source
    by_id = {c.case_id: c for c in splits["test"]}
    for lc in loaded:
        assert lc.model_dump() == by_id[lc.case_id].model_dump()


def test_load_split_rejects_unknown_split_name(tmp_path):
    with pytest.raises(ManifestError):
        load_split(tmp_path, "bogus")


# ── THE Phase 3 done-when test: LeakGuard + RuleStore end-to-end ────────────


def test_leak_guard_loaded_from_written_manifest_refuses_test_case_rule(
    cases, tmp_path
):
    """This is the Phase 3 gate: build → write → load → LeakGuard must
    refuse any rule whose provenance touches a case_id from the test split.
    """
    from medreason.store import LeakGuard, RuleStore, TestSetLeakError

    splits = stratify(cases, seed=42)
    out = tmp_path / "v_phase3"
    write_manifest(splits, out)

    lg = LeakGuard.from_fingerprint_file(out / "fingerprints.json")
    assert len(lg.test_case_ids) == len(splits["test"])
    assert len(lg.train_case_ids) == len(splits["train"])
    assert len(lg.dev_case_ids) == len(splits["dev"])

    # Grab a real test case_id from the written manifest
    test_case_id = next(iter(lg.test_case_ids))

    conn = sqlite3.connect(":memory:")
    store = RuleStore(conn, leak_guard=lg)

    bad_rule = ReasoningRule(
        trigger=RuleTrigger(),
        action="bogus",
        evidence=RuleEvidence(
            supporting_case_ids=[test_case_id],
            source_policy_citation="test",
        ),
    )
    with pytest.raises(TestSetLeakError) as exc:
        store.put(bad_rule)
    assert test_case_id in str(exc.value)
    # Rule must NOT have been persisted
    assert store.count() == 0

    # Train case_ids must go through cleanly
    train_case_id = next(iter(lg.train_case_ids))
    good_rule = ReasoningRule(
        trigger=RuleTrigger(),
        action="legitimate",
        evidence=RuleEvidence(
            supporting_case_ids=[train_case_id],
            source_policy_citation="test",
        ),
    )
    store.put(good_rule)
    assert store.count() == 1


def test_fingerprints_json_schema_matches_leak_guard_expectations(cases, tmp_path):
    splits = stratify(cases, seed=42)
    out = tmp_path / "schema"
    write_manifest(splits, out)
    fp_file = out / "fingerprints.json"
    data = json.loads(fp_file.read_text())
    # Top-level keys must be exactly the three split names
    assert set(data.keys()) == set(MANIFEST_SPLITS)
    # Each split must be a {case_id: sha256hex} mapping
    for split in MANIFEST_SPLITS:
        for case_id, digest in data[split].items():
            assert isinstance(case_id, str) and case_id.startswith("case_")
            assert isinstance(digest, str) and len(digest) == 64
            int(digest, 16)  # must parse as hex
