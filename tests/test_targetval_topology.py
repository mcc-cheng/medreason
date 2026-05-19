"""Tests for medreason.targetval.topology.PathwayCache + bypass signals."""

from __future__ import annotations

from medreason.targetval.topology import (
    BypassSignal,
    PathwayCache,
    derive_bypass_signals,
)


def _braf_cache() -> PathwayCache:
    return PathwayCache(
        paralogs_by_gene={"BRAF": ["ARAF", "RAF1"]},
        family_by_gene={"BRAF": "RAF_kinase"},
        feedback_loops_by_gene={"BRAF": ["MEK-ERK rebound"]},
        downstream_redundancy={"BRAF": 0.4},
    )


def test_paralog_lookup_returns_count_and_family():
    cache = _braf_cache()
    feat = cache.lookup_paralogs("BRAF")
    assert feat.count == 2
    assert feat.family == "RAF_kinase"
    assert set(feat.paralogs) == {"ARAF", "RAF1"}


def test_unknown_gene_returns_empty_paralog_feature():
    cache = _braf_cache()
    feat = cache.lookup_paralogs("UNKNOWN_GENE")
    assert feat.count == 0
    assert feat.family is None


def test_derive_bypass_signals_emits_three_for_braf():
    cache = _braf_cache()
    signals = derive_bypass_signals("BRAF", cache)
    mechs = {s.mechanism for s in signals}
    assert "paralog_compensation" in mechs
    assert "downstream_feedback" in mechs
    assert "alternative_pathway" in mechs


def test_paralog_strength_caps_at_threshold():
    cache = PathwayCache(
        paralogs_by_gene={"X": ["A", "B", "C", "D", "E", "F", "G"]},  # 7 paralogs
        family_by_gene={"X": "fake_family"},
    )
    signals = derive_bypass_signals("X", cache)
    paralog = [s for s in signals if s.mechanism == "paralog_compensation"][0]
    assert paralog.strength <= 0.75  # capped


def test_no_signals_for_isolated_target():
    cache = PathwayCache()
    signals = derive_bypass_signals("LONELY_GENE", cache)
    assert signals == []
