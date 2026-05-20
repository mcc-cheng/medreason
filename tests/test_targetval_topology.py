"""Tests for medreason.targetval.topology.PathwayCache + bypass signals."""

from __future__ import annotations

from pathlib import Path

from medreason.targetval.case import BypassMechanism, Modality
from medreason.targetval.topology import (
    BypassSignal,
    PathwayCache,
    derive_bypass_signals,
    dominant_bypass_mechanism,
)


def _braf_cache() -> PathwayCache:
    return PathwayCache(
        paralogs_by_gene={"BRAF": ["ARAF", "RAF1"]},
        family_by_gene={"BRAF": "RAF_kinase"},
        feedback_loops_by_gene={"BRAF": ["MEK-ERK rebound"]},
        downstream_redundancy={"BRAF": 0.4},
    )


def _mapk_cache_v0_2() -> PathwayCache:
    """A slightly richer MAPK-shaped cache used by the new v0.2 tests.

    Captures the canonical RAF / MEK / ERK paralog families plus a known
    BRAF reactivation loop. Not the full ~30-node retro fixture
    (that's mapk-curator's scope); just enough to exercise the new
    contract changes.
    """
    return PathwayCache(
        paralogs_by_gene={
            "BRAF": ["ARAF", "RAF1"],
            "MAP2K1": ["MAP2K2"],
            "MAPK1": ["MAPK3"],
            "KRAS": ["NRAS", "HRAS"],
        },
        family_by_gene={
            "BRAF": "RAF_kinase",
            "ARAF": "RAF_kinase",
            "RAF1": "RAF_kinase",
            "MAP2K1": "MEK",
            "MAP2K2": "MEK",
            "MAPK1": "ERK",
            "MAPK3": "ERK",
            "KRAS": "RAS_GTPase",
            "NRAS": "RAS_GTPase",
            "HRAS": "RAS_GTPase",
        },
        feedback_loops_by_gene={
            "BRAF": ["MEK-ERK rebound", "RAS-GTP reactivation"],
            "MAP2K1": ["ERK negative feedback to RAF"],
        },
        downstream_redundancy={
            "BRAF": 0.4,
            "MAP2K1": 0.25,
            "KRAS": 0.6,
        },
        upstream_nodes_by_gene={
            "BRAF": ["KRAS", "NRAS", "HRAS"],
            "MAP2K1": ["BRAF", "ARAF", "RAF1"],
            "MAPK1": ["MAP2K1", "MAP2K2"],
        },
        downstream_nodes_by_gene={
            "BRAF": ["MAP2K1", "MAP2K2"],
            "MAP2K1": ["MAPK1", "MAPK3"],
            "KRAS": ["BRAF", "ARAF", "RAF1", "PIK3CA"],
        },
        cache_version="mapk_v0.2_test",
    )


# ── Existing tests (must remain green) ───────────────────────────────────────


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
    # BRAF triggers the curated reactivation loop (higher-prior variant of
    # downstream_feedback). The legacy test only required "any feedback
    # signal" — feedback_reactivation satisfies that contract.
    assert "feedback_reactivation" in mechs or "downstream_feedback" in mechs
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


# ── New v0.2 tests ───────────────────────────────────────────────────────────


def test_modality_filter_drops_feedback_for_antibody():
    """Antibodies can't engage intracellular RAF feedback rebound."""
    cache = _mapk_cache_v0_2()
    signals = derive_bypass_signals("BRAF", cache, modality=Modality.ANTIBODY)
    mechs = {s.mechanism for s in signals}
    assert "downstream_feedback" not in mechs
    assert "feedback_reactivation" not in mechs
    # Paralog + alternative-pathway should still come through.
    assert "paralog_compensation" in mechs
    assert "alternative_pathway" in mechs


def test_modality_filter_default_none_preserves_all_signals():
    """modality=None must reproduce the pre-v0.2 behaviour."""
    cache = _mapk_cache_v0_2()
    signals = derive_bypass_signals("BRAF", cache, modality=None)
    mechs = {s.mechanism for s in signals}
    assert "paralog_compensation" in mechs
    assert "feedback_reactivation" in mechs  # BRAF triggers the reactivation variant
    assert "alternative_pathway" in mechs


def test_modality_filter_small_molecule_allows_feedback():
    cache = _mapk_cache_v0_2()
    signals = derive_bypass_signals("BRAF", cache, modality=Modality.SMALL_MOLECULE)
    mechs = {s.mechanism for s in signals}
    assert "feedback_reactivation" in mechs


def test_feedback_reactivation_fires_for_known_mapk_target():
    """BRAF / RAF1 carry the curated reactivation loop, higher prior."""
    cache = _mapk_cache_v0_2()
    signals = derive_bypass_signals("BRAF", cache)
    reactivation = [s for s in signals if s.mechanism == "feedback_reactivation"]
    assert len(reactivation) == 1
    assert reactivation[0].strength > 0.5  # higher than generic feedback


def test_feedback_generic_for_non_reactivation_gene():
    """A non-RAF target with a feedback loop gets the generic mechanism."""
    cache = _mapk_cache_v0_2()
    signals = derive_bypass_signals("MAP2K1", cache)
    mechs = {s.mechanism for s in signals}
    assert "downstream_feedback" in mechs
    assert "feedback_reactivation" not in mechs


def test_bypass_signal_contributing_features_populated():
    cache = _braf_cache()
    signals = derive_bypass_signals("BRAF", cache)
    paralog = [s for s in signals if s.mechanism == "paralog_compensation"][0]
    assert "paralog_count=2" in paralog.contributing_features
    assert any("paralog_family=" in f for f in paralog.contributing_features)


def test_pathway_cache_roundtrip_json():
    cache = _mapk_cache_v0_2()
    payload = cache.to_json()
    restored = PathwayCache.from_json(payload)
    assert restored == cache
    # Snapshot must be deterministic — same payload twice byte-for-byte.
    assert restored.to_json() == payload


def test_pathway_cache_file_roundtrip(tmp_path: Path):
    cache = _mapk_cache_v0_2()
    snapshot_path = tmp_path / "snap" / "cache.json"
    written = cache.write(snapshot_path)
    assert written == snapshot_path
    assert snapshot_path.exists()
    restored = PathwayCache.load(snapshot_path)
    assert restored == cache
    assert restored.cache_version == "mapk_v0.2_test"


def test_pathway_cache_from_json_rejects_unknown_field():
    bad = '{"paralogs_by_gene": {}, "unexpected_field": 1}'
    try:
        PathwayCache.from_json(bad)
    except ValueError as exc:
        assert "unexpected_field" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown field")


def test_dominant_bypass_mechanism_empty_returns_no_bypass():
    assert dominant_bypass_mechanism([]) is BypassMechanism.NO_BYPASS_KNOWN


def test_dominant_bypass_mechanism_all_zero_returns_no_bypass():
    signals = [
        BypassSignal(mechanism="paralog_compensation", strength=0.0, evidence_summary=""),
    ]
    assert dominant_bypass_mechanism(signals) is BypassMechanism.NO_BYPASS_KNOWN


def test_dominant_bypass_mechanism_picks_highest_strength():
    signals = [
        BypassSignal(mechanism="paralog_compensation", strength=0.3, evidence_summary=""),
        BypassSignal(mechanism="alternative_pathway", strength=0.8, evidence_summary=""),
        BypassSignal(mechanism="downstream_feedback", strength=0.5, evidence_summary=""),
    ]
    assert dominant_bypass_mechanism(signals) is BypassMechanism.ALTERNATIVE_PATHWAY


def test_dominant_bypass_mechanism_collapses_reactivation_into_feedback():
    """feedback_reactivation maps onto DOWNSTREAM_FEEDBACK in the public enum."""
    signals = [
        BypassSignal(mechanism="feedback_reactivation", strength=0.7, evidence_summary=""),
    ]
    assert dominant_bypass_mechanism(signals) is BypassMechanism.DOWNSTREAM_FEEDBACK


def test_pathway_cache_new_lookup_helpers():
    cache = _mapk_cache_v0_2()
    assert cache.lookup_upstream_nodes("BRAF") == ["KRAS", "NRAS", "HRAS"]
    assert cache.lookup_downstream_nodes("BRAF") == ["MAP2K1", "MAP2K2"]
    assert cache.lookup_upstream_nodes("UNKNOWN") == []
