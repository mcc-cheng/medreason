"""Tests for medreason.retrieval.embedder — Phase 5 Commit 2."""

from __future__ import annotations

import math

import pytest

from medreason.retrieval import (
    DEFAULT_OPENAI_EMBED_MODEL,
    Embedder,
    FakeEmbedder,
    OPENAI_EMBED_MODELS,
    OpenAIEmbedder,
    cosine_similarity,
)


# ── FakeEmbedder ─────────────────────────────────────────────────────────────


def test_fake_embedder_protocol_conformance():
    e = FakeEmbedder()
    assert isinstance(e, Embedder)
    assert e.dimension == 64
    assert e.model_id == "fake-embedder-v0"


def test_fake_embedder_custom_dimension():
    e = FakeEmbedder(dimension=128)
    assert e.dimension == 128
    v = e.embed("hello")
    assert len(v) == 128


def test_fake_embedder_rejects_nonpositive_dimension():
    with pytest.raises(ValueError):
        FakeEmbedder(dimension=0)
    with pytest.raises(ValueError):
        FakeEmbedder(dimension=-5)


def test_fake_embedder_empty_string_is_zero_vector():
    e = FakeEmbedder()
    v = e.embed("")
    assert len(v) == e.dimension
    assert all(x == 0.0 for x in v)


def test_fake_embedder_deterministic():
    e = FakeEmbedder()
    v1 = e.embed("lumbar MRI conservative therapy")
    v2 = e.embed("lumbar MRI conservative therapy")
    assert v1 == v2


def test_fake_embedder_l2_normalized():
    e = FakeEmbedder()
    v = e.embed("some test tokens here and more")
    norm = math.sqrt(sum(x * x for x in v))
    assert norm == pytest.approx(1.0, abs=1e-6)


def test_fake_embedder_shared_vocabulary_clusters():
    """Two texts that share vocabulary should have non-zero cosine.
    Two texts with disjoint vocabulary should have (approximately)
    zero cosine."""
    e = FakeEmbedder()
    a = e.embed("lumbar MRI conservative therapy physical")
    b = e.embed("lumbar MRI conservative therapy exam")  # 4 shared
    c = e.embed("cardiac catheterization angiography contrast")  # 0 shared

    ab = cosine_similarity(a, b)
    ac = cosine_similarity(a, c)
    assert ab > 0.5
    assert ac == pytest.approx(0.0, abs=1e-6) or ac < 0.05


def test_fake_embedder_batch_matches_single():
    e = FakeEmbedder()
    texts = ["hello world", "foo bar baz"]
    batch = e.embed_batch(texts)
    single = [e.embed(t) for t in texts]
    assert batch == single


def test_fake_embedder_tokenization_is_case_insensitive():
    e = FakeEmbedder()
    v1 = e.embed("Lumbar Mri")
    v2 = e.embed("LUMBAR mri")
    assert v1 == v2


# ── OpenAIEmbedder ──────────────────────────────────────────────────────────


def test_openai_embedder_known_model_dimensions():
    for model, dim in OPENAI_EMBED_MODELS.items():
        e = OpenAIEmbedder(api_key="fake", model=model)
        assert e.model_id == model
        assert e.dimension == dim


def test_openai_embedder_rejects_unknown_model():
    with pytest.raises(ValueError):
        OpenAIEmbedder(api_key="fake", model="some-future-model")


def test_openai_embedder_without_api_key_raises_at_embed():
    """Constructing without a key is OK (for test injection); calling
    embed() without a key and without an injected client must raise
    loudly — NO deterministic-hash fallback."""
    e = OpenAIEmbedder(api_key="")
    assert e._client is None
    with pytest.raises(RuntimeError) as exc:
        e.embed("anything")
    msg = str(exc.value)
    assert "OPENAI_API_KEY" in msg or "api_key" in msg
    assert "FakeEmbedder" in msg  # the error should steer tests toward Fake


def test_openai_embedder_default_model_is_large():
    assert DEFAULT_OPENAI_EMBED_MODEL == "text-embedding-3-large"
    assert OPENAI_EMBED_MODELS[DEFAULT_OPENAI_EMBED_MODEL] == 3072


# ── cosine_similarity ───────────────────────────────────────────────────────


def test_cosine_identical_vectors_is_one():
    v = [1.0, 2.0, 3.0]
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_orthogonal_vectors_is_zero():
    a = [1.0, 0.0, 0.0]
    b = [0.0, 1.0, 0.0]
    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_opposite_vectors_is_minus_one():
    a = [1.0, 2.0, 3.0]
    b = [-1.0, -2.0, -3.0]
    assert cosine_similarity(a, b) == pytest.approx(-1.0)


def test_cosine_zero_norm_returns_zero():
    a = [0.0, 0.0, 0.0]
    b = [1.0, 2.0, 3.0]
    assert cosine_similarity(a, b) == 0.0
    assert cosine_similarity(b, a) == 0.0


def test_cosine_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])
