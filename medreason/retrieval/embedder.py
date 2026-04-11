"""Embedder Protocol + real OpenAI implementation + deterministic fake.

Design constraints:

1. **No deterministic-hash fallback.** The pre-rework store.py silently
   fell back to a hash-based pseudo-embedding when OPENAI_API_KEY was
   missing. The retrieved "top 3 by cosine similarity" ranking was
   essentially noise. This module replaces that with an explicit
   failure mode: the real OpenAIEmbedder raises RuntimeError on
   instantiation if no key is present. Tests use FakeEmbedder
   explicitly; nothing silently degrades.

2. **FakeEmbedder has structure.** A hash-only fake would be useless
   for testing the retrieval pipeline. FakeEmbedder instead uses a
   bag-of-words projection into a fixed low-dim space, so texts that
   share vocabulary cluster and texts that don't stay apart. This is
   enough to write meaningful tier 2 tests.

3. **Dimension is an attribute, not a constructor arg.** Each embedder
   implementation has a native dimension (OpenAI text-embedding-3-large
   is 3072; FakeEmbedder is 64). Code that caches embeddings on a rule
   must also store the embedder model id so re-embedding can be
   triggered on model change.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Optional, Protocol, runtime_checkable

from ..config import OPENAI_API_KEY


# ── Protocol ────────────────────────────────────────────────────────────────


@runtime_checkable
class Embedder(Protocol):
    model_id: str
    dimension: int

    def embed(self, text: str) -> list[float]:
        """Return the embedding for `text`. Implementations MUST return
        a list of exactly `self.dimension` floats. Empty input should
        return an all-zero vector."""
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch. Default implementation calls embed() per text;
        concrete implementations should override for batching efficiency."""
        ...


# ── OpenAI real implementation ──────────────────────────────────────────────


OPENAI_EMBED_MODELS: dict[str, int] = {
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
    "text-embedding-ada-002": 1536,
}

DEFAULT_OPENAI_EMBED_MODEL = "text-embedding-3-large"


class OpenAIEmbedder:
    model_id: str
    dimension: int

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        model: str = DEFAULT_OPENAI_EMBED_MODEL,
    ):
        if model not in OPENAI_EMBED_MODELS:
            raise ValueError(
                f"Unknown OpenAI embedding model {model!r}. Known: "
                f"{sorted(OPENAI_EMBED_MODELS.keys())}"
            )
        self._api_key = api_key if api_key is not None else OPENAI_API_KEY
        self.model_id = model
        self.dimension = OPENAI_EMBED_MODELS[model]
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise RuntimeError(
                "OpenAIEmbedder requires OPENAI_API_KEY to be set, an "
                "api_key argument, or a test client injected into "
                "._client. There is NO deterministic-hash fallback — use "
                "FakeEmbedder for tests."
            )
        from openai import OpenAI  # noqa: PLC0415
        self._client = OpenAI(api_key=self._api_key)
        return self._client

    def embed(self, text: str) -> list[float]:
        if not text:
            return [0.0] * self.dimension
        client = self._get_client()
        resp = client.embeddings.create(model=self.model_id, input=text)
        vec = list(resp.data[0].embedding)
        if len(vec) != self.dimension:
            raise RuntimeError(
                f"OpenAI returned embedding of length {len(vec)}, expected "
                f"{self.dimension} for model {self.model_id}"
            )
        return vec

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        client = self._get_client()
        # OpenAI accepts a list of strings and returns the embeddings
        # in order. Empty strings must be handled locally.
        cleaned = [t if t else " " for t in texts]
        resp = client.embeddings.create(model=self.model_id, input=cleaned)
        return [list(item.embedding) for item in resp.data]


# ── FakeEmbedder (test helper with semantic structure) ─────────────────────


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


class FakeEmbedder:
    """Deterministic embedder with enough semantic structure for tier 2
    tests to be meaningful.

    Each input text is tokenized (alphanumeric words), lowercased, and
    each token is hashed into one of `dimension` buckets. The final
    vector is the L2-normalized bucket count vector. Two texts that
    share vocabulary get high cosine similarity; two texts with
    disjoint vocabulary get zero similarity.

    This is NOT a good embedder for production — it's a test fixture
    that produces ranking behavior realistic enough to validate the
    dense-retrieval tier.
    """
    model_id: str = "fake-embedder-v0"
    dimension: int

    def __init__(self, *, dimension: int = 64):
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dimension
        if not text:
            return vec
        for tok in _TOKEN_RE.findall(text):
            bucket = self._bucket_for(tok.lower())
            vec[bucket] += 1.0
        # L2 normalize
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    def _bucket_for(self, token: str) -> int:
        h = hashlib.sha1(token.encode("utf-8")).digest()
        return int.from_bytes(h[:4], "big") % self.dimension


# ── Cosine similarity helper ────────────────────────────────────────────────


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Returns 0.0 if either vector has zero norm. Asserts equal length."""
    if len(a) != len(b):
        raise ValueError(f"vector length mismatch: {len(a)} vs {len(b)}")
    if not a:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))
