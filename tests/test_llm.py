"""Tests for medreason.llm — Phase 5 Commit 1."""

from __future__ import annotations

import pytest


# ── FakeLLMClient behavior ──────────────────────────────────────────────────


def test_fake_llm_client_is_protocol_conformant():
    from medreason.llm import FakeLLMClient, LLMClient
    c = FakeLLMClient()
    assert isinstance(c, LLMClient)
    assert hasattr(c, "model_version")
    assert hasattr(c, "complete")


def test_fake_llm_client_returns_queued_responses_in_order():
    from medreason.llm import FakeLLMClient
    c = FakeLLMClient(responses=["first", "second", "third"])
    assert c.complete(system="s", user="u1").text == "first"
    assert c.complete(system="s", user="u2").text == "second"
    assert c.complete(system="s", user="u3").text == "third"


def test_fake_llm_client_falls_back_to_default_after_queue_drained():
    from medreason.llm import FakeLLMClient
    c = FakeLLMClient(responses=["one"], default_text="fallback")
    assert c.complete(system="s", user="u").text == "one"
    assert c.complete(system="s", user="u").text == "fallback"
    assert c.complete(system="s", user="u").text == "fallback"


def test_fake_llm_client_records_calls():
    from medreason.llm import FakeLLMClient
    c = FakeLLMClient()
    c.complete(system="sys1", user="usr1")
    c.complete(system="sys2", user="usr2")
    assert len(c.calls) == 2
    assert c.calls[0] == ("sys1", "usr1")
    assert c.calls[1] == ("sys2", "usr2")


def test_fake_llm_client_reports_usage_and_cost():
    from medreason.llm import FakeLLMClient
    c = FakeLLMClient(
        input_tokens_per_call=250,
        output_tokens_per_call=80,
        cost_per_call_usd=0.0012,
    )
    r = c.complete(system="s", user="u")
    assert r.input_tokens == 250
    assert r.output_tokens == 80
    assert r.cost_usd == pytest.approx(0.0012)
    assert r.model == "fake-llm-v0"


# ── ClaudeLLMClient with FakeClient (no network) ────────────────────────────


class _FakeUsage:
    def __init__(self, in_t, out_t):
        self.input_tokens = in_t
        self.output_tokens = out_t


class _FakeBlock:
    def __init__(self, text):
        self.text = text


class _FakeResp:
    def __init__(self, text, in_t, out_t):
        self.content = [_FakeBlock(text)]
        self.usage = _FakeUsage(in_t, out_t)


class _FakeClient:
    def __init__(self, resp):
        self._resp = resp
        self.messages = self
        self.last_kwargs = {}

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._resp


def test_claude_llm_client_happy_path():
    from medreason.llm import ClaudeLLMClient
    client = ClaudeLLMClient(api_key="fake")
    client._client = _FakeClient(_FakeResp("model output", 200, 50))
    r = client.complete(system="sys", user="usr", max_tokens=1024)
    assert r.text == "model output"
    assert r.input_tokens == 200
    assert r.output_tokens == 50
    assert r.model == "claude-sonnet-4-20250514"
    # Cost math: 200 * 3/1e6 + 50 * 15/1e6 = 0.0006 + 0.00075 = 0.00135
    assert r.cost_usd == pytest.approx(0.00135, rel=1e-6)


def test_claude_llm_client_forwards_pinned_model_and_system_user():
    from medreason.llm import ClaudeLLMClient
    client = ClaudeLLMClient(api_key="fake")
    client._client = _FakeClient(_FakeResp("ok", 100, 20))
    client.complete(system="MY SYSTEM", user="MY USER")
    kw = client._client.last_kwargs
    assert kw["model"] == "claude-sonnet-4-20250514"
    assert kw["system"] == "MY SYSTEM"
    assert kw["messages"] == [{"role": "user", "content": "MY USER"}]


def test_claude_llm_client_without_api_key_raises_at_complete():
    from medreason.llm import ClaudeLLMClient
    client = ClaudeLLMClient(api_key="")
    client._client = None
    with pytest.raises(RuntimeError) as exc:
        client.complete(system="s", user="u")
    assert "ANTHROPIC_API_KEY" in str(exc.value) or "api_key" in str(exc.value)


def test_claude_llm_client_rejects_unknown_model():
    from medreason.llm import ClaudeLLMClient
    with pytest.raises(ValueError):
        ClaudeLLMClient(model="claude-unknown")


# ── Skeleton clients ────────────────────────────────────────────────────────


def test_openai_llm_client_raises_not_implemented():
    from medreason.llm import OpenAILLMClient
    c = OpenAILLMClient(api_key="fake")
    with pytest.raises(NotImplementedError) as exc:
        c.complete(system="s", user="u")
    assert "Phase 7" in str(exc.value) or "skeleton" in str(exc.value).lower()


def test_gemini_llm_client_raises_not_implemented():
    from medreason.llm import GeminiLLMClient
    c = GeminiLLMClient(api_key="fake")
    with pytest.raises(NotImplementedError):
        c.complete(system="s", user="u")


def test_openai_llm_client_rejects_unknown_model():
    from medreason.llm import OpenAILLMClient
    with pytest.raises(ValueError):
        OpenAILLMClient(model="gpt-99")


def test_gemini_llm_client_rejects_unknown_model():
    from medreason.llm import GeminiLLMClient
    with pytest.raises(ValueError):
        GeminiLLMClient(model="gemini-99")
