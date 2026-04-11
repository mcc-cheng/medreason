"""Tests for medreason.runners — Phase 2.

Covers:
- AgentRunner Protocol conformance for all three adapters.
- build_case_prompt() determinism and content.
- parse_json_response() against plain / fenced / prose-wrapped / malformed
  inputs.
- compute_cost_usd() math at boundaries.
- ClaudeRunner end-to-end with a FakeClient (no network, no API key).
- ClaudeRunner parse-error path: returns a valid AgentResult with
  mode tagged and cost 0.
- OpenAIRunner / GeminiRunner: structural Protocol conformance, clear
  NotImplementedError from run().
- Legacy modules still import cleanly.
"""

from __future__ import annotations

import json

import pytest

from tests.conftest import make_case


# ── Protocol conformance ─────────────────────────────────────────────────────


def test_agent_runner_protocol_claude():
    from medreason.runners import AgentRunner, ClaudeRunner
    r = ClaudeRunner(api_key="fake")
    assert isinstance(r, AgentRunner)
    assert isinstance(r.runner_id, str) and r.runner_id
    assert isinstance(r.model_version, str) and r.model_version
    assert isinstance(r.supports_memory, bool)
    assert callable(r.run)
    assert callable(r.estimated_cost_per_call)


def test_agent_runner_protocol_openai():
    from medreason.runners import AgentRunner, OpenAIRunner
    r = OpenAIRunner(api_key="fake")
    assert isinstance(r, AgentRunner)
    assert isinstance(r.runner_id, str) and r.runner_id
    assert r.supports_memory is True


def test_agent_runner_protocol_gemini():
    from medreason.runners import AgentRunner, GeminiRunner
    r = GeminiRunner(api_key="fake")
    assert isinstance(r, AgentRunner)
    assert isinstance(r.runner_id, str) and r.runner_id
    assert r.supports_memory is True


def test_runner_id_suffix():
    from medreason.runners import ClaudeRunner
    r = ClaudeRunner(api_key="fake", runner_id_suffix="memory")
    assert r.runner_id == "claude-sonnet-4-20250514:memory"
    assert r.model_version == "claude-sonnet-4-20250514"


def test_claude_runner_unknown_model_raises():
    from medreason.runners import ClaudeRunner
    with pytest.raises(ValueError) as exc:
        ClaudeRunner(api_key="fake", model="claude-nonsense-1")
    assert "Unknown Claude model" in str(exc.value)


# ── build_case_prompt ─────────────────────────────────────────────────────────


def test_build_case_prompt_deterministic():
    """Two identical cases must produce identical prompts — no timestamps,
    no random UUIDs. The eval harness depends on this for reproducibility."""
    from medreason.runners import build_case_prompt
    c = make_case()
    p1 = build_case_prompt(c)
    p2 = build_case_prompt(c)
    assert p1 == p2


def test_build_case_prompt_contains_task_fields():
    from medreason.runners import build_case_prompt
    c = make_case(cpt="72148", icds=["M54.5", "M51.16"])
    p = build_case_prompt(c)
    assert "Aetna" in p
    assert "72148" in p
    assert "M54.5" in p
    assert "M51.16" in p
    assert "outpatient" in p


def test_build_case_prompt_withholds_policy_by_default():
    from medreason.runners import build_case_prompt
    c = make_case(policy_excerpt="Secret policy text must not leak.")
    p = build_case_prompt(c)  # include_policy=False by default
    assert "Secret policy text" not in p
    assert "NOT provided" in p or "not provided" in p.lower()


def test_build_case_prompt_includes_policy_when_requested():
    from medreason.runners import build_case_prompt
    c = make_case(policy_excerpt="Allowed after 6 weeks PT.")
    p = build_case_prompt(c, include_policy=True)
    assert "Allowed after 6 weeks PT." in p


def test_build_case_prompt_includes_modifiers_and_eobs():
    from medreason.runners import build_case_prompt
    c = make_case(modifiers=["59", "RT"], prior_eobs=["denied 2025-02"])
    p = build_case_prompt(c)
    assert "59" in p
    assert "RT" in p
    assert "denied 2025-02" in p


# ── parse_json_response ──────────────────────────────────────────────────────


def test_parse_plain_json():
    from medreason.runners import parse_json_response
    text = '{"determination": "approved", "confidence": 0.9}'
    obj = parse_json_response(text)
    assert obj["determination"] == "approved"
    assert obj["confidence"] == 0.9


def test_parse_fenced_json():
    from medreason.runners import parse_json_response
    text = '```json\n{"determination": "denied", "confidence": 0.7}\n```'
    obj = parse_json_response(text)
    assert obj["determination"] == "denied"


def test_parse_fenced_no_language():
    from medreason.runners import parse_json_response
    text = '```\n{"a": 1}\n```'
    obj = parse_json_response(text)
    assert obj["a"] == 1


def test_parse_json_with_leading_prose():
    from medreason.runners import parse_json_response
    text = 'Here is my answer:\n{"determination": "approved", "confidence": 0.8}\nHope that helps.'
    obj = parse_json_response(text)
    assert obj["determination"] == "approved"


def test_parse_rejects_empty():
    from medreason.runners import ResponseParseError, parse_json_response
    with pytest.raises(ResponseParseError):
        parse_json_response("")


def test_parse_rejects_none():
    from medreason.runners import ResponseParseError, parse_json_response
    with pytest.raises(ResponseParseError):
        parse_json_response(None)  # type: ignore[arg-type]


def test_parse_rejects_no_json():
    from medreason.runners import ResponseParseError, parse_json_response
    with pytest.raises(ResponseParseError):
        parse_json_response("I cannot determine this case without more information.")


def test_parse_rejects_json_array():
    """The schema expects an object at top level, not an array."""
    from medreason.runners import ResponseParseError, parse_json_response
    with pytest.raises(ResponseParseError):
        parse_json_response('["approved", "denied"]')


def test_parse_rejects_malformed_braces():
    from medreason.runners import ResponseParseError, parse_json_response
    with pytest.raises(ResponseParseError):
        parse_json_response("Here you go: { not json at all")


# ── compute_cost_usd ─────────────────────────────────────────────────────────


def test_compute_cost_usd_claude_sonnet_4():
    from medreason.runners import compute_cost_usd
    # 1M input tokens at $3/MTok + 500k output at $15/MTok
    cost = compute_cost_usd(
        input_tokens=1_000_000,
        output_tokens=500_000,
        input_per_mtok=3.0,
        output_per_mtok=15.0,
    )
    assert cost == pytest.approx(3.0 + 7.5)


def test_compute_cost_usd_zero_tokens():
    from medreason.runners import compute_cost_usd
    assert compute_cost_usd(0, 0, 3.0, 15.0) == 0.0


def test_compute_cost_usd_zero_rates():
    from medreason.runners import compute_cost_usd
    assert compute_cost_usd(1000, 1000, 0.0, 0.0) == 0.0


# ── ClaudeRunner end-to-end with FakeClient ──────────────────────────────────


class _FakeUsage:
    def __init__(self, in_t: int, out_t: int):
        self.input_tokens = in_t
        self.output_tokens = out_t


class _FakeContentBlock:
    def __init__(self, text: str):
        self.text = text


class _FakeResponse:
    def __init__(self, text: str, in_t: int, out_t: int):
        self.content = [_FakeContentBlock(text)]
        self.usage = _FakeUsage(in_t, out_t)


class _FakeClient:
    """Mimics the minimal surface of anthropic.Anthropic() used by
    ClaudeRunner.run(): client.messages.create(...) -> response."""
    def __init__(self, response: _FakeResponse):
        self._response = response
        self.messages = self
        self.last_kwargs: dict = {}
        self.call_count = 0

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        self.call_count += 1
        return self._response


def _fake_response(
    determination: str = "approved",
    confidence: float = 0.9,
    reasoning: str = "step 1; step 2",
    in_t: int = 412,
    out_t: int = 88,
) -> _FakeResponse:
    payload = {
        "determination": determination,
        "reasoning_chain": reasoning,
        "confidence": confidence,
        "key_factors": ["PT duration", "imaging findings"],
    }
    return _FakeResponse(json.dumps(payload), in_t=in_t, out_t=out_t)


def test_claude_runner_end_to_end_happy_path():
    from medreason.ontology import Outcome
    from medreason.runners import ClaudeRunner

    runner = ClaudeRunner(api_key="fake")
    runner._client = _FakeClient(_fake_response(determination="approved"))

    case = make_case(ground_truth_outcome=Outcome.APPROVED)
    result = runner.run(case, seed=11)

    assert result.case_id == case.case_id
    assert result.determination == Outcome.APPROVED
    assert result.correct is True
    assert result.confidence == pytest.approx(0.9)
    assert result.input_tokens == 412
    assert result.output_tokens == 88
    assert result.seed == 11
    assert result.runner_id == "claude-sonnet-4-20250514"
    assert result.mode == "zero_shot"
    # Cost: 412 * 3 / 1e6 + 88 * 15 / 1e6 = 0.001236 + 0.00132 = 0.002556
    assert result.cost_usd == pytest.approx(0.002556, rel=1e-6)
    assert result.latency_ms >= 0.0
    assert len(result.key_factors) == 2


def test_claude_runner_wrong_determination_marks_incorrect():
    from medreason.ontology import Outcome
    from medreason.runners import ClaudeRunner
    runner = ClaudeRunner(api_key="fake")
    runner._client = _FakeClient(_fake_response(determination="denied"))
    case = make_case(ground_truth_outcome=Outcome.APPROVED)
    result = runner.run(case)
    assert result.determination == Outcome.DENIED
    assert result.correct is False


def test_claude_runner_unknown_determination_falls_back_to_denied():
    """An unparseable determination must NEVER silently approve."""
    from medreason.ontology import Outcome
    from medreason.runners import ClaudeRunner
    runner = ClaudeRunner(api_key="fake")
    runner._client = _FakeClient(_fake_response(determination="MAYBE"))
    case = make_case(ground_truth_outcome=Outcome.APPROVED)
    result = runner.run(case)
    assert result.determination == Outcome.DENIED


def test_claude_runner_parse_failure_returns_result_not_exception():
    """A malformed response must produce a concrete AgentResult with
    correct=False, cost 0, and the parse error visible in reasoning.
    The harness may retry, but the runner itself must not throw."""
    from medreason.runners import ClaudeRunner
    runner = ClaudeRunner(api_key="fake")
    runner._client = _FakeClient(_FakeResponse("not json at all", 100, 50))
    case = make_case()
    result = runner.run(case)
    assert result.correct is False
    assert result.confidence == 0.0
    assert result.cost_usd == 0.0
    assert "parse_error" in result.reasoning_chain
    assert result.input_tokens == 100
    assert result.output_tokens == 50


def test_claude_runner_system_extra_prepended():
    """Injected memory must land as a prefix to the system prompt, so
    rules appear BEFORE the base adversarial reviewer instructions."""
    from medreason.runners import ClaudeRunner
    runner = ClaudeRunner(api_key="fake")
    runner._client = _FakeClient(_fake_response())
    case = make_case()
    runner.run(case, system_extra="=== MEMORY RULES ===\nrule 1\nrule 2")
    system = runner._client.last_kwargs["system"]
    assert system.startswith("=== MEMORY RULES ===")
    assert "prior authorization" in system.lower()
    # Order matters: memory first, base prompt second
    mem_idx = system.index("MEMORY RULES")
    base_idx = system.lower().index("prior authorization")
    assert mem_idx < base_idx


def test_claude_runner_mode_memory_when_system_extra():
    from medreason.runners import ClaudeRunner
    runner = ClaudeRunner(api_key="fake")
    runner._client = _FakeClient(_fake_response())
    case = make_case()
    result = runner.run(case, system_extra="some rules here")
    assert result.mode == "memory"


def test_claude_runner_mode_zero_shot_when_no_system_extra():
    from medreason.runners import ClaudeRunner
    runner = ClaudeRunner(api_key="fake")
    runner._client = _FakeClient(_fake_response())
    case = make_case()
    result = runner.run(case)
    assert result.mode == "zero_shot"


def test_claude_runner_forwards_pinned_model_in_api_call():
    from medreason.runners import ClaudeRunner
    runner = ClaudeRunner(api_key="fake")
    runner._client = _FakeClient(_fake_response())
    case = make_case()
    runner.run(case)
    assert runner._client.last_kwargs["model"] == "claude-sonnet-4-20250514"


def test_claude_runner_without_api_key_raises_at_run_time():
    """Constructing ClaudeRunner without a key must succeed (for tests
    that inject a fake client), but calling .run() without either a key
    or an injected client must raise with a clear message."""
    from medreason.runners import ClaudeRunner
    runner = ClaudeRunner(api_key="")  # no key, no injection
    runner._client = None  # explicit, default is None anyway
    case = make_case()
    with pytest.raises(RuntimeError) as exc:
        runner.run(case)
    assert "ANTHROPIC_API_KEY" in str(exc.value) or "api_key" in str(exc.value)


def test_claude_runner_fenced_json_response_parses():
    """A Claude response that wraps JSON in a ```json fence must still
    parse. This IS a failure mode of the model in practice."""
    from medreason.runners import ClaudeRunner
    runner = ClaudeRunner(api_key="fake")
    fenced = (
        '```json\n'
        '{"determination":"denied","reasoning_chain":"failed criteria",'
        '"confidence":0.8,"key_factors":["missing PT"]}\n'
        '```'
    )
    runner._client = _FakeClient(_FakeResponse(fenced, 300, 60))
    from medreason.ontology import Outcome
    case = make_case(ground_truth_outcome=Outcome.DENIED)
    result = runner.run(case)
    assert result.determination == Outcome.DENIED
    assert result.correct is True


# ── Skeleton runners ─────────────────────────────────────────────────────────


def test_openai_runner_run_raises_not_implemented():
    from medreason.runners import OpenAIRunner
    runner = OpenAIRunner(api_key="fake")
    case = make_case()
    with pytest.raises(NotImplementedError) as exc:
        runner.run(case)
    assert "Phase 7" in str(exc.value) or "skeleton" in str(exc.value).lower()


def test_gemini_runner_run_raises_not_implemented():
    from medreason.runners import GeminiRunner
    runner = GeminiRunner(api_key="fake")
    case = make_case()
    with pytest.raises(NotImplementedError) as exc:
        runner.run(case)
    assert "Phase 7" in str(exc.value) or "skeleton" in str(exc.value).lower()


def test_openai_runner_unknown_model_raises():
    from medreason.runners import OpenAIRunner
    with pytest.raises(ValueError):
        OpenAIRunner(model="gpt-99")


def test_gemini_runner_unknown_model_raises():
    from medreason.runners import GeminiRunner
    with pytest.raises(ValueError):
        GeminiRunner(model="gemini-99")


# ── Legacy backward compat ───────────────────────────────────────────────────


def test_legacy_modules_still_import_after_phase_2():
    """Every pre-rework medreason.* module must still import cleanly."""
    import importlib
    for mod in [
        "medreason.agent",
        "medreason.store",
        "medreason.injector",
        "medreason.extractor",
        "medreason.generator",
        "medreason.benchmark",
        "medreason.local_cases",
    ]:
        importlib.import_module(mod)
