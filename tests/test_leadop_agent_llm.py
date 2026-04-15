"""Smoke tests for the LLM-backed lead-op agent.

Uses a FakeLLMClient so no network calls. Tests cover: prompt
construction includes/excludes the memory block, JSON parser handles
fences + stray prose, direction/compound ranking repair.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from medreason.llm.base import LLMResponse

from medreason_bench.leadop.agent_llm import (
    build_user_prompt,
    _extract_json,
    _repair_compound_ranking,
    _repair_direction_ranking,
    propose_llm,
)
from medreason_bench.leadop.harness import build_contexts
from medreason_bench.leadop.schema import connect_campaign_db
from medreason_bench.leadop.synthetic import CAMPAIGN_ID, write_toy_campaign


class _FakeLLM:
    model_version = "fake"

    def __init__(self, reply: str, *, input_tokens: int = 10, output_tokens: int = 20):
        self.reply = reply
        self.calls: list[tuple[str, str]] = []
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

    def complete(self, *, system, user, max_tokens=2048, seed=0) -> LLMResponse:
        self.calls.append((system, user))
        return LLMResponse(
            text=self.reply,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cost_usd=0.0,
            model=self.model_version,
        )


@pytest.fixture
def toy_contexts(tmp_path: Path):
    db = write_toy_campaign(tmp_path / "toy.db")
    con = connect_campaign_db(db)
    try:
        return build_contexts(con, CAMPAIGN_ID)
    finally:
        con.close()


def test_user_prompt_includes_memory_block_when_on(toy_contexts) -> None:
    dp2 = toy_contexts[1]
    on = build_user_prompt(dp2, use_memory=True)
    off = build_user_prompt(dp2, use_memory=False)
    assert "METACOGNITIVE MEMORY" in on
    assert "METACOGNITIVE MEMORY" not in off
    assert "PRIOR DECISION RATIONALES" in on
    assert "PRIOR DECISION RATIONALES" not in off


def test_user_prompt_masks_candidate_outcomes(toy_contexts) -> None:
    dp1 = toy_contexts[0]
    prompt = build_user_prompt(dp1, use_memory=False)
    # Round 2 compound TOY_002 has outcome "failed_herg" — must NOT be leaked
    # into the candidate pool block.
    cand_block = prompt.split("CANDIDATE POOL FOR NEXT ROUND")[1]
    assert "failed_herg" not in cand_block
    assert "TOY_002" in cand_block


def test_user_prompt_shows_visible_compound_outcomes(toy_contexts) -> None:
    dp2 = toy_contexts[1]
    prompt = build_user_prompt(dp2, use_memory=False)
    vis_block = prompt.split("VISIBLE COMPOUND HISTORY")[1].split("CANDIDATE POOL")[0]
    # Round 2 failed_herg compound is now in visible block with outcome.
    assert "failed_herg" in vis_block


def test_extract_json_handles_plain() -> None:
    assert _extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_handles_fences() -> None:
    text = '```json\n{"a": 2}\n```'
    assert _extract_json(text) == {"a": 2}


def test_extract_json_handles_prose_wrap() -> None:
    text = 'Sure, here is my answer: {"a": 3} let me know.'
    assert _extract_json(text) == {"a": 3}


def test_repair_direction_pads_missing(toy_contexts) -> None:
    ctx = toy_contexts[0]
    result = _repair_direction_ranking(["ADMET"], ctx)
    assert result[0] == "ADMET"
    assert set(result) == {"potency", "selectivity", "ADMET", "scaffold_hop"}


def test_repair_direction_drops_invalid(toy_contexts) -> None:
    ctx = toy_contexts[0]
    result = _repair_direction_ranking(["nonsense", "potency", 5, "ADMET"], ctx)
    assert result[0] == "potency"
    assert result[1] == "ADMET"


def test_repair_compound_keeps_pool_only(toy_contexts) -> None:
    ctx = toy_contexts[1]
    pool_ids = {c.compound_id for c in ctx.candidate_pool}
    result = _repair_compound_ranking(["TOY_BOGUS", "TOY_004"], ctx)
    assert set(result) == pool_ids
    assert result[0] == "TOY_004"


def test_propose_llm_happy_path(toy_contexts) -> None:
    ctx = toy_contexts[0]
    reply = (
        '{"direction_ranking": ["ADMET", "potency", "selectivity", "scaffold_hop"],'
        ' "compound_ranking": ["TOY_002", "TOY_003"],'
        ' "rationale": "pivot"}'
    )
    llm = _FakeLLM(reply)
    decision, cost = propose_llm(ctx, use_memory=True, llm=llm, seed=1)
    assert decision.direction_ranking[0] == "ADMET"
    assert set(decision.compound_ranking) == {"TOY_002", "TOY_003"}
    assert cost.input_tokens == 10


def test_propose_llm_rejects_unparseable(toy_contexts) -> None:
    ctx = toy_contexts[0]
    llm = _FakeLLM("I cannot answer.")
    with pytest.raises(ValueError):
        propose_llm(ctx, use_memory=False, llm=llm)
