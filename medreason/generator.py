"""Synthetic benchmark case generator."""

from __future__ import annotations

import json
import random
from uuid import uuid4

import anthropic

from .config import ANTHROPIC_API_KEY, BENCHMARK_DIR, CLAUDE_MODEL
from .local_cases import LOCAL_CASES
from .ontology import (
    BenchmarkCase, DenialReason, Difficulty, FacilityType, Outcome, Payer,
    PriorAuthTaskConfig,
)

COMMON_PROCEDURES = [
    {"cpt": "99213", "desc": "Office visit established moderate", "icd10": ["M54.5", "M79.3"], "denial_rate": 0.15},
    {"cpt": "99214", "desc": "Office visit established high", "icd10": ["M54.5", "G89.29"], "denial_rate": 0.25},
    {"cpt": "27447", "desc": "Total knee arthroplasty", "icd10": ["M17.11", "M17.12"], "denial_rate": 0.35},
    {"cpt": "29881", "desc": "Knee arthroscopy meniscectomy", "icd10": ["M23.21", "S83.511A"], "denial_rate": 0.30},
    {"cpt": "72148", "desc": "MRI lumbar without contrast", "icd10": ["M54.5", "M51.16"], "denial_rate": 0.40},
    {"cpt": "70553", "desc": "MRI brain with/without contrast", "icd10": ["R51.9", "G43.909"], "denial_rate": 0.20},
    {"cpt": "43239", "desc": "Upper GI endoscopy with biopsy", "icd10": ["K21.0", "K25.9"], "denial_rate": 0.18},
    {"cpt": "64483", "desc": "Epidural steroid injection lumbar", "icd10": ["M54.5", "M54.16"], "denial_rate": 0.45},
    {"cpt": "77386", "desc": "IMRT radiation delivery", "icd10": ["C61", "C34.11"], "denial_rate": 0.22},
    {"cpt": "90837", "desc": "Psychotherapy 60 min", "icd10": ["F33.1", "F41.1"], "denial_rate": 0.28},
    {"cpt": "97110", "desc": "Therapeutic exercises", "icd10": ["M54.5", "S83.511A"], "denial_rate": 0.32},
    {"cpt": "20610", "desc": "Joint injection major joint", "icd10": ["M17.11", "M19.011"], "denial_rate": 0.20},
]

PAYERS = list(Payer)
FACILITIES = list(FacilityType)
DENIAL_REASONS = list(DenialReason)
MODIFIERS = ["25", "59", "76", "77", "XE", "XS", "XP", "XU"]


def _assign_difficulty(rng: random.Random) -> Difficulty:
    r = rng.random()
    if r < 0.25:
        return Difficulty.EASY
    elif r < 0.75:
        return Difficulty.MEDIUM
    return Difficulty.HARD


def _assign_outcome(proc: dict, difficulty: Difficulty, rng: random.Random) -> tuple[Outcome, DenialReason | None]:
    base_denial = proc["denial_rate"]
    if difficulty == Difficulty.EASY:
        denial_prob = base_denial * 0.3
    elif difficulty == Difficulty.MEDIUM:
        denial_prob = base_denial * 1.0
    else:
        denial_prob = base_denial * 1.5

    r = rng.random()
    if r < denial_prob:
        if difficulty == Difficulty.HARD and rng.random() < 0.4:
            return Outcome.OVERTURNED_ON_APPEAL, rng.choice(DENIAL_REASONS)
        return Outcome.DENIED, rng.choice(DENIAL_REASONS)
    return Outcome.APPROVED, None


def generate_local(n: int, seed: int = 42) -> list[BenchmarkCase]:
    """Generate cases from the pre-written local bank, cycling if n > len(LOCAL_CASES)."""
    rng = random.Random(seed)
    cases = []
    pool = list(LOCAL_CASES)
    rng.shuffle(pool)
    for i in range(n):
        case = pool[i % len(pool)].model_copy(deep=True)
        if i >= len(pool):
            case.case_id = f"local-{i+1:03d}"
        cases.append(case)
    return cases


def generate_with_api(n: int, seed: int = 42) -> list[BenchmarkCase]:
    """Generate cases using Claude API for realistic clinical content."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    rng = random.Random(seed)
    cases: list[BenchmarkCase] = []

    for i in range(n):
        proc = rng.choice(COMMON_PROCEDURES)
        payer = rng.choice(PAYERS)
        facility = rng.choice(FACILITIES)
        difficulty = _assign_difficulty(rng)
        outcome, denial_reason = _assign_outcome(proc, difficulty, rng)
        mods = rng.sample(MODIFIERS, k=rng.randint(0, 2))

        config = PriorAuthTaskConfig(
            payer=payer,
            cpt_code=proc["cpt"],
            icd10_codes=proc["icd10"],
            modifiers=mods,
            facility_type=facility,
            denial_reason=denial_reason,
        )

        prompt = f"""Generate a realistic prior authorization benchmark case.

Procedure: {proc['desc']} (CPT {proc['cpt']})
Payer: {payer.value}
Facility: {facility.value}
Difficulty: {difficulty.value}
Outcome: {outcome.value}
{f"Denial reason: {denial_reason.value}" if denial_reason else ""}
ICD-10 codes: {', '.join(proc['icd10'])}

Return JSON with these fields:
- clinical_notes: 3-5 sentences of de-identified but realistic clinical details (include specific measurements, durations, findings)
- policy_excerpt: 2-3 sentences of specific payer policy criteria
- ground_truth_reasoning: array of 3-5 logical reasoning steps that lead to the outcome
- prior_eobs: array of 0-2 brief prior EOB summaries (empty array if not applicable)

Be clinically accurate. Include specific values (scores, durations, measurements). Make the reasoning chain logical and step-by-step.

Return ONLY valid JSON, no markdown fences."""

        try:
            response = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            data = json.loads(text)

            cases.append(BenchmarkCase(
                case_id=f"gen-{uuid4().hex[:8]}",
                task_config=config,
                clinical_notes=data["clinical_notes"],
                prior_eobs=data.get("prior_eobs", []),
                policy_excerpt=data["policy_excerpt"],
                ground_truth_outcome=outcome,
                ground_truth_reasoning=data["ground_truth_reasoning"],
                difficulty=difficulty,
            ))
        except Exception as e:
            print(f"  Warning: API case generation failed ({e}), using local fallback")
            fallback = LOCAL_CASES[i % len(LOCAL_CASES)].model_copy(deep=True)
            fallback.case_id = f"gen-{uuid4().hex[:8]}"
            cases.append(fallback)

    return cases


def save_cases(cases: list[BenchmarkCase], filename: str = "benchmark_cases.json"):
    path = BENCHMARK_DIR / filename
    data = [c.model_dump(mode="json") for c in cases]
    path.write_text(json.dumps(data, indent=2))
    return path


def load_cases(filename: str = "benchmark_cases.json") -> list[BenchmarkCase]:
    path = BENCHMARK_DIR / filename
    data = json.loads(path.read_text())
    return [BenchmarkCase(**c) for c in data]
