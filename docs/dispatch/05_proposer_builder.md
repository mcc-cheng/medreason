# Dispatch — `proposer-builder`

**From:** architect
**Phase 2 plan:** `/Users/davidzhang/Desktop/Origin/Personal/medreason/docs/targetval_phase2_plan.md` (§Builder 5)
**Working dir:** `/Users/davidzhang/Desktop/Origin/Personal/medreason`
**Python:** `/opt/miniconda3/bin/python`
**Verify command:** `cd /Users/davidzhang/Desktop/Origin/Personal/medreason && /opt/miniconda3/bin/python -m pytest tests/test_targetval_*.py -q`

## Scope

Replace `cross_agent_analyzer.propose_corrective_rules`'s `return []`
stub with a real LLM-driven proposer that emits CANDIDATE
`ReasoningRule`s, one per `SystematicError` (where severity ≥ floor).
This is the moat — the cross-agent metacognitive memory turning
systematic miss-patterns into universal-layer rules.

## Files to edit / create

- `medreason/targetval/cross_agent_analyzer.py` — implement
  `propose_corrective_rules`. Keep the existing `detect_systematic_errors`
  unchanged (its tests must keep passing).
- (new) `medreason/targetval/cross_agent_prompts.py` — the prompt
  builder for cross-case rules.
- `tests/test_targetval_cross_agent.py` — add proposer tests; REPLACE
  the existing `test_propose_corrective_rules_skeleton_returns_empty`
  (which asserts the stub) with a real "emits a rule" test.

## Interface signature

```python
# medreason/targetval/cross_agent_analyzer.py

def propose_corrective_rules(
    errors: list[SystematicError],
    cases: list[TargetValidationCase],
    llm: LLMClient,
    *,
    proposer_run_id: Optional[str] = None,
    severity_floor: float = _DEFAULT_SEVERITY_FLOOR,
    seed: int = 0,
) -> list[ReasoningRule]:
    ...
```

(Return type stays `list[ReasoningRule]` so existing callers don't break.)

Implementation contract:
1. For each `SystematicError` with `severity >= severity_floor`:
   - Build a de-identified prompt using
     `cross_agent_prompts.build_proposer_prompt(error, cases)`.
     - List `error.affected_case_ids` (opaque IDs only).
     - Include the swarm's average `bypass_risk_score` across those
       case_ids.
     - Include the ground-truth pattern (mechanism name + frequency).
     - DO NOT include `InternalEvidence.readouts` anywhere. Filter
       explicitly even if the cases come from a retro fixture that
       has none — the prompt builder must defensively skip it.
2. Call `llm.complete(system=SYSTEM_PROMPT_CROSS_AGENT, user=user_msg, seed=seed)`.
3. Parse JSON output. Expected shape:
   ```json
   {"rules": [{"semantic_predicate": "...", "action": "...",
                 "rationale": "...", "polarity": "..."}]}
   ```
4. For each candidate dict, build a `ReasoningRule` with:
   - auto-generated `rule_id`
   - `status = RuleStatus.CANDIDATE`
   - `trigger = RuleTrigger(semantic_predicate=...)` — leave CPT/ICD/
     payer fields empty (these are not prior-auth ontology).
   - `action`, `rationale`, `polarity` from the parsed dict.
   - `evidence = RuleEvidence(supporting_case_ids=list(error.affected_case_ids),
        source_policy_citation=f"cross_agent:{error.error_kind}:n={len(error.affected_case_ids)}",
        proposer_model=llm.model_version,
        proposer_run_id=proposer_run_id or f"crossagent_{uuid4().hex[:8]}")`
5. Reject candidates that fail:
   - `_count_words(action) > ACTION_MAX_WORDS` (reuse from
     `medreason.extraction.rule_proposer`),
   - `_has_patient_identifier(text)` on action / rationale /
     semantic_predicate (reuse from `rule_proposer`),
   - empty action or empty semantic_predicate.
   Rejected candidates are silently dropped (no `ProposalResult`
   wrapper at this layer; this is internal to the cross-agent path).
   `propose_corrective_rules` returns the surviving rules only.

`cross_agent_prompts.SYSTEM_PROMPT_CROSS_AGENT` should make these
expectations explicit to the LLM: emit JSON with a `rules` key, each
rule ≤ 25 words on `action`, no patient identifiers, no specific
gene symbols if avoidable.

## Test expectations

New tests:
- `test_proposer_emits_no_rules_for_empty_errors` — `propose_corrective_rules([], [], FakeLLMClient())` returns `[]`.
- `test_proposer_emits_rule_for_missed_bypass` — canned
  `FakeLLMClient(responses=['{"rules": [{"semantic_predicate": "kinase target with paralog_count >= 2", "action": "Raise bypass_risk by 0.3.", "rationale": "Paralog redundancy enables compensation.", "polarity": "requires_check"}]}'])` →
  returns one rule with `status==CANDIDATE`.
- `test_proposer_rejects_overlong_action` — canned 40-word action → 0 rules.
- `test_proposer_rejects_patient_identifier_leak` — `"patient John ..."` → 0 rules.
- `test_proposer_supporting_case_ids_come_from_affected` — round-trip the
  `error.affected_case_ids` into `rule.evidence.supporting_case_ids`.
- `test_proposer_skips_below_severity_floor` — severity=0.2, floor=0.5 → 0 rules.
- `test_run_cross_agent_analysis_returns_candidate_rules_now` — uses
  the same `_swarm_that_misses_kras_and_egfr_bypass` fixture plus a
  wired FakeLLM → `analysis.candidate_rules` non-empty.

REPLACE: `test_propose_corrective_rules_skeleton_returns_empty` —
delete it; the stub is gone.

Existing tests that MUST still pass:
- `test_detect_missed_bypass_categories`
- `test_severity_floor_filters_low_prevalence`

(The third existing test, `test_run_cross_agent_analysis_returns_analysis_record`,
currently asserts `analysis.candidate_rules == []`. Update that line
to `assert analysis.candidate_rules` since the analyzer now emits
candidates. This is the only edit needed to that test.)

## Dependencies on other builders

- `TargetMemo.bypass_signals_seen` from `swarm-llm-builder` — the
  prompt can mention "the swarm did see these signals but ranked the
  case low anyway", which makes the corrective rule more pointed.
- The emitted `ReasoningRule` shape must pass
  `LayerPolicy.validate_rule(rule, customer_tag=None)` for `Layer.UNIVERSAL`
  (per `layer-store-builder`'s lock). Specifically: never stamp a
  `customer_tag` into provenance.

## Hard constraints

- Tests use `FakeLLMClient` ONLY, with `responses=[...]` for each
  scenario.
- NO `InternalEvidence.readouts` in the prompt. Filter explicitly.
- NEVER stamp `customer_tag` into the corrective-rule provenance —
  these are universal rules.
- Do NOT touch `dashboard/`.
- Do NOT run `git add` / `git commit`.

## When done

SendMessage `tester` with a summary:
- replaced + added tests,
- the canned JSON shape your tests expect from the LLM (so
  `mapk-curator`'s end-to-end test uses the same canned shape),
- pytest count before vs after.
