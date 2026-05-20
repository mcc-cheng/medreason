"""Hand-seeded metacognitive rules for drug discovery v0.1.

These 8 rules encode the most common failure patterns in lead
optimization triage — the "how does default reasoning go wrong"
metacognitive patterns that the user identified as the core insight.

All trigger enum lists are empty (wildcard) so Tier 1 structural
matching passes everything through. The real matching surface is
`semantic_predicate` via Tier 2 dense embedding retrieval.
"""

from __future__ import annotations

from medreason.ontology import (
    ReasoningRule,
    RuleEvidence,
    RuleStatus,
    RuleTrigger,
)


def build_drugdisc_seed_rules() -> list[ReasoningRule]:
    """Return 8 hand-crafted metacognitive rules for drug discovery."""

    rules: list[ReasoningRule] = []

    def _rule(
        rule_id: str,
        predicate: str,
        action: str,
        rationale: str,
        polarity: str,
        citation: str,
    ) -> ReasoningRule:
        return ReasoningRule(
            rule_id=rule_id,
            status=RuleStatus.ACTIVE,
            trigger=RuleTrigger(
                cpt_families=[],
                icd10_chapters=[],
                payers=[],
                semantic_predicate=predicate,
            ),
            action=action,
            rationale=rationale,
            polarity=polarity,
            evidence=RuleEvidence(
                supporting_case_ids=["seed_manual"],
                source_policy_citation=citation,
                proposer_model="human-seed",
                proposer_run_id="drugdisc_v01_seed",
            ),
        )

    # Rule 1 — hERG trap
    rules.append(_rule(
        rule_id="dd_seed_001",
        predicate="high potency kinase inhibitor with cLogP > 3.5 and basic nitrogen",
        action="Deny default potency-first optimization; check hERG IC50 BEFORE advancing when cLogP > 3.5 plus basic nitrogen present.",
        rationale="Default reasoning prioritizes sub-nanomolar potency without checking cardiac safety. hERG liability probability exceeds 70% when cLogP > 3.5 + basic amine.",
        polarity="supports_denial",
        citation="§3.1",
    ))

    # Rule 2 — P-gp efflux (MW reduction trap)
    rules.append(_rule(
        rule_id="dd_seed_002",
        predicate="poor oral bioavailability with high efflux ratio in Caco-2",
        action="Deny default MW reduction for bioavailability; check efflux ratio FIRST — P-gp efflux requires HBD reduction, not MW reduction.",
        rationale="Default reasoning treats poor bioavailability as a passive permeability problem and reduces MW. When efflux ratio > 2, the issue is active P-gp transport — MW reduction alone will not fix it.",
        polarity="supports_denial",
        citation="§2.1 + §3.5",
    ))

    # Rule 3 — RO5 exception for macrocyclics / PROTACs
    rules.append(_rule(
        rule_id="dd_seed_003",
        predicate="macrocyclic peptide or PROTAC degrader with multiple Lipinski violations",
        action="Approve default for macrocyclics and PROTACs despite RO5 violations; RO5 does not apply to these modalities.",
        rationale="Default reasoning rejects any molecule violating RO5. RO5 was derived from oral small molecules and does not apply to macrocyclic peptides, PROTACs, or ADC payloads.",
        polarity="supports_approval",
        citation="§1.1",
    ))

    # Rule 4 — CYP3A4 TDI hard stop
    rules.append(_rule(
        rule_id="dd_seed_004",
        predicate="CYP3A4 time-dependent inhibition detected in candidate compound",
        action="Deny regardless of potency; CYP3A4 time-dependent inhibition is a hard stop — cannot be fixed without scaffold change.",
        rationale="Default reasoning may deprioritize CYP TDI when potency is exceptional. TDI creates irreversible DDI risk that outweighs any potency benefit.",
        polarity="supports_denial",
        citation="§3.2",
    ))

    # Rule 5 — CNS BBB: P-gp is a hard exclusion
    rules.append(_rule(
        rule_id="dd_seed_005",
        predicate="CNS target compound with P-glycoprotein substrate activity",
        action="Deny default passive-permeability optimization for CNS; P-gp efflux at BBB is a hard exclusion — fix substrate status first.",
        rationale="Default reasoning for CNS compounds focuses on TPSA and MW for BBB penetration. When the compound is a P-gp substrate, efflux at the BBB prevents brain exposure regardless of passive permeability.",
        polarity="supports_denial",
        citation="§4",
    ))

    # Rule 6 — PAINS with validated binding
    rules.append(_rule(
        rule_id="dd_seed_006",
        predicate="PAINS-flagged compound with orthogonal binding validation",
        action="Approve default despite PAINS flag when SPR, ITC, or crystallography confirms specific binding mechanism.",
        rationale="Default reasoning auto-rejects PAINS-flagged compounds. Some PAINS scaffolds (hydroxamic acids, catechols) are legitimate pharmacophores when binding is validated orthogonally.",
        polarity="supports_approval",
        citation="§5.1",
    ))

    # Rule 7 — Allosteric potency scaling
    rules.append(_rule(
        rule_id="dd_seed_007",
        predicate="allosteric modulator with IC50 or EC50 in 100-1000 nM range",
        action="Approve default for allosteric mechanisms showing 100-1000 nM potency; evaluate on functional efficacy, not raw IC50.",
        rationale="Default reasoning applies orthosteric potency thresholds to allosteric modulators. Allosteric mechanisms have different potency scaling — 100-1000 nM is the expected range.",
        polarity="supports_approval",
        citation="§6.2",
    ))

    # Rule 8 — Oncology benefit-risk for structural alerts
    rules.append(_rule(
        rule_id="dd_seed_008",
        predicate="structural alert for genotoxicity or reactive metabolite in oncology compound",
        action="Approve default in oncology when structural alert is present BUT compound is cytotoxic by design with high unmet need indication.",
        rationale="Default reasoning rejects compounds with genotoxicity or reactive-met alerts regardless of indication. Oncology benefit-risk calculus accepts these alerts when the compound targets cancers with no standard of care.",
        polarity="supports_approval",
        citation="§9.1",
    ))

    # Rule 9 — Prodrug evaluation trap
    rules.append(_rule(
        rule_id="dd_seed_009",
        predicate="prodrug candidate with poor in vitro ADMET properties for the prodrug form",
        action="Approve default for prodrug candidates despite poor prodrug-form properties; evaluate the ACTIVE PARENT compound's properties and the prodrug formulation's in vivo PK, not the prodrug form's in vitro data.",
        rationale="Default reasoning penalizes high CLint, low permeability, and low potency without checking whether the candidate is a prodrug. Prodrug forms are designed to have poor in vitro properties — the relevant data is the active parent's potency/ADMET and the prodrug's in vivo bioavailability.",
        polarity="supports_approval",
        citation="§7",
    ))

    return rules
