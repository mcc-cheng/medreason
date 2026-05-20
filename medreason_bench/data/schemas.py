"""Policy schemas — the internal shape every ingestion source must produce.

The case builder, the stratifier, and the leaderboard dataset card all
speak this vocabulary. Different sources (CMS LCD XML, CMS NCD, OIG
appeals PDFs, payer medical policy HTML) each have their own extractor
that outputs LCDPolicy objects. Phase 3 ships the CMS LCD extractor only.

Keep these lean: one criterion = one atomic, quotable requirement. The
rule_proposer (Phase 5b) extracts IF-THEN rules from these, so criteria
that bundle multiple checks ("6 weeks PT AND positive exam AND imaging
correlation") must be split at ingestion time, not later.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class LCDCriterion(BaseModel):
    """One atomic, quotable coverage criterion.

    `criterion_id` is the section marker from the source document
    (e.g. "C.1"). The rule_proposer cites this in rule.evidence, so
    downstream traceability depends on it being stable.
    """
    criterion_id: str                 # source-local id, e.g. "C.1"
    text: str                         # exact text from the policy
    tag: str = ""                     # optional semantic tag
                                      # (e.g. "conservative_therapy",
                                      # "neurological_findings")


class LCDLimitation(BaseModel):
    """One exclusion / frequency limit / repeat-study restriction."""
    limitation_id: str                # e.g. "L.1"
    text: str
    tag: str = ""


class LCDPolicy(BaseModel):
    """An ingested coverage determination document.

    Minimal fields needed by the case builder and the rule proposer.
    Additional source-specific fields (jurisdiction, contractor id,
    revision history) can be added without breaking consumers because
    they default to None / empty.
    """
    document_id: str                  # e.g. "L34522"
    title: str
    source: str = "cms_lcd"           # "cms_lcd" | "cms_ncd" | "oig_appeal" | ...
    contractor: Optional[str] = None
    jurisdiction: Optional[str] = None
    effective_date: Optional[str] = None         # ISO date
    revision_effective_date: Optional[str] = None
    url: Optional[str] = None

    cpt_codes: list[str] = Field(default_factory=list)
    covered_icd10: list[str] = Field(default_factory=list)
    indications: list[LCDCriterion] = Field(default_factory=list)
    limitations: list[LCDLimitation] = Field(default_factory=list)

    def citation(self, criterion_id: str) -> str:
        """Return a human-readable citation for a specific criterion.

        Used by the rule_proposer to fill RuleEvidence.source_policy_citation.
        """
        return f"CMS LCD {self.document_id} §{criterion_id}"
