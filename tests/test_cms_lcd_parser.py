"""Tests for medreason_bench.data.cms_lcd_ncd — Phase 3."""

from __future__ import annotations

from pathlib import Path

import pytest

from medreason_bench.data import (
    CMSIngestionError,
    LCDPolicy,
    parse_lcd_bytes,
    parse_lcd_xml,
)


FIXTURE = Path(__file__).parent.parent / "medreason_bench" / "data" / "fixtures" / "sample_lcd.xml"


# ── Happy path ───────────────────────────────────────────────────────────────


def test_parse_sample_fixture_returns_policy():
    policy = parse_lcd_xml(FIXTURE)
    assert isinstance(policy, LCDPolicy)
    assert policy.document_id == "L34522"
    assert "Lumbar Spine" in policy.title
    assert policy.source == "cms_lcd"
    assert policy.contractor == "Novitas Solutions, Inc."
    assert policy.jurisdiction == "JL"
    assert policy.effective_date == "2024-01-01"
    assert policy.revision_effective_date == "2025-04-15"
    assert policy.url and "cms.gov" in policy.url


def test_parse_sample_fixture_cpts():
    policy = parse_lcd_xml(FIXTURE)
    assert policy.cpt_codes == ["72148", "72149", "72158"]


def test_parse_sample_fixture_covered_icds():
    policy = parse_lcd_xml(FIXTURE)
    assert "M51.16" in policy.covered_icd10
    assert "M54.5" in policy.covered_icd10
    assert "G83.4" in policy.covered_icd10
    assert len(policy.covered_icd10) == 6


def test_parse_sample_fixture_indications_preserve_order_and_tags():
    policy = parse_lcd_xml(FIXTURE)
    assert len(policy.indications) == 4
    ids = [c.criterion_id for c in policy.indications]
    assert ids == ["C.1", "C.2", "C.3", "C.4"]
    assert policy.indications[0].tag == "conservative_therapy"
    assert "conservative therapy" in policy.indications[0].text.lower()
    assert policy.indications[1].tag == "neurological_findings"


def test_parse_sample_fixture_limitations():
    policy = parse_lcd_xml(FIXTURE)
    assert len(policy.limitations) == 3
    ids = [l.limitation_id for l in policy.limitations]
    assert ids == ["L.1", "L.2", "L.3"]
    assert policy.limitations[0].tag == "repeat_timeframe"
    assert "repeat" in policy.limitations[0].text.lower()


def test_policy_citation_format():
    policy = parse_lcd_xml(FIXTURE)
    assert policy.citation("C.1") == "CMS LCD L34522 §C.1"
    assert policy.citation("L.2") == "CMS LCD L34522 §L.2"


# ── Robustness ───────────────────────────────────────────────────────────────


def test_parse_rejects_missing_file():
    with pytest.raises(CMSIngestionError) as exc:
        parse_lcd_xml(Path("/nonexistent/path/to/lcd.xml"))
    assert "not found" in str(exc.value).lower()


def test_parse_rejects_malformed_xml():
    with pytest.raises(CMSIngestionError) as exc:
        parse_lcd_bytes(b"<lcd><document_id>L1</document")  # truncated
    assert "malformed" in str(exc.value).lower()


def test_parse_rejects_missing_document_id():
    xml = b"""<?xml version="1.0"?>
    <lcd><title>Missing ID</title><cpt_codes><cpt>99213</cpt></cpt_codes></lcd>"""
    with pytest.raises(CMSIngestionError) as exc:
        parse_lcd_bytes(xml)
    assert "document_id" in str(exc.value).lower()


def test_parse_rejects_missing_title():
    xml = b"""<?xml version="1.0"?>
    <lcd><document_id>L1</document_id><cpt_codes><cpt>99213</cpt></cpt_codes></lcd>"""
    with pytest.raises(CMSIngestionError) as exc:
        parse_lcd_bytes(xml)
    assert "title" in str(exc.value).lower()


def test_parse_rejects_empty_cpt_list():
    xml = b"""<?xml version="1.0"?>
    <lcd><document_id>L1</document_id><title>T</title><cpt_codes/></lcd>"""
    with pytest.raises(CMSIngestionError) as exc:
        parse_lcd_bytes(xml)
    assert "cpt" in str(exc.value).lower()


def test_parse_accepts_missing_optional_fields():
    xml = b"""<?xml version="1.0"?>
    <lcd>
      <document_id>L9999</document_id>
      <title>Minimal test</title>
      <cpt_codes><cpt>99213</cpt></cpt_codes>
    </lcd>"""
    policy = parse_lcd_bytes(xml)
    assert policy.document_id == "L9999"
    assert policy.contractor is None
    assert policy.jurisdiction is None
    assert policy.covered_icd10 == []
    assert policy.indications == []
    assert policy.limitations == []


def test_parse_skips_degenerate_criteria():
    """A <criterion> element without an id or text is skipped, not fatal."""
    xml = b"""<?xml version="1.0"?>
    <lcd>
      <document_id>L1</document_id>
      <title>T</title>
      <cpt_codes><cpt>99213</cpt></cpt_codes>
      <indications>
        <criterion id="C.1" tag="good">Real criterion text.</criterion>
        <criterion id="">empty id</criterion>
        <criterion id="C.2"></criterion>
      </indications>
    </lcd>"""
    policy = parse_lcd_bytes(xml)
    assert len(policy.indications) == 1
    assert policy.indications[0].criterion_id == "C.1"


def test_parser_accepts_cpt_range_child_tag():
    """Real CMS exports sometimes use <cpt_range> instead of <cpt>."""
    xml = b"""<?xml version="1.0"?>
    <lcd>
      <document_id>L1</document_id>
      <title>T</title>
      <cpt_codes><cpt_range>72148</cpt_range><cpt>72149</cpt></cpt_codes>
    </lcd>"""
    policy = parse_lcd_bytes(xml)
    assert policy.cpt_codes == ["72148", "72149"]


def test_download_lcd_is_stub():
    from medreason_bench.data import download_lcd
    with pytest.raises(NotImplementedError) as exc:
        download_lcd("L34522", Path("/tmp"))
    assert "Phase 6" in str(exc.value) or "phase 6" in str(exc.value).lower()
