"""CMS LCD/NCD ingestion.

Phase 3 scope: parse a single LCD XML file into an LCDPolicy object.
Phase 6 will add the downloader (network + caching), NCD support, and
tolerance for the real CMS export schema (which has substantially more
fields than the v0.0 fixture).

Why stdlib xml.etree instead of lxml:
- lxml is a C extension that complicates Windows installs.
- Our needs at this phase are XPath-free and small.
- Phase 6 may move to lxml if the real CMS schema requires it; this
  module's public surface (parse_lcd_xml, parse_lcd_bytes) will not change.

Security note: `xml.etree.ElementTree.fromstring` is safe against the
common XML attack vectors (no entity resolution, no external DTD load)
on modern Python. We still never accept unvetted network inputs at this
phase — download_lcd() is a stub.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

from .schemas import LCDCriterion, LCDLimitation, LCDPolicy


class CMSIngestionError(RuntimeError):
    """Raised when an LCD/NCD document cannot be parsed into LCDPolicy."""


# ── Low-level element helpers ─────────────────────────────────────────────────


def _text(el: ET.Element | None, default: str = "") -> str:
    if el is None:
        return default
    return (el.text or default).strip()


def _optional(el: ET.Element, tag: str) -> str | None:
    child = el.find(tag)
    if child is None:
        return None
    value = (child.text or "").strip()
    return value or None


def _required(el: ET.Element, tag: str) -> str:
    child = el.find(tag)
    if child is None:
        raise CMSIngestionError(f"Missing required element <{tag}>")
    value = (child.text or "").strip()
    if not value:
        raise CMSIngestionError(f"Required element <{tag}> is empty")
    return value


def _collect_children_text(parent: ET.Element | None, child_tags: Iterable[str]) -> list[str]:
    """Collect stripped text from any of the given child tags, in order.

    Multiple tag names are accepted because the real CMS schema sometimes
    uses <cpt> and sometimes <cpt_range>, for example.
    """
    if parent is None:
        return []
    out: list[str] = []
    for child in parent:
        if child.tag in child_tags:
            value = (child.text or "").strip()
            if value:
                out.append(value)
    return out


# ── Public parse API ─────────────────────────────────────────────────────────


def parse_lcd_xml(path: Path | str) -> LCDPolicy:
    """Parse an LCD XML file from disk into an LCDPolicy.

    Raises:
        CMSIngestionError: if the document is missing required fields
            (document_id, title) or is not well-formed XML.
    """
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise CMSIngestionError(f"LCD file not found: {p}")
    return parse_lcd_bytes(p.read_bytes())


def parse_lcd_bytes(raw: bytes) -> LCDPolicy:
    """Parse raw LCD XML bytes into an LCDPolicy.

    Tolerant of missing optional fields (limitations, jurisdiction,
    revision date). Required fields: <document_id>, <title>, and at
    least one <cpt*> under <cpt_codes>.
    """
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as e:
        raise CMSIngestionError(f"Malformed XML: {e}") from e

    document_id = _required(root, "document_id")
    title = _required(root, "title")

    cpts = _collect_children_text(root.find("cpt_codes"), {"cpt", "cpt_range"})
    if not cpts:
        raise CMSIngestionError(
            f"LCD {document_id}: <cpt_codes> has no <cpt> or <cpt_range> children"
        )

    covered_icd = _collect_children_text(
        root.find("covered_icd10"), {"icd", "icd10"}
    )

    indications: list[LCDCriterion] = []
    ind_parent = root.find("indications")
    if ind_parent is not None:
        for crit in ind_parent.findall("criterion"):
            crit_id = crit.get("id") or ""
            text = (crit.text or "").strip()
            tag = crit.get("tag") or ""
            if not crit_id or not text:
                # Skip degenerate criteria rather than failing — real CMS
                # dumps occasionally include placeholder nodes.
                continue
            indications.append(LCDCriterion(criterion_id=crit_id, text=text, tag=tag))

    limitations: list[LCDLimitation] = []
    lim_parent = root.find("limitations")
    if lim_parent is not None:
        for lim in lim_parent.findall("limitation"):
            lim_id = lim.get("id") or ""
            text = (lim.text or "").strip()
            tag = lim.get("tag") or ""
            if not lim_id or not text:
                continue
            limitations.append(LCDLimitation(limitation_id=lim_id, text=text, tag=tag))

    return LCDPolicy(
        document_id=document_id,
        title=title,
        source="cms_lcd",
        contractor=_optional(root, "contractor"),
        jurisdiction=_optional(root, "jurisdiction"),
        effective_date=_optional(root, "effective_date"),
        revision_effective_date=_optional(root, "revision_effective_date"),
        url=_optional(root, "url"),
        cpt_codes=cpts,
        covered_icd10=covered_icd,
        indications=indications,
        limitations=limitations,
    )


# ── Downloader (Phase 6 target) ──────────────────────────────────────────────


def download_lcd(lcd_id: str, cache_dir: Path | str) -> Path:
    """Download a single LCD XML by id, cache to disk, return the path.

    NOT IMPLEMENTED in Phase 3. The real implementation will:
    1. Hit https://www.cms.gov/medicare-coverage-database/reports/downloads.aspx
       to fetch the quarterly lcd.zip, unpack once per quarter, and cache.
    2. Look up `lcd_id` in the unpacked index and return the file path.
    3. Verify SHA256 against a version manifest so stale caches surface.

    Phase 3 callers should use parse_lcd_xml() against a local fixture or
    a file the developer has already downloaded.
    """
    raise NotImplementedError(
        "download_lcd() is a Phase 6 target. For Phase 3, point the CLI "
        "at a local LCD file via --lcd <path>. See "
        "medreason_bench/data/fixtures/sample_lcd.xml for the expected shape."
    )
