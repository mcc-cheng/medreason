"""Curated MAPK retro entry table — Phase 2 v0.2.

This module owns the ``MapkRetroEntry`` dataclass + the assembled
``MAPK_RETRO_ENTRIES_V0_2`` tuple. The actual row data is split across
``mapk_retro_entries.py`` (BRAF/MEK/ERK/RAS axes) and
``mapk_retro_entries_extra.py`` (RTKs + non-MAPK adjuncts) so each file
stays under the 500-line cap and the curation diff stays reviewable.

Universal-safe surface
----------------------
Every entry here describes a *publicly documented* (target, disease)
pair. The ``InternalEvidence`` sub-bundle of the case is intentionally
left empty in the builder (``_entry_from`` in ``mapk_retro.py``) — no
entry in this module may reference customer data or per-tenant
readouts. This is the moat: Universal-layer training data must be
cross-customer-safe.

Source convention
-----------------
Every ``notes`` field cites either a clinicaltrials.gov NCT ID or a
short ``per literature: <study description>`` reference patterned on
the proposer-builder convention. We do NOT invent DOIs — the
descriptions are deliberately verbose so a biologist can spot-check the
claim.

Coverage
--------
Twenty-two entries (≥ 20 required), drawn from the working list in
``mapk_retro.py``'s module docstring. The set is balanced for the
swarm validation surface:

- Outcomes: 9 APPROVED_LATER, 10 PHASE2_EFFICACY_NO,
  1 PHASE2_SAFETY_FAILURE, 1 ACTIVE_UNKNOWN, 1 best-effort.
- Bypass mechanisms: PARALOG_COMPENSATION, DOWNSTREAM_FEEDBACK,
  ALTERNATIVE_PATHWAY, RESISTANCE_MUTATION, NO_BYPASS_KNOWN. NO entry
  has ``UNKNOWN`` — that's an inference-time-only label.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MapkRetroEntry:
    """One row of the MAPK retro fixture.

    The fields map 1:1 onto the ``_entry_from`` builder in
    ``mapk_retro.py``. ``notes`` is required to carry a defensible
    source citation (NCT ID or ``per literature:`` description).
    """

    case_id: str
    gene: str
    family: str
    disease: str
    paralogs: tuple[str, ...]
    paralog_count: int
    redundancy: float
    feedback_loops: tuple[str, ...]
    n_trials: int
    n_p2_fail: int
    n_approvals: int
    trial_summary: str
    outcome: object  # GroundTruthOutcome — typed in source files
    bypass: object  # BypassMechanism — typed in source files
    notes: str = ""


# Assemble the v0.2 entry table from its two source files. Splitting
# the rows keeps each file under the project's 500-line cap. The order
# is curation-stable (BRAF → MEK → ERK → RAS → RTKs → non-MAPK adjuncts)
# so test assertions that index entries by case_id remain reproducible.
from .mapk_retro_entries import MAPK_RETRO_ENTRIES_CORE
from .mapk_retro_entries_extra import MAPK_RETRO_ENTRIES_EXTRA

MAPK_RETRO_ENTRIES_V0_2: tuple[MapkRetroEntry, ...] = (
    *MAPK_RETRO_ENTRIES_CORE,
    *MAPK_RETRO_ENTRIES_EXTRA,
)
