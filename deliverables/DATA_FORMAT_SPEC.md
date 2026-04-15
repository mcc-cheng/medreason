# Data Format Spec — Veridicus Lead-Op Retrospective

One-page spec describing the shape of de-identified campaign data the
Veridicus harness ingests. If you can export your campaign data into
roughly this shape (CSV, JSONL, Parquet, DuckDB — any tabular format
works), the retro can run in ~7 days from the time the files arrive.

---

## Table 1 — `compounds`

One row per synthesized compound, in chronological submission order.

| Column                  | Type      | Required | Notes |
|-------------------------|-----------|----------|-------|
| `compound_id`           | string    | yes      | Opaque token. Internal IDs, project codenames, or CAS numbers should be replaced with stable opaque tokens (e.g., `CMPD_A_0001`). |
| `timestamp`             | ISO 8601  | yes      | Submission date; month-level precision is fine. |
| `round_index`           | int       | yes      | Which SAR round this compound belongs to (1-indexed). Maps to `decision_points.round_index`. |
| `smiles`                | string    | preferred | Canonical SMILES. Omit if structure disclosure is NDA-gated — descriptors alone still work, with degraded Murcko-scaffold matching. |
| `mw`                    | float     | preferred | If `smiles` is provided, we compute descriptors ourselves. |
| `clogp`, `tpsa`, `hbd`, `hba` | numeric | preferred | Same as MW — computed from SMILES if available. |
| `proposed_modification` | string    | optional | Free-text (e.g., "add F to ring A", "N-methylate"). Improves interpretability of our rationales. |
| `assay_readouts`        | JSON      | yes      | Assay measurements. See Table 3 below. |
| `outcome_label`         | string    | yes      | `advanced` \| `failed_<reason>` \| `withdrawn`. The `_<reason>` suffix is free-text but we'll bucket common ones (hERG, selectivity, admet, permeability, toxicity). |

---

## Table 2 — `decision_points`

One row per "boundary" between SAR rounds — i.e., each moment the team
decided what to pursue next after reviewing the prior round's assays.

| Column                   | Type      | Required | Notes |
|--------------------------|-----------|----------|-------|
| `decision_point_id`      | string    | yes      | Opaque, e.g. `DP_01`. |
| `round_index`            | int       | yes      | The round_index this DP launches (i.e., the DP at the start of round 3 has round_index = 3). |
| `timestamp`              | ISO 8601  | yes      | When the decision was made; month-level precision fine. |
| `team_direction_chosen`  | enum      | yes      | One of: `potency`, `selectivity`, `ADMET`, `scaffold_hop`. If the team pursued multiple directions in parallel, mark the dominant one and note it in `notes`; we'll switch to top-K scoring for those DPs. |
| `notes`                  | string    | optional | "why this direction" in your own words — a single sentence is plenty. Helps us calibrate our rationales against yours. |

---

## Table 3 — assay readouts JSON schema

Inside `compounds.assay_readouts`. All keys optional; include only
what you have. Values are per-compound, post-QC. Common fields:

| Key                  | Units         | Notes |
|----------------------|---------------|-------|
| `ic50_nm`            | nM            | On-target IC50 (or Ki / EC50 equivalent — note which in notes). |
| `selectivity_ratio`  | dimensionless | Off-target IC50 / on-target IC50. Against the most relevant counter-target. |
| `herg_ic50_um`       | μM            | hERG patch-clamp IC50. |
| `caco2_papp`         | 10⁻⁶ cm/s    | Apparent permeability (A → B is fine). |
| `clint_ul_min_mg`    | μL/min/mg     | Microsomal intrinsic clearance. Species assumed HLM unless stated. |
| `cyp3a4_tdi`         | bool          | Time-dependent CYP3A4 inhibition flag. |
| `solubility_ugml`    | μg/mL         | Kinetic or thermodynamic; either is fine if labeled. |
| `plasma_stability`   | % remaining   | Optional. |

Other assay fields are welcome — we'll include them in the agent's
context. The four directions the agent ranks (potency, selectivity,
ADMET, scaffold_hop) only need the first five rows above to work.

---

## Example row (compounds)

```json
{
  "compound_id": "CMPD_A_0017",
  "timestamp": "2023-06-01",
  "round_index": 3,
  "smiles": "O=C(Nc1ccc(F)cc1)C2CCN(Cc3ccncn3)CC2",
  "mw": 345.4,
  "clogp": 2.1,
  "tpsa": 55.1,
  "hbd": 1,
  "hba": 6,
  "proposed_modification": "aminopyrimidine tail, reduce basicity",
  "assay_readouts": {
    "ic50_nm": 180,
    "selectivity_ratio": 7.2,
    "herg_ic50_um": 18.3,
    "clint_ul_min_mg": 85,
    "cyp3a4_tdi": false
  },
  "outcome_label": "advanced"
}
```

---

## De-identification policy

- **compound_id** must be an opaque token the retro cannot deanonymize.
  Map internal IDs → tokens in a file you keep; we never see the map.
- **Project codenames** should be stripped from free-text fields
  (`proposed_modification`, `notes`).
- **SMILES** are not intrinsically revealing unless your compound
  series is already in a public filing; if they are, omit them and
  we'll use descriptors alone (minor accuracy hit).
- No ELN text, no slide decks, no patient data. Just the five tables
  above.

## Cross-customer data handling

Your campaign data never leaves the retro run. No rules extracted
from this retro enter a cross-customer rule store. The
abstraction-gate policy governing any future cross-customer rule
extraction is stated separately in Veridicus's engagement terms.

---

## Turnaround commitment

From the moment the files arrive in the above shape, Veridicus
commits to a 7-calendar-day retro turnaround:

1. Ingest into DuckDB campaign schema (day 1).
2. Blind-replay harness runs, 3 seeds per configuration (day 2-4).
3. Report scored against the pre-registered prediction card (day 5-6).
4. Delivered as a PDF report + reproducible notebook (day 7).

If your data has edge cases (multi-direction rounds, very sparse
assay coverage, or extensive missing-outcome labels), the 7-day
clock restarts once the schema questions are resolved — we'll flag
these on day 1.
