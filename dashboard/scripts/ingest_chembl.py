"""
ingest_chembl.py — Load ChEMBL drug-target interaction data into the knowledge graph.

Pulls from the free ChEMBL REST API (no key needed):
  - Human protein targets (UniProt-linked)
  - Approved/clinical small-molecule drugs
  - IC50 / Ki / EC50 binding affinities → Bayesian confidence scores
  - UniProt sequences and GO terms for protein nodes

Usage (from repo root, venv active):
    python dashboard/scripts/ingest_chembl.py [--limit 500] [--dry-run]

Flags:
    --limit N    max activity records to fetch per target (default 500)
    --dry-run    print plan, no DB writes
"""

import argparse
import os
import re
import sys
import time
import uuid
from pathlib import Path

# ── Env / DB ──────────────────────────────────────────────────

def load_db_url() -> str:
    env = Path(__file__).parent.parent / ".env.local"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("DATABASE_URL="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get("DATABASE_URL", "")

# ── Confidence from pChEMBL ───────────────────────────────────
# pChEMBL = -log10(IC50 in molar), so 7 = 100 nM, 9 = 1 nM

def pchembl_to_confidence(pchembl: float) -> float:
    """Maps pChEMBL (5–10) to confidence (0.2–0.98)."""
    # sigmoid centred at pChEMBL=7 (100 nM), steepness=1.5
    import math
    return round(1 / (1 + math.exp(-1.5 * (pchembl - 7))), 4)

def confidence_to_priors(conf: float) -> tuple[float, float]:
    alpha = round(conf * 10, 2)
    beta  = round((1 - conf) * 10, 2)
    return max(alpha, 0.1), max(beta, 0.1)

# ── ChEMBL REST helpers ───────────────────────────────────────

BASE = "https://www.ebi.ac.uk/chembl/api/data"

def get_json(url: str, params: dict | None = None) -> dict:
    import urllib.request, urllib.parse, json
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())

def chembl_paginate(endpoint: str, params: dict, limit: int = 200) -> list[dict]:
    """Fetch all pages from a ChEMBL list endpoint."""
    params = {**params, "format": "json", "limit": 200, "offset": 0}
    results = []
    while True:
        data = get_json(f"{BASE}/{endpoint}", params)
        page = data.get(endpoint, []) or data.get("activities", []) or []
        results.extend(page)
        nxt = (data.get("page_meta") or {}).get("next")
        if not nxt or len(results) >= limit:
            break
        params["offset"] += 200
        time.sleep(0.2)
    return results[:limit]

# ── UniProt helpers ───────────────────────────────────────────

def fetch_uniprot(accession: str) -> dict:
    """Return {sequence, go_terms} for a UniProt accession."""
    import urllib.request, json
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            data = json.loads(r.read())
        seq = data.get("sequence", {}).get("value", "")
        go_terms = [
            ref["id"]
            for ref in data.get("dbReferences", [])
            if ref.get("type") == "GO"
        ]
        return {"sequence": seq, "go_terms": go_terms}
    except Exception:
        return {"sequence": "", "go_terms": []}

# ── Node ID helpers ───────────────────────────────────────────

def protein_node_id(gene_name: str) -> str:
    return re.sub(r"[^A-Z0-9_-]", "-", gene_name.upper().strip())[:40]

def compound_node_id(name: str) -> str:
    return re.sub(r"[^A-Z0-9_-]", "-", name.upper().strip())[:48]

# ── DB upserts ────────────────────────────────────────────────

def upsert_protein(cur, node_id: str, name: str, uniprot_id: str,
                   sequence: str, go_terms: list[str], dry_run: bool) -> None:
    if dry_run:
        print(f"  [protein] {node_id} | {name[:50]} | uniprot:{uniprot_id} | {len(go_terms)} GO terms")
        return
    go_array = "{" + ",".join(go_terms) + "}"
    cur.execute(
        """
        INSERT INTO "Node" (id, name, type, "molecularWeight", "dangerLevel",
                            "originDepartment", "externalId", sequence, "goTerms",
                            "createdAt", "updatedAt")
        VALUES (%s, %s, 'BIOMOLECULE'::"NodeType", NULL, NULL,
                'ChEMBL / UniProt', %s, %s, %s, NOW(), NOW())
        ON CONFLICT (id) DO UPDATE
          SET name        = EXCLUDED.name,
              "externalId"= COALESCE(EXCLUDED."externalId", "Node"."externalId"),
              sequence    = COALESCE(EXCLUDED.sequence,    "Node".sequence),
              "goTerms"   = CASE WHEN array_length(EXCLUDED."goTerms", 1) > 0
                                 THEN EXCLUDED."goTerms" ELSE "Node"."goTerms" END,
              "updatedAt" = NOW()
        """,
        (node_id, name, f"uniprot:{uniprot_id}", sequence or None, go_array),
    )

def upsert_compound(cur, node_id: str, name: str, chembl_id: str,
                    smiles: str, mw: float | None, dry_run: bool) -> None:
    if dry_run:
        print(f"  [compound] {node_id} | {name[:50]} | {chembl_id}")
        return
    cur.execute(
        """
        INSERT INTO "Node" (id, name, type, "molecularWeight", "dangerLevel",
                            "originDepartment", "externalId", smiles, "goTerms",
                            "createdAt", "updatedAt")
        VALUES (%s, %s, 'CHEMICAL_CANDIDATE'::"NodeType", %s, NULL,
                'ChEMBL', %s, %s, '{}', NOW(), NOW())
        ON CONFLICT (id) DO UPDATE
          SET name         = EXCLUDED.name,
              "externalId" = COALESCE(EXCLUDED."externalId", "Node"."externalId"),
              smiles       = COALESCE(EXCLUDED.smiles, "Node".smiles),
              "molecularWeight" = COALESCE(EXCLUDED."molecularWeight", "Node"."molecularWeight"),
              "updatedAt"  = NOW()
        """,
        (node_id, name, mw, f"chembl:{chembl_id}", smiles or None),
    )

def upsert_inhibits_edge(cur, source_id: str, target_id: str,
                          confidence: float, obs: int,
                          alpha: float, beta: float,
                          reasoning: str, dry_run: bool) -> None:
    if dry_run:
        print(f"  [edge] {source_id} --[INHIBITS conf={confidence:.3f}]--> {target_id}")
        return
    cur.execute(
        """
        INSERT INTO "Edge" (id, "sourceId", "targetId",
                            "interactionType",
                            "confidenceScore", "observationCount",
                            "priorAlpha", "priorBeta", "createdAt", "updatedAt")
        VALUES (%s, %s, %s, 'INHIBITS'::"EdgeInteractionType", %s, %s, %s, %s, NOW(), NOW())
        ON CONFLICT ("sourceId", "targetId", "interactionType") DO UPDATE
          SET "confidenceScore"  = GREATEST(EXCLUDED."confidenceScore", "Edge"."confidenceScore"),
              "observationCount" = "Edge"."observationCount" + 1,
              "priorAlpha"       = "Edge"."priorAlpha" + EXCLUDED."priorAlpha",
              "priorBeta"        = "Edge"."priorBeta"  + EXCLUDED."priorBeta",
              "updatedAt"        = NOW()
        RETURNING id
        """,
        (str(uuid.uuid4()), source_id, target_id, confidence, obs, alpha, beta),
    )
    row = cur.fetchone()
    if row:
        cur.execute(
            """
            INSERT INTO "ProvenanceEntry"
              (id, "edgeId", "agentReasoningSnapshot",
               "simulationSource",
               "evidenceWeight", "observedOutcome", timestamp)
            VALUES (%s, %s, %s, 'CHEMBL'::"SimulationSource", 1.0, 'SUPPORT'::"ObservedOutcome", NOW())
            """,
            (str(uuid.uuid4()), row[0], reasoning),
        )

# ── Seed targets — well-known human drug targets ──────────────
# These are ChEMBL target IDs for proteins we want to anchor the graph.
# Format: (chembl_target_id, gene_symbol, uniprot_accession)

SEED_TARGETS = [
    ("CHEMBL203",  "EGFR",    "P00533"),
    ("CHEMBL1862", "ABL1",    "P00519"),  # BCR-ABL kinase domain
    ("CHEMBL230",  "PTGS2",   "P35354"),  # COX-2
    ("CHEMBL2842", "BRAF",    "P15056"),
    ("CHEMBL279",  "ERBB2",   "P04626"),  # HER2
    ("CHEMBL340",  "VEGFR2",  "P35968"),  # KDR
    ("CHEMBL2971", "CDK4",    "P11802"),
    ("CHEMBL1824", "CDK6",    "Q00534"),
    ("CHEMBL1963", "PIK3CA",  "P42336"),  # PI3K alpha
    ("CHEMBL4523", "KRAS",    "P01116"),
    ("CHEMBL3045", "PARP1",   "P09874"),
    ("CHEMBL325",  "ESR1",    "P03372"),  # Estrogen receptor
    ("CHEMBL206",  "AR",      "P10275"),  # Androgen receptor
    ("CHEMBL1862", "BCR-ABL", "P00519"),
    ("CHEMBL4860", "BTK",     "Q06187"),
    ("CHEMBL2468", "ALK",     "Q9UM73"),
    ("CHEMBL5145", "MET",     "P08581"),
]

# ── Main ──────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit",   type=int, default=500)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        import psycopg2
    except ImportError:
        sys.exit("Run: pip install psycopg2-binary")

    db_url = load_db_url()
    if not db_url:
        sys.exit("DATABASE_URL not set")

    conn = None if args.dry_run else psycopg2.connect(db_url)
    cur  = None if args.dry_run else conn.cursor()

    proteins_done:  set[str] = set()
    compounds_done: set[str] = set()
    edges_done = 0

    try:
        for chembl_tid, gene_symbol, uniprot_acc in SEED_TARGETS:
            print(f"\n── {gene_symbol} ({chembl_tid}) ──────────────────")

            # Upsert protein node (fetch UniProt data once)
            prot_id = protein_node_id(gene_symbol)
            if prot_id not in proteins_done:
                print(f"  Fetching UniProt {uniprot_acc}…")
                up = fetch_uniprot(uniprot_acc)
                upsert_protein(cur, prot_id, gene_symbol, uniprot_acc,
                               up["sequence"], up["go_terms"], args.dry_run)
                proteins_done.add(prot_id)
                time.sleep(0.1)

            # Fetch activities for this target
            print(f"  Fetching ChEMBL activities…")
            try:
                activities = chembl_paginate("activity", {
                    "target_chembl_id":      chembl_tid,
                    "target_organism":       "Homo sapiens",
                    "standard_type__in":     "IC50,Ki,Kd,EC50",
                    "pchembl_value__isnull": "false",
                    "assay_type":            "B",  # binding assays
                }, limit=args.limit)
            except Exception as e:
                print(f"  Warning: {e}")
                continue

            print(f"  {len(activities)} activities found")

            for act in activities:
                mol_name    = (act.get("molecule_pref_name") or "").strip()
                chembl_mid  = act.get("molecule_chembl_id", "")
                pchembl     = act.get("pchembl_value")
                smiles      = act.get("canonical_smiles") or act.get("molecule_smiles") or ""
                mw          = act.get("mw_freebase")

                if not mol_name or not chembl_mid or not pchembl:
                    continue

                try:
                    pchembl = float(pchembl)
                except (TypeError, ValueError):
                    continue

                conf = pchembl_to_confidence(pchembl)
                if conf < 0.2:
                    continue  # too weak

                comp_id = compound_node_id(mol_name)

                # Upsert compound node
                if comp_id not in compounds_done:
                    upsert_compound(cur, comp_id, mol_name, chembl_mid,
                                    smiles, float(mw) if mw else None, args.dry_run)
                    compounds_done.add(comp_id)

                # Upsert INHIBITS edge
                alpha, beta = confidence_to_priors(conf)
                reasoning = (
                    f"ChEMBL {chembl_mid} → {chembl_tid} ({gene_symbol}): "
                    f"pChEMBL={pchembl:.2f} → conf={conf:.3f}"
                )
                upsert_inhibits_edge(cur, comp_id, prot_id, conf, 1,
                                     alpha, beta, reasoning, args.dry_run)
                edges_done += 1

            time.sleep(0.3)

        if conn:
            conn.commit()

    except Exception as e:
        if conn:
            conn.rollback()
        raise e
    finally:
        if cur:  cur.close()
        if conn: conn.close()

    print(f"\n── Summary ──────────────────────────────────────────")
    print(f"  Proteins  : {len(proteins_done)}")
    print(f"  Compounds : {len(compounds_done)}")
    print(f"  Edges     : {edges_done}")
    print(f"\nDone. Run propagate next:")
    print(f"  curl -X POST http://localhost:3000/api/propagate")

if __name__ == "__main__":
    main()
