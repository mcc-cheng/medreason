"""
ingest_rxrx3.py — Load rxrx3-core compound/gene data into the Drug Discovery Canvas
knowledge graph (PostgreSQL via direct psycopg2 inserts).

Only metadata and DTI annotations are downloaded — the 18 GB image corpus is skipped.

Usage:
    pip install datasets huggingface_hub psycopg2-binary python-dotenv
    python dashboard/scripts/ingest_rxrx3.py [--limit 200] [--dry-run] [--genes-only]

Flags:
    --limit N       max compound nodes to import (default 200)
    --dry-run       print what would be inserted, no DB writes
    --genes-only    only upsert gene nodes, skip compounds and edges
"""

import argparse
import os
import re
import sys
import uuid
from pathlib import Path

# ── Env ───────────────────────────────────────────────────────

def load_env() -> str:
    """Return DATABASE_URL from dashboard/.env.local (or environment)."""
    env_path = Path(__file__).parent.parent / ".env.local"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("DATABASE_URL="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    url = os.environ.get("DATABASE_URL")
    if not url:
        sys.exit("DATABASE_URL not found in dashboard/.env.local or environment")
    return url


# ── EC50 → Bayesian confidence ────────────────────────────────

def ec50_to_confidence(ec50_um: float) -> float:
    """Sigmoid: 0.1 µM → 0.91  |  10 µM → 0.5  |  100 µM → 0.09"""
    return round(1.0 / (1.0 + ec50_um / 10.0), 4)


def confidence_to_priors(conf: float) -> tuple[float, float]:
    """Return (alpha, beta) pseudo-counts summing to 10."""
    alpha = round(conf * 10, 2)
    beta  = round(10 - alpha, 2)
    return max(alpha, 0.1), max(beta, 0.1)


# ── Node ID helpers ───────────────────────────────────────────

def gene_node_id(gene: str) -> str:
    return re.sub(r"[^A-Z0-9_-]", "-", gene.upper().strip())


def compound_node_id(name: str) -> str:
    return re.sub(r"[^A-Z0-9_-]", "-", name.upper().strip())[:48]


# ── DB upserts ────────────────────────────────────────────────

def upsert_node(cur, node_id: str, name: str, node_type: str,
                mw: float | None, external_id: str, smiles: str | None,
                dry_run: bool) -> None:
    if dry_run:
        print(f"  [node] {node_id} | {node_type} | {name[:60]}")
        return
    cur.execute(
        """
        INSERT INTO "Node" (id, name, type, "molecularWeight", "dangerLevel",
                            "originDepartment", smiles, "externalId",
                            "createdAt", "updatedAt")
        VALUES (%s, %s, %s::"NodeType", %s, NULL, %s, %s, %s, NOW(), NOW())
        ON CONFLICT (id) DO UPDATE
          SET name        = EXCLUDED.name,
              smiles      = COALESCE(EXCLUDED.smiles, "Node".smiles),
              "externalId"= COALESCE(EXCLUDED."externalId", "Node"."externalId"),
              "updatedAt" = NOW()
        """,
        (node_id, name, node_type, mw, "rxrx3-core", smiles, external_id),
    )


def upsert_edge_with_provenance(cur, source_id: str, target_id: str,
                                 interaction: str, confidence: float,
                                 obs_count: int, alpha: float, beta: float,
                                 reasoning: str, dry_run: bool) -> None:
    if dry_run:
        print(f"  [edge] {source_id} --[{interaction} conf={confidence}]--> {target_id}")
        return

    cur.execute(
        """
        INSERT INTO "Edge" (id, "sourceId", "targetId", "interactionType"::"EdgeInteractionType",
                            "confidenceScore", "observationCount",
                            "priorAlpha", "priorBeta", "createdAt", "updatedAt")
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
        ON CONFLICT ("sourceId", "targetId", "interactionType") DO UPDATE
          SET "confidenceScore"  = EXCLUDED."confidenceScore",
              "observationCount" = EXCLUDED."observationCount",
              "priorAlpha"       = EXCLUDED."priorAlpha",
              "priorBeta"        = EXCLUDED."priorBeta",
              "updatedAt"        = NOW()
        RETURNING id
        """,
        (str(uuid.uuid4()), source_id, target_id, interaction,
         confidence, obs_count, alpha, beta),
    )
    row = cur.fetchone()
    if row:
        edge_id = row[0]
        cur.execute(
            """
            INSERT INTO "ProvenanceEntry"
              (id, "edgeId", "agentReasoningSnapshot", "simulationSource"::"SimulationSource",
               "evidenceWeight", "observedOutcome"::"ObservedOutcome", timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, NOW())
            """,
            (str(uuid.uuid4()), edge_id, reasoning,
             "RXRX3_PHENOMICS", 1.0, "SUPPORT"),
        )


# ── Main ──────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest rxrx3-core into the knowledge graph")
    parser.add_argument("--limit",      type=int, default=200, help="Max compounds to import")
    parser.add_argument("--dry-run",    action="store_true",   help="Print plan, no DB writes")
    parser.add_argument("--genes-only", action="store_true",   help="Only ingest gene nodes")
    args = parser.parse_args()

    # ── Lazy imports (require pip install) ────────────────────
    try:
        import psycopg2
    except ImportError:
        sys.exit("Run: pip install psycopg2-binary")

    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Run: pip install datasets")

    print("Loading rxrx3-core metadata (streaming — no images downloaded)…")

    # ── Load dataset streaming ────────────────────────────────
    # streaming=True avoids downloading the 18 GB image corpus.
    # We only iterate the metadata fields we need.
    ds = load_dataset(
        "recursionpharma/rxrx3-core",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    genes: dict[str, str]     = {}   # node_id → display name
    compounds: dict[str, str] = {}   # node_id → display name
    # DTI edges: (compound_id, gene_id) → {'ec50': float, 'interaction': str}
    dti_edges: dict[tuple[str, str], dict] = {}

    seen = 0
    print("Scanning metadata rows…", end="", flush=True)
    for row in ds:
        seen += 1
        if seen % 50_000 == 0:
            print(f" {seen//1000}k", end="", flush=True)

        treatment = (row.get("treatment") or row.get("gene") or "").strip()
        conc      = row.get("treatment_conc") or row.get("concentration")
        perttype  = row.get("perturbation_type") or ""

        if not treatment:
            continue

        # Classify as gene knockout vs compound based on concentration presence
        is_compound = conc is not None and str(conc).replace(".", "").isdigit()
        is_gene     = not is_compound or "crispr" in perttype.lower() or "sirna" in perttype.lower()

        if is_gene and not is_compound:
            nid = gene_node_id(treatment)
            genes.setdefault(nid, treatment)
        elif is_compound:
            nid = compound_node_id(treatment)
            compounds.setdefault(nid, treatment)

            # If there's a gene target column, record a DTI edge
            target_gene = (row.get("target") or row.get("gene_target") or "").strip()
            if target_gene:
                gnid = gene_node_id(target_gene)
                genes.setdefault(gnid, target_gene)
                ec50 = float(conc) if conc else 10.0
                key  = (nid, gnid)
                if key not in dti_edges or ec50 < dti_edges[key]["ec50"]:
                    dti_edges[key] = {"ec50": ec50, "interaction": "INHIBITS"}

    print(f"\nFound {len(genes)} gene targets, {len(compounds)} compounds, {len(dti_edges)} DTI annotations")

    # ── Limit compounds ───────────────────────────────────────
    compound_items = list(compounds.items())[: args.limit]
    allowed_compound_ids = {cid for cid, _ in compound_items}

    # ── Connect to DB ─────────────────────────────────────────
    database_url = load_env()
    if not args.dry_run:
        conn = psycopg2.connect(database_url)
        cur  = conn.cursor()
    else:
        conn = cur = None  # type: ignore

    try:
        # ── Gene nodes ────────────────────────────────────────
        print(f"\nUpserting {len(genes)} gene nodes…")
        for nid, name in genes.items():
            upsert_node(cur, nid, name, "BIOMOLECULE",
                        None, f"rxrx3:{nid}", None, args.dry_run)

        if not args.genes_only:
            # ── Compound nodes ────────────────────────────────
            print(f"Upserting {len(compound_items)} compound nodes (limit={args.limit})…")
            for nid, name in compound_items:
                upsert_node(cur, nid, name, "CHEMICAL_CANDIDATE",
                            None, f"rxrx3:{nid}", None, args.dry_run)

            # ── DTI edges ─────────────────────────────────────
            edge_count = 0
            print(f"Upserting DTI edges…")
            for (cid, gid), info in dti_edges.items():
                if cid not in allowed_compound_ids:
                    continue
                if gid not in genes:
                    continue
                ec50       = info["ec50"]
                conf       = ec50_to_confidence(ec50)
                alpha, beta = confidence_to_priors(conf)
                reasoning  = (
                    f"rxrx3-core DTI annotation: {cid} → {gid} "
                    f"(EC50={ec50:.2f} µM, confidence={conf:.3f})"
                )
                upsert_edge_with_provenance(
                    cur, cid, gid, info["interaction"],
                    conf, round(alpha), alpha, beta,
                    reasoning, args.dry_run,
                )
                edge_count += 1

            print(f"  {edge_count} edges written")

        if not args.dry_run and conn:
            conn.commit()
            print("\nCommitted. Run `npm run dev` and refresh the graph view.")

    except Exception as e:
        if conn:
            conn.rollback()
        raise e
    finally:
        if cur:  cur.close()
        if conn: conn.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
