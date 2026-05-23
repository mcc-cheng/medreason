"""
ingest_string.py — Load STRING protein-protein similarity scores into the graph.

Uses the STRING REST API (no key needed) to get combined similarity scores
between all protein nodes currently in the database. Creates SIMILAR_TO edges.

Usage (from repo root, venv active):
    python dashboard/scripts/ingest_string.py [--min-score 0.4] [--dry-run]

Flags:
    --min-score F  minimum STRING combined score to include (default 0.4)
    --dry-run      print plan, no DB writes
"""

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

HUMAN_TAXON = "9606"
STRING_API  = "https://string-db.org/api/json"

# ── Env / DB ──────────────────────────────────────────────────

def load_db_url() -> str:
    env = Path(__file__).parent.parent / ".env.local"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("DATABASE_URL="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get("DATABASE_URL", "")

# ── STRING API ────────────────────────────────────────────────

def string_get(endpoint: str, params: dict) -> list[dict]:
    url = f"{STRING_API}/{endpoint}?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read())
    except Exception as e:
        print(f"  STRING API error: {e}")
        return []

def resolve_string_ids(gene_names: list[str]) -> dict[str, str]:
    """Map gene names → STRING protein IDs."""
    if not gene_names:
        return {}
    identifiers = "\r".join(gene_names)
    results = string_get("get_string_ids", {
        "identifiers": identifiers,
        "species":     HUMAN_TAXON,
        "limit":       1,
        "echo_query":  1,
    })
    mapping: dict[str, str] = {}
    for r in results:
        query   = r.get("queryItem", "").upper()
        str_id  = r.get("stringId", "")
        if query and str_id:
            mapping[query] = str_id
    return mapping

def get_ppi_network(string_ids: list[str], min_score: float) -> list[dict]:
    """Return all interactions between the given STRING IDs."""
    if len(string_ids) < 2:
        return []
    identifiers = "\r".join(string_ids)
    return string_get("network", {
        "identifiers":    identifiers,
        "species":        HUMAN_TAXON,
        "required_score": int(min_score * 1000),
        "network_type":   "functional",
    })

# ── DB helpers ────────────────────────────────────────────────

def get_all_protein_ids(cur) -> list[tuple[str, str]]:
    """Return [(node_id, gene_name)] for all BIOMOLECULE nodes."""
    cur.execute("""
        SELECT id, name FROM "Node"
        WHERE type = 'BIOMOLECULE'::"NodeType"
    """)
    return cur.fetchall()

def upsert_similar_to(cur, source_id: str, target_id: str,
                       score: float, reasoning: str, dry_run: bool) -> None:
    if dry_run:
        print(f"  [sim] {source_id} ~ {target_id}  score={score:.3f}")
        return
    # Use LEAST to canonicalise order so (A,B) and (B,A) don't both get stored
    # We store both directions since the graph is directed (traversal is easier)
    for src, tgt in [(source_id, target_id), (target_id, source_id)]:
        cur.execute(
            """
            INSERT INTO "Edge" (id, "sourceId", "targetId",
                                "interactionType",
                                "confidenceScore", "observationCount",
                                "priorAlpha", "priorBeta",
                                "createdAt", "updatedAt")
            VALUES (%s, %s, %s, 'SIMILAR_TO'::"EdgeInteractionType", %s, 1, %s, %s, NOW(), NOW())
            ON CONFLICT ("sourceId", "targetId", "interactionType") DO UPDATE
              SET "confidenceScore" = GREATEST(EXCLUDED."confidenceScore",
                                               "Edge"."confidenceScore"),
                  "updatedAt" = NOW()
            RETURNING id
            """,
            (str(uuid.uuid4()), src, tgt, score,
             round(score * 5, 2), round((1 - score) * 5, 2)),
        )
        row = cur.fetchone()
        if row:
            cur.execute(
                """
                INSERT INTO "ProvenanceEntry"
                  (id, "edgeId", "agentReasoningSnapshot",
                   "simulationSource",
                   "evidenceWeight", "observedOutcome", timestamp)
                VALUES (%s, %s, %s, 'UNIPROT_STRING'::"SimulationSource", 1.0, 'SUPPORT'::"ObservedOutcome", NOW())
                """,
                (str(uuid.uuid4()), row[0], reasoning),
            )

# ── Main ──────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-score", type=float, default=0.4,
                        help="Minimum STRING combined score (0–1)")
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

    try:
        # ── Get all protein nodes from the DB ─────────────────
        if args.dry_run:
            proteins = [("EGFR", "EGFR"), ("BCR-ABL", "ABL1"),
                        ("COX-2", "PTGS2"), ("ERBB2", "ERBB2"), ("BRAF", "BRAF")]
        else:
            proteins = get_all_protein_ids(cur)

        gene_names = [name for _, name in proteins]
        id_by_name = {name.upper(): nid for nid, name in proteins}

        print(f"Resolving {len(gene_names)} proteins against STRING…")
        string_map = resolve_string_ids(gene_names)
        print(f"  Resolved {len(string_map)} / {len(gene_names)}")

        if len(string_map) < 2:
            print("Not enough proteins resolved — run ingest_chembl.py first.")
            return

        string_ids = list(string_map.values())
        print(f"Fetching STRING network (min_score={args.min_score})…")
        interactions = get_ppi_network(string_ids, args.min_score)
        print(f"  {len(interactions)} interactions returned")

        edges_written = 0
        for inter in interactions:
            # Use preferredName_A/B directly — avoids STRING ID round-trip mismatch
            gene_a = inter.get("preferredName_A", "").upper()
            gene_b = inter.get("preferredName_B", "").upper()
            score  = inter.get("score", 0)  # STRING JSON already returns 0-1 float

            node_a = id_by_name.get(gene_a)
            node_b = id_by_name.get(gene_b)

            if not node_a or not node_b or node_a == node_b:
                continue
            if score < args.min_score:
                continue

            reasoning = (
                f"STRING functional similarity: {gene_a} ~ {gene_b} "
                f"(combined score={score:.3f}, human proteome)"
            )
            upsert_similar_to(cur, node_a, node_b, score, reasoning, args.dry_run)
            edges_written += 1
            time.sleep(0.01)

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
    print(f"  SIMILAR_TO edges written : {edges_written * 2} (both directions)")
    print(f"\nRun propagation to infer new drug–protein edges:")
    print(f"  curl -X POST http://localhost:3000/api/propagate")

if __name__ == "__main__":
    main()
