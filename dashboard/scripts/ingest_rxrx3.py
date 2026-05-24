"""
ingest_rxrx3.py — Load rxrx3-core into the Drug Discovery Canvas knowledge graph.

The dataset has two perturbation types:
  - COMPOUND: small molecule applied to HUVEC cells → phenotypic embedding captured
  - CRISPR:   gene knockout applied to HUVEC cells  → phenotypic embedding captured

Compound→gene interaction scores are derived by cosine similarity between the
mean phenotypic embedding of each compound and each gene knockout. When a compound
produces a cellular phenotype similar to knocking out a gene, it suggests that
compound acts (at least partially) through that gene's pathway.

Downloads only the metadata CSV and OpenPhenom embeddings (~510 MB total).
The 18 GB image corpus is skipped entirely.

Usage:
    python dashboard/scripts/ingest_rxrx3.py [--top-k 5] [--min-sim 0.5] [--dry-run]

Flags:
    --top-k N      top-N gene targets per compound to insert as edges (default 5)
    --min-sim F    minimum cosine similarity to include (default 0.5)
    --dry-run      print plan without writing to DB
    --max-compounds N  cap compounds ingested (default: all 1674)
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import uuid
from pathlib import Path


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_env() -> str:
    for candidate in [
        Path(__file__).parent.parent / ".env.local",
        Path(__file__).parent.parent / ".env",
    ]:
        if candidate.exists():
            for line in candidate.read_text().splitlines():
                line = line.strip()
                if line.startswith("DATABASE_URL="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        sys.exit("DATABASE_URL not found in dashboard/.env or dashboard/.env.local")
    return url


def sanitize_id(s: str, max_len: int = 48) -> str:
    return re.sub(r"[^A-Z0-9_\-]", "-", s.upper().strip())[:max_len]


def cosine_sim_matrix(A, B):
    """Return (n_A, n_B) cosine similarity matrix. A: (n,d), B: (m,d)."""
    import numpy as np
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return A_norm @ B_norm.T   # (n, m)


def sim_to_confidence(sim: float) -> float:
    """Map cosine similarity [-1,1] → Bayesian confidence [0,1]."""
    return round(max(0.0, float(sim)), 4)


def upsert_node(cur, node_id, name, node_type, smiles, external_id):
    cur.execute(
        """
        INSERT INTO "Node"
          (id, name, type, "molecularWeight", "dangerLevel", "originDepartment",
           smiles, "externalId", "createdAt", "updatedAt")
        VALUES (%s, %s, %s::"NodeType", NULL, NULL, %s, %s, %s, NOW(), NOW())
        ON CONFLICT (id) DO UPDATE
          SET name         = EXCLUDED.name,
              smiles       = COALESCE(EXCLUDED.smiles, "Node".smiles),
              "externalId" = COALESCE(EXCLUDED."externalId", "Node"."externalId"),
              "updatedAt"  = NOW()
        """,
        (node_id, name, node_type, "rxrx3-core", smiles, external_id),
    )


def upsert_edge(cur, source_id, target_id, confidence, obs_count, reasoning):
    alpha = round(confidence * 10, 2)
    beta  = round(10 - alpha, 2)
    alpha = max(alpha, 0.1)
    beta  = max(beta,  0.1)

    cur.execute(
        """
        INSERT INTO "Edge"
          (id, "sourceId", "targetId", "interactionType"::"EdgeInteractionType",
           "confidenceScore", "observationCount",
           "priorAlpha", "priorBeta", "createdAt", "updatedAt")
        VALUES (%s, %s, %s, 'TARGETS', %s, %s, %s, %s, NOW(), NOW())
        ON CONFLICT ("sourceId", "targetId", "interactionType") DO UPDATE
          SET "confidenceScore"  = GREATEST("Edge"."confidenceScore", EXCLUDED."confidenceScore"),
              "observationCount" = EXCLUDED."observationCount",
              "priorAlpha"       = EXCLUDED."priorAlpha",
              "priorBeta"        = EXCLUDED."priorBeta",
              "updatedAt"        = NOW()
        RETURNING id
        """,
        (str(uuid.uuid4()), source_id, target_id, confidence, obs_count, alpha, beta),
    )
    row = cur.fetchone()
    if row:
        cur.execute(
            """
            INSERT INTO "ProvenanceEntry"
              (id, "edgeId", "agentReasoningSnapshot",
               "simulationSource"::"SimulationSource",
               "evidenceWeight", "observedOutcome"::"ObservedOutcome", timestamp)
            VALUES (%s, %s, %s, 'RXRX3_PHENOMICS', 1.0, 'SUPPORT', NOW())
            """,
            (str(uuid.uuid4()), row[0], reasoning),
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k",         type=int,   default=5,    help="Top-N genes per compound")
    parser.add_argument("--min-sim",        type=float, default=0.5,  help="Minimum cosine similarity")
    parser.add_argument("--dry-run",        action="store_true",      help="No DB writes")
    parser.add_argument("--max-compounds",  type=int,   default=None, help="Cap number of compounds")
    args = parser.parse_args()

    try:
        import numpy as np
        import pandas as pd
        import psycopg2
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        sys.exit(f"Missing dependency ({exc.name}). Run: pip install numpy pandas huggingface_hub pyarrow psycopg2-binary")

    # ── Download ──────────────────────────────────────────────────────────────

    print("Fetching metadata CSV from HuggingFace…")
    meta_path = hf_hub_download(
        "recursionpharma/rxrx3-core",
        filename="metadata_rxrx3_core.csv",
        repo_type="dataset",
    )

    print("Fetching OpenPhenom embeddings (510 MB, cached after first run)…")
    emb_path = hf_hub_download(
        "recursionpharma/rxrx3-core",
        filename="OpenPhenom_rxrx3_core_embeddings.parquet",
        repo_type="dataset",
    )

    # ── Load & split ──────────────────────────────────────────────────────────

    print("Loading metadata…")
    meta = pd.read_csv(meta_path, low_memory=False)

    # Strip control wells
    meta = meta[~meta["treatment"].str.contains("control|EMPTY", case=False, na=False)]

    compounds_meta = meta[meta["perturbation_type"] == "COMPOUND"].copy()
    genes_meta     = meta[meta["perturbation_type"] == "CRISPR"].copy()

    print(f"  {len(compounds_meta):,} compound wells · {len(genes_meta):,} CRISPR wells")
    print(f"  {compounds_meta['treatment'].nunique()} unique compounds · {genes_meta['gene'].nunique()} unique genes")

    print("Loading embeddings…")
    emb = pd.read_parquet(emb_path)
    feat_cols = [c for c in emb.columns if c.startswith("feature_")]

    # ── Average embeddings per compound ──────────────────────────────────────

    print("Computing mean embeddings per compound…")
    comp_emb = (
        compounds_meta[["well_id", "treatment", "SMILES"]]
        .merge(emb[["well_id"] + feat_cols], on="well_id", how="inner")
    )
    comp_mean = (
        comp_emb.groupby("treatment")[feat_cols].mean()
    )
    # Keep one SMILES per compound (first non-null)
    comp_smiles = (
        comp_emb.groupby("treatment")["SMILES"]
        .first()
        .fillna("")
    )

    if args.max_compounds:
        comp_mean = comp_mean.iloc[: args.max_compounds]

    # ── Average embeddings per gene ───────────────────────────────────────────

    print("Computing mean embeddings per gene…")
    gene_emb = (
        genes_meta[["well_id", "gene"]]
        .merge(emb[["well_id"] + feat_cols], on="well_id", how="inner")
    )
    gene_mean = gene_emb.groupby("gene")[feat_cols].mean()

    # ── Cosine similarity ─────────────────────────────────────────────────────

    print(f"Computing cosine similarity ({len(comp_mean)} compounds × {len(gene_mean)} genes)…")
    C = comp_mean.values.astype(np.float32)
    G = gene_mean.values.astype(np.float32)
    sim_matrix = cosine_sim_matrix(C, G)   # (n_compounds, n_genes)

    comp_names = comp_mean.index.tolist()
    gene_names = gene_mean.index.tolist()

    # ── Build edge list ───────────────────────────────────────────────────────

    print(f"Selecting top-{args.top_k} gene targets per compound (min_sim={args.min_sim})…")
    edges: list[tuple[str, str, float, int]] = []

    for ci, comp in enumerate(comp_names):
        sims = sim_matrix[ci]
        top_idx = np.argsort(sims)[::-1][: args.top_k]
        for gi in top_idx:
            s = float(sims[gi])
            if s < args.min_sim:
                break
            edges.append((comp, gene_names[gi], s, ci))

    print(f"  {len(edges):,} edges above threshold")

    # ── DB writes ─────────────────────────────────────────────────────────────

    if args.dry_run:
        print("\n[DRY RUN] Sample of what would be written:")
        for comp_name, gene_name, sim, _ in edges[:20]:
            cid = sanitize_id(comp_name)
            gid = sanitize_id(gene_name)
            print(f"  {cid} --[TARGETS conf={sim:.3f}]--> {gid}")
        print(f"\n  … {len(comp_mean)} compound nodes, {len(gene_mean)} gene nodes, {len(edges)} edges total")
        return

    db_url  = load_env()
    conn    = psycopg2.connect(db_url)
    cur     = conn.cursor()

    try:
        # Gene nodes
        print(f"\nUpserting {len(gene_mean)} gene nodes…")
        for gene in gene_names:
            gid = sanitize_id(gene)
            upsert_node(cur, gid, gene, "BIOMOLECULE", None, f"rxrx3:{gene}")

        # Compound nodes (with SMILES)
        print(f"Upserting {len(comp_mean)} compound nodes…")
        for comp in comp_names:
            cid    = sanitize_id(comp)
            smiles = comp_smiles.get(comp) or None
            if smiles == "":
                smiles = None
            upsert_node(cur, cid, comp, "CHEMICAL_CANDIDATE", smiles, f"rxrx3:{comp}")

        # Edges
        print(f"Upserting {len(edges)} phenotypic-similarity edges…")
        for i, (comp_name, gene_name, sim, _) in enumerate(edges):
            if i % 500 == 0:
                print(f"  {i}/{len(edges)}", end="\r")
            cid        = sanitize_id(comp_name)
            gid        = sanitize_id(gene_name)
            confidence = sim_to_confidence(sim)
            reasoning  = (
                f"rxrx3-core OpenPhenom phenotypic similarity: "
                f"{comp_name} → {gene_name} (cosine={sim:.4f}). "
                f"Compound's cellular phenotype resembles gene knockout phenotype, "
                f"suggesting functional interaction through this pathway."
            )
            upsert_edge(cur, cid, gid, confidence, 1, reasoning)

        conn.commit()
        print(f"\nDone. {len(gene_mean)} genes, {len(comp_mean)} compounds, {len(edges)} edges committed.")
        print("Refresh the graph at http://localhost:3000 to see the new data.")

    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()
