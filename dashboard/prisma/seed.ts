import {
  PrismaClient,
  NodeType,
  DangerLevel,
  EdgeInteractionType,
  SimulationSource,
  ObservedOutcome,
} from '@prisma/client';

const prisma = new PrismaClient();

// ─── Real drug-protein interactions ───────────────────────────
// BCR-ABL / Imatinib  → classic targeted oncology (CML)
// EGFR    / Gefitinib → EGFR+ non-small cell lung cancer
// COX-2   / Aspirin   → COX-2 irreversible acetylation (anti-inflammatory)

async function main() {
  // ── Proteins (BIOMOLECULE) ──────────────────────────────────

  await prisma.node.upsert({
    where: { id: 'BCR-ABL' },
    update: {},
    create: {
      id: 'BCR-ABL',
      name: 'BCR-ABL fusion tyrosine kinase',
      type: NodeType.BIOMOLECULE,
      molecularWeight: 210000,   // ~210 kDa fusion protein
      dangerLevel: DangerLevel.MEDIUM,
      originDepartment: 'Computational Biology Lab',
    },
  });

  await prisma.node.upsert({
    where: { id: 'EGFR' },
    update: {},
    create: {
      id: 'EGFR',
      name: 'Epidermal Growth Factor Receptor (EGFR/HER1)',
      type: NodeType.BIOMOLECULE,
      molecularWeight: 134000,   // ~134 kDa transmembrane receptor
      dangerLevel: DangerLevel.MEDIUM,
      originDepartment: 'Computational Biology Lab',
    },
  });

  await prisma.node.upsert({
    where: { id: 'COX-2' },
    update: {},
    create: {
      id: 'COX-2',
      name: 'Cyclooxygenase-2 (PTGS2)',
      type: NodeType.BIOMOLECULE,
      molecularWeight: 70000,    // ~70 kDa homodimer subunit
      dangerLevel: DangerLevel.LOW,
      originDepartment: 'Computational Biology Lab',
    },
  });

  // ── Compounds (CHEMICAL_CANDIDATE) ──────────────────────────

  await prisma.node.upsert({
    where: { id: 'IMATINIB' },
    update: {},
    create: {
      id: 'IMATINIB',
      name: 'Imatinib (Gleevec / STI-571)',
      type: NodeType.CHEMICAL_CANDIDATE,
      molecularWeight: 493.6,    // free base, Da
      dangerLevel: DangerLevel.LOW,
      originDepartment: 'Computational Biology Lab',
    },
  });

  await prisma.node.upsert({
    where: { id: 'GEFITINIB' },
    update: {},
    create: {
      id: 'GEFITINIB',
      name: 'Gefitinib (Iressa / ZD1839)',
      type: NodeType.CHEMICAL_CANDIDATE,
      molecularWeight: 446.9,    // Da
      dangerLevel: DangerLevel.MEDIUM,
      originDepartment: 'Computational Biology Lab',
    },
  });

  await prisma.node.upsert({
    where: { id: 'ASPIRIN' },
    update: {},
    create: {
      id: 'ASPIRIN',
      name: 'Aspirin (Acetylsalicylic acid)',
      type: NodeType.CHEMICAL_CANDIDATE,
      molecularWeight: 180.2,    // Da
      dangerLevel: DangerLevel.LOW,
      originDepartment: 'Computational Biology Lab',
    },
  });

  // ── Edges (known interactions with prior confidence) ─────────

  // Imatinib → BCR-ABL: well-validated ATP-pocket inhibitor (high confidence)
  const edgeImatinibBcrAbl = await prisma.edge.upsert({
    where: {
      sourceId_targetId_interactionType: {
        sourceId: 'IMATINIB',
        targetId: 'BCR-ABL',
        interactionType: EdgeInteractionType.INHIBITS,
      },
    },
    update: {},
    create: {
      sourceId: 'IMATINIB',
      targetId: 'BCR-ABL',
      interactionType: EdgeInteractionType.INHIBITS,
      confidenceScore: 0.96,
      observationCount: 12,
      priorAlpha: 12,
      priorBeta: 1,
    },
  });

  await prisma.provenanceEntry.create({
    data: {
      edgeId: edgeImatinibBcrAbl.id,
      agentReasoningSnapshot:
        'Seed: Imatinib binds the inactive conformation of BCR-ABL ATP pocket (IC50 ≈ 0.025 µM). FDA-approved for CML. High prior confidence from literature.',
      simulationSource: SimulationSource.RAG_LITERATURE,
      evidenceWeight: 1.0,
      observedOutcome: ObservedOutcome.SUPPORT,
    },
  });

  // Gefitinib → EGFR: reversible ATP-competitive inhibitor
  const edgeGefitinibEgfr = await prisma.edge.upsert({
    where: {
      sourceId_targetId_interactionType: {
        sourceId: 'GEFITINIB',
        targetId: 'EGFR',
        interactionType: EdgeInteractionType.INHIBITS,
      },
    },
    update: {},
    create: {
      sourceId: 'GEFITINIB',
      targetId: 'EGFR',
      interactionType: EdgeInteractionType.INHIBITS,
      confidenceScore: 0.91,
      observationCount: 8,
      priorAlpha: 8,
      priorBeta: 1,
    },
  });

  await prisma.provenanceEntry.create({
    data: {
      edgeId: edgeGefitinibEgfr.id,
      agentReasoningSnapshot:
        'Seed: Gefitinib reversibly inhibits EGFR tyrosine kinase by competing at the ATP binding site (IC50 ≈ 0.033 µM). FDA-approved for EGFR-mutant NSCLC.',
      simulationSource: SimulationSource.RAG_LITERATURE,
      evidenceWeight: 1.0,
      observedOutcome: ObservedOutcome.SUPPORT,
    },
  });

  // Aspirin → COX-2: irreversible covalent acetylation at Ser516
  const edgeAspirinCox2 = await prisma.edge.upsert({
    where: {
      sourceId_targetId_interactionType: {
        sourceId: 'ASPIRIN',
        targetId: 'COX-2',
        interactionType: EdgeInteractionType.INHIBITS,
      },
    },
    update: {},
    create: {
      sourceId: 'ASPIRIN',
      targetId: 'COX-2',
      interactionType: EdgeInteractionType.INHIBITS,
      confidenceScore: 0.89,
      observationCount: 6,
      priorAlpha: 6,
      priorBeta: 1,
    },
  });

  await prisma.provenanceEntry.create({
    data: {
      edgeId: edgeAspirinCox2.id,
      agentReasoningSnapshot:
        'Seed: Aspirin irreversibly acetylates COX-2 Ser516, blocking arachidonic acid access and preventing prostaglandin synthesis. Mechanism confirmed by X-ray crystallography.',
      simulationSource: SimulationSource.RAG_LITERATURE,
      evidenceWeight: 1.0,
      observedOutcome: ObservedOutcome.SUPPORT,
    },
  });

  console.log('Seed complete — 6 nodes, 3 edges seeded.');
}

main()
  .catch((e) => { console.error(e); process.exit(1); })
  .finally(() => prisma.$disconnect());
