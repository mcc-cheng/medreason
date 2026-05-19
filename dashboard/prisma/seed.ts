import { PrismaClient, NodeType, DangerLevel, EdgeInteractionType, SimulationSource, ObservedOutcome } from '@prisma/client';

const prisma = new PrismaClient();

async function main() {
  // Seed mock nodes
  await prisma.node.upsert({
    where: { id: 'GLOW-SQUID-9' },
    update: {},
    create: {
      id: 'GLOW-SQUID-9',
      name: 'Glow Squid Bioluminescent Receptor',
      type: NodeType.BIOMOLECULE,
      molecularWeight: 42300,
      dangerLevel: DangerLevel.LOW,
      originDepartment: 'Department of Mad Science and Taco Logistics',
    },
  });

  await prisma.node.upsert({
    where: { id: 'HONEY-BADGER-X' },
    update: {},
    create: {
      id: 'HONEY-BADGER-X',
      name: 'Honey Badger Resilience Structural Protein',
      type: NodeType.BIOMOLECULE,
      molecularWeight: 88500,
      dangerLevel: DangerLevel.CHAOTIC,
      originDepartment: 'Department of Mad Science and Taco Logistics',
    },
  });

  await prisma.node.upsert({
    where: { id: 'CHIPOTLE-MAYO-42' },
    update: {},
    create: {
      id: 'CHIPOTLE-MAYO-42',
      name: 'Chipotle Mayo Small Molecule Compound',
      type: NodeType.CHEMICAL_CANDIDATE,
      molecularWeight: 312.4,
      dangerLevel: DangerLevel.MEDIUM,
      originDepartment: 'Department of Mad Science and Taco Logistics',
    },
  });

  await prisma.node.upsert({
    where: { id: 'CAFFEINE-OVERDOSE-99' },
    update: {},
    create: {
      id: 'CAFFEINE-OVERDOSE-99',
      name: 'Caffeine Overdose Synthetic Stimulant Derivative',
      type: NodeType.CHEMICAL_CANDIDATE,
      molecularWeight: 194.19,
      dangerLevel: DangerLevel.CHAOTIC,
      originDepartment: 'Department of Mad Science and Taco Logistics',
    },
  });

  // Seed a mock edge with one provenance entry
  const edge = await prisma.edge.upsert({
    where: {
      sourceId_targetId_interactionType: {
        sourceId: 'CHIPOTLE-MAYO-42',
        targetId: 'GLOW-SQUID-9',
        interactionType: EdgeInteractionType.AGGRESSIVELY_TICKLES,
      },
    },
    update: {},
    create: {
      sourceId: 'CHIPOTLE-MAYO-42',
      targetId: 'GLOW-SQUID-9',
      interactionType: EdgeInteractionType.AGGRESSIVELY_TICKLES,
      confidenceScore: 0.72,
      observationCount: 3,
      priorAlpha: 3,
      priorBeta: 1,
    },
  });

  await prisma.provenanceEntry.create({
    data: {
      edgeId: edge.id,
      agentReasoningSnapshot: 'Seed: initial interaction observed in mock simulation run #1.',
      simulationSource: SimulationSource.MOCK_TOOL,
      evidenceWeight: 1.0,
      observedOutcome: ObservedOutcome.SUPPORT,
    },
  });

  console.log('Seed complete.');
}

main()
  .catch((e) => { console.error(e); process.exit(1); })
  .finally(() => prisma.$disconnect());
