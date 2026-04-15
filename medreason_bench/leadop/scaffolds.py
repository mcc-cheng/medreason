"""RDKit Murcko-scaffold + descriptor pipeline for the lead-op harness.

Isolated in this module so swapping out the descriptor set later (or
adding fingerprints) doesn't ripple across the harness.
"""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski
from rdkit.Chem.Scaffolds import MurckoScaffold


@dataclass(frozen=True)
class Descriptors5:
    mw: float
    clogp: float
    tpsa: float
    hbd: int
    hba: int


def _mol(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Unparseable SMILES: {smiles!r}")
    return mol


def murcko_scaffold_smiles(smiles: str) -> str:
    mol = _mol(smiles)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold, canonical=True)


def compute_descriptors(smiles: str) -> Descriptors5:
    mol = _mol(smiles)
    return Descriptors5(
        mw=float(Descriptors.MolWt(mol)),
        clogp=float(Crippen.MolLogP(mol)),
        tpsa=float(Descriptors.TPSA(mol)),
        hbd=int(Lipinski.NumHDonors(mol)),
        hba=int(Lipinski.NumHAcceptors(mol)),
    )


def scaffold_key_and_descriptors(smiles: str) -> tuple[str, Descriptors5]:
    return murcko_scaffold_smiles(smiles), compute_descriptors(smiles)
