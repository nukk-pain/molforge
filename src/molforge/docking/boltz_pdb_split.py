"""Split Boltz-2 complex PDB into (ligand SDF, protein PDB) for PoseBusters.

Boltz-2 emits a single `input_model_<N>.pdb` per diffusion sample with the
protein on chain A (ATOM records) and the ligand on chain B (HETATM records,
residue name `LIG`). No MODEL/ENDMDL framing. PoseBusters requires the
ligand as a proper Mol (with bond orders) and the protein as a separate
PDB. We reconstruct ligand bond orders from the known SMILES via RDKit's
`AssignBondOrdersFromTemplate`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem


@dataclass(frozen=True, slots=True)
class SplitResult:
    ligand_sdf: Path
    protein_pdb: Path


def split_boltz_pdb(
    pdb_path: Path,
    ligand_smiles: str,
    out_dir: Path,
    *,
    ligand_chain: str = "B",
    stem: str | None = None,
) -> SplitResult:
    """Extract the ligand (chain B, HETATM) and protein (all non-ligand ATOM)
    from a Boltz-2 complex PDB. Ligand bond orders are restored from
    `ligand_smiles` via `AssignBondOrdersFromTemplate`.

    Raises RuntimeError with a descriptive message if any step fails, so the
    caller can record the failure against a specific pose instead of
    crashing the probe.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    base = stem or pdb_path.stem

    ligand_pdb_lines: list[str] = []
    protein_pdb_lines: list[str] = []
    with pdb_path.open() as handle:
        for line in handle:
            if line.startswith("HETATM") and len(line) >= 22 and line[21] == ligand_chain:
                ligand_pdb_lines.append(line)
            elif line.startswith("ATOM"):
                protein_pdb_lines.append(line)

    if not ligand_pdb_lines:
        raise RuntimeError(
            f"no HETATM records on chain {ligand_chain!r} in {pdb_path}"
        )
    if not protein_pdb_lines:
        raise RuntimeError(f"no ATOM records (protein) in {pdb_path}")

    protein_path = out_dir / f"{base}_protein.pdb"
    protein_path.write_text("".join(protein_pdb_lines) + "END\n", encoding="utf-8")

    ligand_pdb_path = out_dir / f"{base}_ligand.pdb"
    ligand_pdb_path.write_text(
        "".join(ligand_pdb_lines) + "END\n", encoding="utf-8"
    )

    ligand_mol_raw = Chem.MolFromPDBFile(
        str(ligand_pdb_path), sanitize=False, removeHs=False
    )
    if ligand_mol_raw is None:
        raise RuntimeError(
            f"RDKit could not parse ligand PDB lines from {pdb_path}"
        )

    template = Chem.MolFromSmiles(ligand_smiles)
    if template is None:
        raise RuntimeError(f"invalid template SMILES: {ligand_smiles!r}")

    try:
        ligand_mol = AllChem.AssignBondOrdersFromTemplate(template, ligand_mol_raw)
    except Exception as exc:  # noqa: BLE001 — RDKit raises bare Exception
        raise RuntimeError(
            f"AssignBondOrdersFromTemplate failed for {pdb_path}: {exc}"
        ) from exc

    try:
        Chem.SanitizeMol(ligand_mol)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"ligand sanitize failed for {pdb_path}: {exc}") from exc

    ligand_sdf_path = out_dir / f"{base}_ligand.sdf"
    writer = Chem.SDWriter(str(ligand_sdf_path))
    writer.write(ligand_mol)
    writer.close()

    return SplitResult(ligand_sdf=ligand_sdf_path, protein_pdb=protein_path)
