from __future__ import annotations

import importlib
import re
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers

from contracts.schema import BindingPocket, DockingPose

from ._env_check import require_vina_binary

VINA_RESULT_PATTERN = re.compile(
    r"^REMARK VINA RESULT:\s*(?P<affinity>-?\d+(?:\.\d+)?)"
)


def smiles_to_pdbqt(smiles: str, out_path: Path) -> Path:
    meeko = _require_meeko_module()
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES for PDBQT conversion: {smiles}")
    molecule = _select_largest_fragment(molecule)
    molecule = Chem.AddHs(molecule)
    embed_status = rdDistGeom.EmbedMolecule(molecule, randomSeed=0xF00D)
    if embed_status != 0:
        raise RuntimeError(
            f"Failed to embed 3D coordinates for ligand SMILES: {smiles}"
        )
    _ = rdForceFieldHelpers.UFFOptimizeMolecule(molecule)
    preparation = meeko.MoleculePreparation()
    setups = preparation.prepare(molecule)
    if not setups:
        raise RuntimeError(
            "Meeko did not produce a ligand setup for the provided SMILES."
        )
    pdbqt_string, is_ok, error_message = meeko.PDBQTWriterLegacy.write_string(setups[0])
    if not is_ok:
        raise RuntimeError(f"Failed to write ligand PDBQT: {error_message}")
    out_path.write_text(pdbqt_string, encoding="utf-8")
    return out_path


def structure_to_pdbqt(pdb_path: Path, out_path: Path) -> Path:
    _ = _require_meeko_module()
    receptor_cli = shutil.which("mk_prepare_receptor.py")
    if receptor_cli is None:
        raise RuntimeError(
            "Meeko receptor preparation CLI is required. Install docking extras so `mk_prepare_receptor.py` is available."
        )
    command = [
        receptor_cli,
        "--read_pdb",
        str(pdb_path),
        "--write_pdbqt",
        str(out_path),
        "--allow_bad_res",
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        shell=False,
    )
    if result.returncode != 0:
        _write_fallback_receptor_pdbqt(pdb_path, out_path)
        return out_path
    if not out_path.exists():
        raise RuntimeError(
            f"Meeko receptor preparation did not produce output file: {out_path}"
        )
    return out_path


def _write_fallback_receptor_pdbqt(pdb_path: Path, out_path: Path) -> None:
    atom_lines = [
        line
        for line in pdb_path.read_text(encoding="utf-8").splitlines()
        if line.startswith("ATOM")
    ]
    if not atom_lines:
        raise RuntimeError(
            f"Fallback receptor preparation requires ATOM records: {pdb_path}"
        )

    pdbqt_lines: list[str] = []
    for index, line in enumerate(atom_lines, start=1):
        atom_name = line[12:16]
        altloc = line[16:17]
        residue_name = line[17:20]
        chain_id = line[21:22]
        residue_seq = line[22:26]
        insertion_code = line[26:27]
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
        occupancy = float(line[54:60] or 1.0)
        b_factor = float(line[60:66] or 0.0)
        element = (line[76:78].strip() or atom_name.strip()[0]).upper()
        atom_type = _fallback_atom_type(element)
        pdbqt_lines.append(
            f"ATOM  {index:5d} {atom_name:<4s}{altloc:1s}{residue_name:>3s} {chain_id:1s}{residue_seq:>4s}{insertion_code:1s}   "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{b_factor:6.2f}{0.0:11.3f} {atom_type:<2s}"
        )
    out_path.write_text("\n".join(pdbqt_lines) + "\n", encoding="utf-8")


def _fallback_atom_type(element: str) -> str:
    normalized = element.upper()
    if normalized == "O":
        return "OA"
    if normalized == "N":
        return "N"
    if normalized == "S":
        return "SA"
    if normalized == "P":
        return "P"
    if normalized == "H":
        return "HD"
    if normalized == "F":
        return "F"
    if normalized == "CL":
        return "Cl"
    if normalized == "BR":
        return "Br"
    if normalized == "I":
        return "I"
    return "C"


def dock(
    pocket: BindingPocket,
    ligand_smiles: str,
    *,
    exhaustiveness: int = 8,
    output_dir: Path | None = None,
) -> list[DockingPose]:
    vina_binary = require_vina_binary()
    structure_path = Path(pocket.structure.pdb_path)

    with tempfile.TemporaryDirectory(prefix="molforge-vina-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        receptor_path = structure_to_pdbqt(
            structure_path,
            tmp_path / "receptor.pdbqt",
        )
        ligand_path = smiles_to_pdbqt(ligand_smiles, tmp_path / "ligand.pdbqt")
        if output_dir is None:
            output_path = tmp_path / "poses.pdbqt"
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"pose-{uuid.uuid4().hex}.pdbqt"
        command = [
            str(vina_binary),
            "--receptor",
            str(receptor_path),
            "--ligand",
            str(ligand_path),
            "--center_x",
            f"{pocket.center_xyz[0]:.3f}",
            "--center_y",
            f"{pocket.center_xyz[1]:.3f}",
            "--center_z",
            f"{pocket.center_xyz[2]:.3f}",
            "--size_x",
            f"{pocket.size_xyz[0]:.3f}",
            "--size_y",
            f"{pocket.size_xyz[1]:.3f}",
            "--size_z",
            f"{pocket.size_xyz[2]:.3f}",
            "--exhaustiveness",
            str(exhaustiveness),
            "--out",
            str(output_path),
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            shell=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Vina docking failed: {result.stderr.strip() or result.stdout.strip()}"
            )
        poses = parse_vina_output(
            output_path.read_text(encoding="utf-8"), pocket, ligand_smiles
        )
        return [
            DockingPose(
                ligand_smiles=pose.ligand_smiles,
                pocket=pose.pocket,
                pose_pdb_path=str(output_path),
                vina_score=pose.vina_score,
                rank=pose.rank,
            )
            for pose in poses
        ]


def parse_vina_output(
    pdbqt_text: str,
    pocket: BindingPocket,
    ligand_smiles: str,
) -> list[DockingPose]:
    poses: list[DockingPose] = []
    for line in pdbqt_text.splitlines():
        match = VINA_RESULT_PATTERN.match(line.strip())
        if not match:
            continue
        poses.append(
            DockingPose(
                ligand_smiles=ligand_smiles,
                pocket=pocket,
                pose_pdb_path="",
                vina_score=float(match.group("affinity")),
                rank=len(poses) + 1,
            )
        )
    if not poses:
        raise ValueError("Vina output did not contain any REMARK VINA RESULT lines.")
    return poses[:9]


def _require_meeko_module():
    try:
        return importlib.import_module("meeko")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Meeko and its import-time dependencies are required for PDBQT conversion. "
            "Install them with `uv sync --extra docking`."
        ) from exc


def _select_largest_fragment(molecule: Chem.Mol) -> Chem.Mol:
    fragments = Chem.GetMolFrags(molecule, asMols=True, sanitizeFrags=True)
    if not fragments:
        raise ValueError("Ligand SMILES did not produce any valid RDKit fragments.")
    return max(
        fragments,
        key=lambda fragment: (fragment.GetNumHeavyAtoms(), fragment.GetNumAtoms()),
    )
