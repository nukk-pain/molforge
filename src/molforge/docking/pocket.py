from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from contracts.schema import BindingPocket, ProteinStructure

from ._env_check import which_binary

MIN_POCKET_SIZE = 4.0
FPOCKET_SCORE_PATTERN = re.compile(r"^(?P<key>.+?):\s*(?P<value>-?\d+(?:\.\d+)?)\s*$")


def detect_pocket(
    structure: ProteinStructure,
    *,
    use_fpocket: bool = True,
) -> BindingPocket:
    if use_fpocket and which_binary("fpocket") is not None:
        pocket = _detect_with_fpocket(structure)
        if pocket is not None:
            return pocket
    return _detect_with_geometry(structure)


def _detect_with_fpocket(structure: ProteinStructure) -> BindingPocket | None:
    structure_path = Path(structure.pdb_path)
    output_dir = structure_path.with_suffix("")
    output_dir = output_dir.with_name(f"{output_dir.name}_out")

    _ = subprocess.run(
        [str(which_binary("fpocket")), "-f", str(structure_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    info_path = output_dir / f"{structure_path.stem}_info.txt"
    pockets_dir = output_dir / "pockets"
    vert_path = pockets_dir / "pocket1_vert.pqr"
    atom_path = pockets_dir / "pocket1_atm.pdb"
    if not info_path.exists() or not vert_path.exists() or not atom_path.exists():
        return None

    score_payload = _parse_fpocket_info(info_path)
    center_xyz, size_xyz = _parse_fpocket_vertices(vert_path)
    residues = _parse_pocket_residues(atom_path)

    return BindingPocket(
        structure=structure,
        center_xyz=center_xyz,
        size_xyz=size_xyz,
        druggability_score=score_payload.get("druggability_score"),
        residues=residues,
    )


def _detect_with_geometry(structure: ProteinStructure) -> BindingPocket:
    coords = _extract_atom_coordinates(
        Path(structure.pdb_path).read_text(encoding="utf-8")
    )
    if not coords:
        raise ValueError(f"No ATOM coordinates found in {structure.pdb_path}.")

    xs = [coord[0] for coord in coords]
    ys = [coord[1] for coord in coords]
    zs = [coord[2] for coord in coords]
    center_xyz = (
        sum(xs) / len(xs),
        sum(ys) / len(ys),
        sum(zs) / len(zs),
    )
    size_xyz = (
        max(MIN_POCKET_SIZE, max(xs) - min(xs)),
        max(MIN_POCKET_SIZE, max(ys) - min(ys)),
        max(MIN_POCKET_SIZE, max(zs) - min(zs)),
    )
    residues = _extract_residues_from_pdb(
        Path(structure.pdb_path).read_text(encoding="utf-8")
    )
    return BindingPocket(
        structure=structure,
        center_xyz=center_xyz,
        size_xyz=size_xyz,
        druggability_score=None,
        residues=residues,
    )


def _parse_fpocket_info(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    in_first_pocket = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("Pocket 1"):
            in_first_pocket = True
            continue
        if line.startswith("Pocket ") and not line.startswith("Pocket 1"):
            break
        if not in_first_pocket:
            continue
        match = FPOCKET_SCORE_PATTERN.match(line)
        if not match:
            continue
        key = match.group("key").strip().lower().replace(" ", "_")
        values[key] = float(match.group("value"))
    return values


def _parse_fpocket_vertices(
    path: Path,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    coords = _extract_atom_coordinates(path.read_text(encoding="utf-8"))
    if not coords:
        raise ValueError(f"fpocket vertex file {path} did not contain coordinates.")
    xs = [coord[0] for coord in coords]
    ys = [coord[1] for coord in coords]
    zs = [coord[2] for coord in coords]
    center_xyz = (
        sum(xs) / len(xs),
        sum(ys) / len(ys),
        sum(zs) / len(zs),
    )
    size_xyz = (
        max(MIN_POCKET_SIZE, max(xs) - min(xs)),
        max(MIN_POCKET_SIZE, max(ys) - min(ys)),
        max(MIN_POCKET_SIZE, max(zs) - min(zs)),
    )
    return center_xyz, size_xyz


def _parse_pocket_residues(path: Path) -> list[str]:
    return _extract_residues_from_pdb(path.read_text(encoding="utf-8"))


def _extract_atom_coordinates(pdb_text: str) -> list[tuple[float, float, float]]:
    coords: list[tuple[float, float, float]] = []
    for line in pdb_text.splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        try:
            coords.append(
                (
                    float(line[30:38].strip()),
                    float(line[38:46].strip()),
                    float(line[46:54].strip()),
                )
            )
            continue
        except ValueError:
            pass

        parts = line.split()
        if len(parts) >= 9:
            try:
                coords.append((float(parts[-5]), float(parts[-4]), float(parts[-3])))
            except ValueError:
                continue
    return coords


def _extract_residues_from_pdb(pdb_text: str) -> list[str]:
    residues: list[str] = []
    seen: set[str] = set()
    for line in pdb_text.splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        residue_name = line[17:20].strip()
        residue_id = line[22:26].strip()
        if not residue_name or not residue_id:
            continue
        residue = f"{residue_name}{residue_id}"
        if residue in seen:
            continue
        seen.add(residue)
        residues.append(residue)
    return residues
