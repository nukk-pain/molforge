from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class PoseBenchTarget:
    name: str
    reference_pose_path: Path
    docked_pose_path: Path


@dataclass(frozen=True, slots=True)
class PoseBenchGateResult:
    target_names: tuple[str, ...]
    mean_rmsd: float
    rmsd_under_2a_rate: float
    hard_gate_passed: bool
    soft_gate_passed: bool
    status: str  # "hard_pass" | "soft_pass" | "blocked"


def run_posebench_gate(
    targets: list[PoseBenchTarget],
    *,
    output_dir: Path,
) -> PoseBenchGateResult:
    if len(targets) < 3:
        raise ValueError("PoseBench gate requires at least three targets.")

    target_names: list[str] = []
    rmsd_values: list[float] = []
    under_2a_count = 0

    for target in targets:
        rmsd = calculate_pose_rmsd(target.reference_pose_path, target.docked_pose_path)
        target_names.append(target.name)
        rmsd_values.append(rmsd)
        if rmsd < 2.0:
            under_2a_count += 1

        target_output_dir = output_dir / target.name
        target_output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(target.docked_pose_path, target_output_dir / "pose.pdbqt")
        (target_output_dir / "rmsd.json").write_text(
            json.dumps(
                {
                    "target": target.name,
                    "rmsd": rmsd,
                    "under_2a": rmsd < 2.0,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    mean_rmsd = sum(rmsd_values) / len(rmsd_values)
    rmsd_under_2a_rate = under_2a_count / len(rmsd_values)
    hard_gate_passed = mean_rmsd < 4.0 and rmsd_under_2a_rate >= (2 / 3)
    soft_gate_passed = rmsd_under_2a_rate >= (1 / 3)
    if hard_gate_passed:
        status = "hard_pass"
    elif soft_gate_passed:
        status = "soft_pass"
    else:
        status = "blocked"
    return PoseBenchGateResult(
        target_names=tuple(target_names),
        mean_rmsd=mean_rmsd,
        rmsd_under_2a_rate=rmsd_under_2a_rate,
        hard_gate_passed=hard_gate_passed,
        soft_gate_passed=soft_gate_passed,
        status=status,
    )


def calculate_pose_rmsd(reference_pose_path: Path, docked_pose_path: Path) -> float:
    reference_coords = _extract_pose_coordinates(reference_pose_path)
    docked_coords = _extract_pose_coordinates(docked_pose_path)
    if len(reference_coords) != len(docked_coords):
        raise ValueError("Reference and docked poses must have the same atom count.")
    if not reference_coords:
        raise ValueError("Pose RMSD calculation requires at least one atom coordinate.")

    return _aligned_rmsd(reference_coords, docked_coords)


def _aligned_rmsd(
    reference_coords: list[tuple[float, float, float]],
    docked_coords: list[tuple[float, float, float]],
) -> float:
    reference = np.asarray(reference_coords, dtype=float)
    docked = np.asarray(docked_coords, dtype=float)
    reference_center = reference.mean(axis=0)
    docked_center = docked.mean(axis=0)
    reference_centered = reference - reference_center
    docked_centered = docked - docked_center
    covariance = docked_centered.T @ reference_centered
    left, _singular_values, right_t = np.linalg.svd(covariance)
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0:
        right_t[-1, :] *= -1
        rotation = right_t.T @ left.T
    aligned = docked_centered @ rotation
    deltas = reference_centered - aligned
    squared = np.sum(deltas * deltas, axis=1)
    return math.sqrt(float(np.mean(squared)))


def _extract_pose_coordinates(path: Path) -> list[tuple[float, float, float]]:
    coords: list[tuple[float, float, float]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    has_model_records = any(line.startswith("MODEL") for line in lines)
    in_model = False
    for line in lines:
        if line.startswith("MODEL"):
            if coords:
                break
            in_model = True
            continue
        if line.startswith("ENDMDL"):
            if coords:
                break
            in_model = False
            continue
        if not line.startswith(("ATOM", "HETATM")):
            continue
        if has_model_records and not in_model:
            continue
        atom_type = line[77:79].strip().upper() if len(line) >= 79 else ""
        element = line[76:78].strip().upper() if len(line) >= 78 else ""
        atom_name = line[12:16].strip().upper()
        if atom_type.startswith("H") or element == "H" or atom_name.startswith("H"):
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
        except ValueError as exc:
            parts = line.split()
            if len(parts) >= 9:
                try:
                    coords.append((float(parts[6]), float(parts[7]), float(parts[8])))
                    continue
                except ValueError:
                    pass
            raise ValueError(
                f"Failed to parse pose coordinate line in {path}: {line}"
            ) from exc
    return coords
