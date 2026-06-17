# pyright: reportMissingImports=false
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from molforge.docking.posebench import PoseBenchTarget, run_posebench_gate  # noqa: E402


def write_pose(path: Path, coords: list[tuple[float, float, float]]) -> None:
    lines = []
    for index, (x_coord, y_coord, z_coord) in enumerate(coords, start=1):
        lines.append(
            f"ATOM  {index:5d}  C   LIG A   1      {x_coord:8.3f}{y_coord:8.3f}{z_coord:8.3f}  1.00  0.00           C"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_run_posebench_gate_writes_artifacts_and_applies_thresholds(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "phase2-posebench"
    reference_coords = [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]
    target_specs = [
        ("target-a", [(0.5, 0.0, 0.0), (1.5, 1.0, 1.0)]),
        ("target-b", [(1.0, 0.0, 0.0), (2.0, 1.0, 1.0)]),
        ("target-c", [(5.0, 0.0, 0.0), (6.0, 1.0, 1.0)]),
    ]

    targets: list[PoseBenchTarget] = []
    for target_name, docked_coords in target_specs:
        reference_path = tmp_path / f"{target_name}-ref.pdbqt"
        docked_path = tmp_path / f"{target_name}-dock.pdbqt"
        write_pose(reference_path, reference_coords)
        write_pose(docked_path, docked_coords)
        targets.append(
            PoseBenchTarget(
                name=target_name,
                reference_pose_path=reference_path,
                docked_pose_path=docked_path,
            )
        )

    result = run_posebench_gate(targets, output_dir=output_dir)

    assert result.target_names == ("target-a", "target-b", "target-c")
    assert result.mean_rmsd < 4.0
    assert result.rmsd_under_2a_rate == 1.0
    assert result.hard_gate_passed is True
    assert result.soft_gate_passed is True
    assert result.status == "hard_pass"

    for target_name, _coords in target_specs:
        target_output_dir = output_dir / target_name
        assert (target_output_dir / "pose.pdbqt").exists()
        rmsd_payload = json.loads(
            (target_output_dir / "rmsd.json").read_text(encoding="utf-8")
        )
        assert rmsd_payload["target"] == target_name


def test_run_posebench_gate_returns_soft_pass_at_one_third_rate(tmp_path: Path) -> None:
    """Rate exactly 1/3 must land in soft_pass (not blocked, not hard_pass).

    Use 3-atom geometries where the divergence is a shape distortion
    (not just translation) so Kabsch alignment still produces non-zero RMSD.
    """
    output_dir = tmp_path / "phase2-posebench-soft"
    reference_coords = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    # Aligned RMSD grows with shape mismatch (Kabsch cancels translation+rotation).
    # Use asymmetric distortions to control the aligned RMSD directly.
    target_specs = [
        (
            "target-good",
            [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (0.0, 1.2, 0.0)],
        ),  # aligned RMSD < 2Å
        (
            "target-mid",
            [(0.0, 0.0, 0.0), (6.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        ),  # one atom far off -> aligned RMSD 2-3Å
        (
            "target-bad",
            [(0.0, 0.0, 0.0), (8.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        ),  # one atom even farther
    ]
    targets: list[PoseBenchTarget] = []
    for target_name, docked_coords in target_specs:
        reference_path = tmp_path / f"{target_name}-ref.pdbqt"
        docked_path = tmp_path / f"{target_name}-dock.pdbqt"
        write_pose(reference_path, reference_coords)
        write_pose(docked_path, docked_coords)
        targets.append(
            PoseBenchTarget(
                name=target_name,
                reference_pose_path=reference_path,
                docked_pose_path=docked_path,
            )
        )
    result = run_posebench_gate(targets, output_dir=output_dir)
    assert result.rmsd_under_2a_rate == 1 / 3
    assert result.mean_rmsd < 4.0
    assert result.hard_gate_passed is False
    assert result.soft_gate_passed is True
    assert result.status == "soft_pass"


def test_run_posebench_gate_blocks_when_no_target_under_2a(tmp_path: Path) -> None:
    output_dir = tmp_path / "phase2-posebench-blocked"
    reference_coords = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    target_specs = [
        ("t1", [(0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (0.0, 5.0, 0.0)]),
        ("t2", [(0.0, 0.0, 0.0), (6.0, 0.0, 0.0), (0.0, 6.0, 0.0)]),
        ("t3", [(0.0, 0.0, 0.0), (7.0, 0.0, 0.0), (0.0, 7.0, 0.0)]),
    ]
    targets: list[PoseBenchTarget] = []
    for target_name, docked_coords in target_specs:
        reference_path = tmp_path / f"{target_name}-ref.pdbqt"
        docked_path = tmp_path / f"{target_name}-dock.pdbqt"
        write_pose(reference_path, reference_coords)
        write_pose(docked_path, docked_coords)
        targets.append(
            PoseBenchTarget(
                name=target_name,
                reference_pose_path=reference_path,
                docked_pose_path=docked_path,
            )
        )
    result = run_posebench_gate(targets, output_dir=output_dir)
    assert result.rmsd_under_2a_rate == 0.0
    assert result.hard_gate_passed is False
    assert result.soft_gate_passed is False
    assert result.status == "blocked"
