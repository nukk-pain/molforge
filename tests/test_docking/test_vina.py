# pyright: reportMissingImports=false
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
from rdkit import Chem

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from contracts.schema import BindingPocket, ProteinStructure, StructureSource  # noqa: E402
from molforge.docking import vina  # noqa: E402


def build_pocket() -> BindingPocket:
    return BindingPocket(
        structure=ProteinStructure(
            gene="CXCR4",
            uniprot="P61073",
            pdb_path="/tmp/cxcr4.pdb",
            source=StructureSource.ALPHAFOLD_DB,
            confidence=88.1,
        ),
        center_xyz=(1.0, 2.0, 3.0),
        size_xyz=(10.0, 11.0, 12.0),
        druggability_score=0.8,
        residues=["ASP97"],
    )


def test_dock_parses_vina_pdbqt_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    receptor_source = tmp_path / "cxcr4.pdb"
    receptor_source.write_text(
        "ATOM      1  CA  ASP A  97      10.000  11.000  12.000  1.00 80.00           C\n",
        encoding="utf-8",
    )
    pocket = build_pocket()
    pocket.structure.pdb_path = str(receptor_source)

    monkeypatch.setattr(
        vina, "require_vina_binary", lambda: Path("/usr/local/bin/vina")
    )

    def fake_smiles_to_pdbqt(smiles: str, out_path: Path) -> Path:
        _ = smiles
        out_path.write_text("LIGAND\n", encoding="utf-8")
        return out_path

    def fake_structure_to_pdbqt(pdb_path: Path, out_path: Path) -> Path:
        _ = pdb_path
        out_path.write_text("RECEPTOR\n", encoding="utf-8")
        return out_path

    monkeypatch.setattr(vina, "smiles_to_pdbqt", fake_smiles_to_pdbqt)
    monkeypatch.setattr(vina, "structure_to_pdbqt", fake_structure_to_pdbqt)

    def fake_run(command, capture_output, text, check, shell):
        output_path = Path(command[command.index("--out") + 1])
        output_path.write_text(
            "REMARK VINA RESULT:      -9.5      0.000      0.000\n"
            "MODEL 2\nREMARK VINA RESULT:      -8.2      1.000      1.500\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(vina.subprocess, "run", fake_run)

    poses = vina.dock(pocket, "CCO")

    assert [pose.vina_score for pose in poses] == [-9.5, -8.2]
    assert [pose.rank for pose in poses] == [1, 2]
    assert all(pose.pose_pdb_path.endswith("poses.pdbqt") for pose in poses)


def test_dock_persists_pose_file_when_output_dir_provided(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    receptor_source = tmp_path / "cxcr4.pdb"
    receptor_source.write_text(
        "ATOM      1  CA  ASP A  97      10.000  11.000  12.000  1.00 80.00           C\n",
        encoding="utf-8",
    )
    pocket = build_pocket()
    pocket.structure.pdb_path = str(receptor_source)
    pose_dir = tmp_path / "poses"

    monkeypatch.setattr(
        vina, "require_vina_binary", lambda: Path("/usr/local/bin/vina")
    )

    def fake_smiles_to_pdbqt(smiles: str, out_path: Path) -> Path:
        _ = smiles
        out_path.write_text("LIGAND\n", encoding="utf-8")
        return out_path

    def fake_structure_to_pdbqt(pdb_path: Path, out_path: Path) -> Path:
        _ = pdb_path
        out_path.write_text("RECEPTOR\n", encoding="utf-8")
        return out_path

    monkeypatch.setattr(vina, "smiles_to_pdbqt", fake_smiles_to_pdbqt)
    monkeypatch.setattr(vina, "structure_to_pdbqt", fake_structure_to_pdbqt)

    def fake_run(command, capture_output, text, check, shell):
        output_path = Path(command[command.index("--out") + 1])
        output_path.write_text(
            "REMARK VINA RESULT:      -9.5      0.000      0.000\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(vina.subprocess, "run", fake_run)

    poses = vina.dock(pocket, "CCO", output_dir=pose_dir)

    assert len(poses) == 1
    assert Path(poses[0].pose_pdb_path).exists()


def test_parse_vina_output_rejects_missing_scores() -> None:
    with pytest.raises(ValueError, match="REMARK VINA RESULT"):
        vina.parse_vina_output("MODEL 1\nENDMDL\n", build_pocket(), "CCO")


def test_smiles_to_pdbqt_round_trip_if_meeko_available(tmp_path: Path) -> None:
    if importlib.util.find_spec("meeko") is None:
        pytest.skip("meeko not installed")
    out_path = tmp_path / "ligand.pdbqt"
    result = vina.smiles_to_pdbqt("CCO", out_path)
    assert result == out_path
    assert out_path.read_text(encoding="utf-8").strip()


def test_select_largest_fragment_prefers_main_ligand() -> None:
    molecule = Chem.MolFromSmiles("CCO.[Na+].[Cl-]")
    assert molecule is not None
    fragment = vina._select_largest_fragment(molecule)
    assert Chem.MolToSmiles(fragment) == "CCO"


def test_structure_to_pdbqt_falls_back_when_meeko_receptor_prep_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    receptor_source = tmp_path / "cxcr4.pdb"
    receptor_source.write_text(
        "ATOM      1  N   MET A   1      23.812   4.747 -13.100  1.00 31.70           N\n"
        "ATOM      2  CA  MET A   1      24.919   3.796 -13.357  1.00 31.70           C\n"
        "ATOM      3  O   MET A   1      23.709   3.177 -15.258  1.00 31.70           O\n",
        encoding="utf-8",
    )
    out_path = tmp_path / "receptor.pdbqt"

    monkeypatch.setattr(vina, "_require_meeko_module", lambda: object())
    monkeypatch.setattr(
        vina.shutil, "which", lambda _: "/fake/bin/mk_prepare_receptor.py"
    )

    def fake_run(command, capture_output, text, check, shell):
        return subprocess.CompletedProcess(
            command,
            1,
            stdout="",
            stderr="RuntimeError: Updated 1 H positions but deleted 9",
        )

    monkeypatch.setattr(vina.subprocess, "run", fake_run)

    result = vina.structure_to_pdbqt(receptor_source, out_path)

    assert result == out_path
    pdbqt_text = out_path.read_text(encoding="utf-8")
    assert "     0.000 N " in pdbqt_text
    assert "     0.000 C " in pdbqt_text
    assert "     0.000 OA" in pdbqt_text
