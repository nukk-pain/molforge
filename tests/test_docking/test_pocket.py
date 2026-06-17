# pyright: reportMissingImports=false
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from contracts.schema import ProteinStructure, StructureSource  # noqa: E402
from molforge.docking import pocket  # noqa: E402


def build_structure(pdb_path: Path) -> ProteinStructure:
    return ProteinStructure(
        gene="CXCR4",
        uniprot="P61073",
        pdb_path=str(pdb_path),
        source=StructureSource.ALPHAFOLD_DB,
        confidence=88.1,
    )


def test_detect_pocket_uses_fpocket_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    structure_path = tmp_path / "cxcr4.pdb"
    structure_path.write_text(
        "ATOM      1  CA  ASP A  97      10.000  11.000  12.000  1.00 80.00           C\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "cxcr4_out"
    pockets_dir = output_dir / "pockets"
    pockets_dir.mkdir(parents=True)
    (output_dir / "cxcr4_info.txt").write_text(
        "Pocket 1 :\nScore : 42.0\nDruggability Score : 0.73\n",
        encoding="utf-8",
    )
    (pockets_dir / "pocket1_vert.pqr").write_text(
        "ATOM      1  APOL STP A   1      10.000  11.000  12.000  0.00  0.00\n"
        "ATOM      2  APOL STP A   2      14.000  15.000  16.000  0.00  0.00\n",
        encoding="utf-8",
    )
    (pockets_dir / "pocket1_atm.pdb").write_text(
        "ATOM      1  CA  ASP A  97      10.000  11.000  12.000  1.00 80.00           C\n"
        "ATOM      2  CA  TYR A 116      14.000  15.000  16.000  1.00 80.00           C\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        pocket, "which_binary", lambda name: Path("/usr/local/bin/fpocket")
    )
    monkeypatch.setattr(
        pocket.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout="ok", stderr=""
        ),
    )

    result = pocket.detect_pocket(build_structure(structure_path))

    assert result.center_xyz == pytest.approx((12.0, 13.0, 14.0))
    assert result.size_xyz == pytest.approx((4.0, 4.0, 4.0))
    assert result.druggability_score == pytest.approx(0.73)
    assert result.residues == ["ASP97", "TYR116"]


def test_detect_pocket_falls_back_to_geometry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    structure_path = tmp_path / "cxcr4.pdb"
    structure_path.write_text(
        "ATOM      1  CA  ASP A  97      10.000  11.000  12.000  1.00 80.00           C\n"
        "ATOM      2  CA  TYR A 116      16.000  17.000  18.000  1.00 80.00           C\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(pocket, "which_binary", lambda name: None)

    result = pocket.detect_pocket(build_structure(structure_path))

    assert result.center_xyz == pytest.approx((13.0, 14.0, 15.0))
    assert result.size_xyz == pytest.approx((6.0, 6.0, 6.0))
    assert result.druggability_score is None
    assert result.residues == ["ASP97", "TYR116"]
