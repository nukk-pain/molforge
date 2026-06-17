# pyright: reportMissingImports=false
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from molforge.docking.pose_rescorer import (  # noqa: E402
    PoseBustersFilter,
    PoseRescorer,
    RescoreResult,
)


def _write_boltz_pdb(path: Path, *, chain_b_etatm: str) -> None:
    # Minimal Boltz-style complex: chain A two protein residues + chain B
    # HETATMs supplied by caller. Coordinates chosen so that the protein
    # and ligand are spatially separated (no clash) for PASS cases.
    path.write_text(
        "ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00 50.00           N  \n"
        "ATOM      2  CA  GLY A   1       1.500   0.000   0.000  1.00 50.00           C  \n"
        "ATOM      3  C   GLY A   1       2.250   1.300   0.000  1.00 50.00           C  \n"
        "ATOM      4  O   GLY A   1       1.600   2.330   0.000  1.00 50.00           O  \n"
        "ATOM      5  N   ALA A   2       3.570   1.300   0.000  1.00 50.00           N  \n"
        "ATOM      6  CA  ALA A   2       4.320   2.560   0.000  1.00 50.00           C  \n"
        "ATOM      7  C   ALA A   2       5.820   2.380   0.000  1.00 50.00           C  \n"
        "ATOM      8  O   ALA A   2       6.500   3.390   0.000  1.00 50.00           O  \n"
        "ATOM      9  CB  ALA A   2       3.950   3.310   1.250  1.00 50.00           C  \n"
        + chain_b_etatm
        + "END\n",
        encoding="utf-8",
    )


def test_pose_rescorer_protocol_runtime_checkable() -> None:
    filt = PoseBustersFilter()
    assert isinstance(filt, PoseRescorer)


def test_posebusters_filter_valid_pose_passes(tmp_path: Path) -> None:
    # Ethanol (CCO) placed 20Å away from protein — no steric clash.
    # Coordinates chosen for a valid sp3 geometry (rough, sanitized by
    # RDKit template assignment).
    ligand_etatm = (
        "HETATM   10  C1  LIG B   1      20.000   0.000   0.000  1.00 50.00           C  \n"
        "HETATM   11  C2  LIG B   1      21.540   0.000   0.000  1.00 50.00           C  \n"
        "HETATM   12  O1  LIG B   1      22.110   1.400   0.000  1.00 50.00           O  \n"
    )
    pdb_path = tmp_path / "pose_valid.pdb"
    _write_boltz_pdb(pdb_path, chain_b_etatm=ligand_etatm)

    filt = PoseBustersFilter()
    result = filt.score(complex_pdb=pdb_path, ligand_smiles="CCO")
    # We tolerate "not_too_far_away" failing (ligand is 20Å away on purpose).
    # What we care about: the adapter worked AND the result is a RescoreResult.
    assert isinstance(result, RescoreResult)
    assert result.score is None  # binary filter
    assert "checks" in result.auxiliary
    # Core chemistry checks must pass for a well-formed ethanol pose.
    checks = result.auxiliary["checks"]
    assert checks["mol_pred_loaded"] is True
    assert checks["sanitization"] is True
    assert checks["all_atoms_connected"] is True


def test_posebusters_filter_adapter_error_is_recorded(tmp_path: Path) -> None:
    # Empty file → adapter raises → RescoreResult carries adapter_error.
    bad_pdb = tmp_path / "bad.pdb"
    bad_pdb.write_text("", encoding="utf-8")

    filt = PoseBustersFilter()
    result = filt.score(complex_pdb=bad_pdb, ligand_smiles="CCO")
    assert result.valid is False
    assert any("adapter_error" in r for r in result.fail_reasons)


def test_posebusters_filter_invalid_smiles_is_recorded(tmp_path: Path) -> None:
    ligand_etatm = (
        "HETATM   10  C1  LIG B   1      20.000   0.000   0.000  1.00 50.00           C  \n"
        "HETATM   11  C2  LIG B   1      21.540   0.000   0.000  1.00 50.00           C  \n"
        "HETATM   12  O1  LIG B   1      22.110   1.400   0.000  1.00 50.00           O  \n"
    )
    pdb_path = tmp_path / "pose.pdb"
    _write_boltz_pdb(pdb_path, chain_b_etatm=ligand_etatm)

    filt = PoseBustersFilter()
    result = filt.score(complex_pdb=pdb_path, ligand_smiles="NOT_A_SMILES")
    assert result.valid is False
    assert any("adapter_error" in r for r in result.fail_reasons)


@dataclass(frozen=True, slots=True)
class _AlwaysValidRescorer:
    name: str = "mock_always_valid"

    def score(self, *, complex_pdb: Path, ligand_smiles: str) -> RescoreResult:
        _ = complex_pdb, ligand_smiles
        return RescoreResult(valid=True, score=None)


def test_mock_rescorer_conforms_to_protocol() -> None:
    m = _AlwaysValidRescorer()
    assert isinstance(m, PoseRescorer)
