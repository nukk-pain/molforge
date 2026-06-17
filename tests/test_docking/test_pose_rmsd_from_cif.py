# pyright: reportMissingImports=false
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from molforge.docking.pose_rmsd_from_cif import (  # noqa: E402
    _ca_coords_from_pdb_text,
    _find_unique_exact_ca_gapped_pairs,
    _find_unique_exact_ca_subsequence_pairs,
    _ordered_ca_residues_from_pdb_text,
    extract_ligand_heavy_atom_coords,
    rmsd_against_reference,
    rmsd_receptor_aligned_symmetry_aware,
)
from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402


def _write_hetatm(path: Path, coords: list[tuple[float, float, float, str]]) -> None:
    """Helper: write a fake PDB with HETATM lines. coords = (x, y, z, element)."""
    lines = []
    for index, (x, y, z, element) in enumerate(coords, start=1):
        name = f" {element.upper()}" if len(element) == 1 else element.upper()
        lines.append(
            f"HETATM{index:5d} {name:<4s} LIG B   1    {x:8.3f}{y:8.3f}{z:8.3f}"
            f"  1.00  0.00          {element.upper():>2s}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_extract_ligand_coords_skips_hydrogens(tmp_path: Path) -> None:
    pdb = tmp_path / "ligand.pdb"
    _write_hetatm(
        pdb,
        [
            (0.0, 0.0, 0.0, "C"),
            (1.0, 0.0, 0.0, "H"),  # must be skipped
            (1.5, 1.5, 0.0, "O"),
        ],
    )
    coords = extract_ligand_heavy_atom_coords(pdb)
    assert coords == [(0.0, 0.0, 0.0), (1.5, 1.5, 0.0)]


def test_rmsd_against_reference_matches_identical_structure(tmp_path: Path) -> None:
    pred = tmp_path / "pred.pdb"
    ref = tmp_path / "ref.pdbqt"
    coords = [
        (0.0, 0.0, 0.0, "C"),
        (1.0, 0.0, 0.0, "C"),
        (0.0, 1.0, 0.0, "O"),
    ]
    _write_hetatm(pred, coords)
    _write_hetatm(ref, coords)
    assert rmsd_against_reference(pred, ref) == pytest.approx(0.0, abs=1e-6)


def test_rmsd_against_reference_raises_on_atom_count_mismatch(tmp_path: Path) -> None:
    pred = tmp_path / "pred.pdb"
    ref = tmp_path / "ref.pdbqt"
    _write_hetatm(pred, [(0.0, 0.0, 0.0, "C"), (1.0, 0.0, 0.0, "C")])
    _write_hetatm(ref, [(0.0, 0.0, 0.0, "C")])
    with pytest.raises(ValueError, match="atom count mismatch"):
        rmsd_against_reference(pred, ref)


def test_rmsd_against_reference_reproduces_phase2_number(tmp_path: Path) -> None:
    """Sanity: running the new util on the existing phase2 artifact should
    reproduce the RMSD recorded in phase2-posebench-pass/6X8D_ARA/rmsd.json."""
    reference = (
        REPO_ROOT / "archive/runs/phase2-posebench-pass/6X8D_ARA/reference_pose.pdbqt"
    )
    predicted = REPO_ROOT / "archive/runs/phase2-posebench-pass/6X8D_ARA/pose.pdbqt"
    if not reference.exists() or not predicted.exists():
        pytest.skip("phase2 artifacts absent from this checkout")
    value = rmsd_against_reference(predicted, reference)
    # phase2 recorded 1.9902331354169112 — allow small drift from atom ordering.
    assert value == pytest.approx(1.99, abs=0.05)


def _write_ca_only_pdb(
    path: Path,
    residues: list[tuple[int, str, str, float, float, float]],
    *,
    chain: str = "A",
) -> None:
    lines: list[str] = []
    for index, (resseq, icode, resname, x, y, z) in enumerate(residues, start=1):
        insertion = icode or " "
        lines.append(
            f"ATOM  {index:5d}  CA  {resname:>3s} {chain}{resseq:4d}{insertion}   "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 10.00           C"
        )
    path.write_text("\n".join(lines) + "\nEND\n", encoding="utf-8")


def _append_ligand_hetatm(
    path: Path,
    coords: list[tuple[float, float, float, str]],
    *,
    ligand_code: str = "LIG",
    chain: str = "A",
) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    atom_index = sum(1 for line in lines if line.startswith(("ATOM", "HETATM"))) + 1
    if lines and lines[-1] == "END":
        lines.pop()
    for offset, (x, y, z, element) in enumerate(coords, start=0):
        name = f" {element.upper()}" if len(element) == 1 else element.upper()
        lines.append(
            f"HETATM{atom_index + offset:5d} {name:<4s} {ligand_code:>3s} {chain}   1    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element.upper():>2s}"
        )
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_ethanol_sdf(path: Path) -> Path:
    mol = Chem.MolFromSmiles("CCO")
    mol = Chem.AddHs(mol, addCoords=False)
    mol.RemoveAllConformers()
    conf = Chem.Conformer(mol.GetNumAtoms())
    mol.AddConformer(conf)
    heavy_indices = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() != "H"]
    for idx, xyz in zip(
        heavy_indices,
        [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (2.1, 1.3, 0.0)],
    ):
        conf.SetAtomPosition(idx, xyz)
    mol = Chem.RemoveHs(mol)
    writer = Chem.SDWriter(str(path))
    writer.write(mol)
    writer.close()
    return path


def test_ca_coords_preserve_insertion_codes() -> None:
    text = "\n".join(
        [
            "ATOM      1  CA  ALA A  10      1.000   2.000   3.000  1.00 10.00           C",
            "ATOM      2  CA  ALA A  10A     4.000   5.000   6.000  1.00 10.00           C",
            "ATOM      3  CA  ALA A  11      7.000   8.000   9.000  1.00 10.00           C",
        ]
    )
    coords = _ca_coords_from_pdb_text(text, chain="A")
    assert coords == {
        (10, ""): (1.0, 2.0, 3.0),
        (10, "A"): (4.0, 5.0, 6.0),
        (11, ""): (7.0, 8.0, 9.0),
    }


def test_receptor_aligned_reports_residue_mismatch_diagnostics(tmp_path: Path) -> None:
    predicted = tmp_path / "predicted.pdb"
    crystal = tmp_path / "crystal.pdb"
    predicted_residues = [
        (idx, "A" if idx == 5 else "", "ALA", float(idx), 0.0, 0.0)
        for idx in range(1, 13)
    ]
    crystal_residues = [
        (idx + 100, "", "ALA", float(idx), 0.0, 0.0) for idx in range(1, 11)
    ]
    _write_ca_only_pdb(predicted, predicted_residues)
    _write_ca_only_pdb(crystal, crystal_residues)

    with pytest.raises(ValueError, match="too few common Cα") as excinfo:
        rmsd_receptor_aligned_symmetry_aware(
            predicted,
            crystal,
            "LIG",
            "CCO",
        )

    message = str(excinfo.value)
    assert "insertion_codes_present=predicted:True,crystal:False" in message
    assert "predicted_only_sample=['1', '2', '3', '4', '5A']" in message
    assert "crystal_only_sample=['101', '102', '103', '104', '105']" in message
    assert "fallback_attempted=True" in message
    assert "fallback_reason=ambiguous_exact_gapped_match" in message


def _ca_text(sequence: list[str], *, residue_start: int = 1) -> str:
    lines: list[str] = []
    for index, resname in enumerate(sequence, start=0):
        resid = residue_start + index
        x = float(index)
        lines.append(
            f"ATOM  {index + 1:5d}  CA  {resname:>3s} A{resid:4d}    "
            f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 10.00           C"
        )
    return "\n".join(lines)


def test_unique_exact_subsequence_fallback_matches_single_numbering_offset() -> None:
    sequence = ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU"]
    predicted = _ordered_ca_residues_from_pdb_text(_ca_text(sequence, residue_start=1))
    crystal = _ordered_ca_residues_from_pdb_text(_ca_text(sequence, residue_start=101))
    result = _find_unique_exact_ca_subsequence_pairs(predicted, crystal)
    assert result.reason is None
    assert result.matched_len == 10
    assert result.unique_hit_count == 1
    assert result.predicted_coords == result.crystal_coords


def test_unique_exact_subsequence_fallback_rejects_ambiguous_repeat() -> None:
    predicted = _ordered_ca_residues_from_pdb_text(
        _ca_text(["ALA"] * 12, residue_start=1)
    )
    crystal = _ordered_ca_residues_from_pdb_text(
        _ca_text(["ALA"] * 10, residue_start=101)
    )
    result = _find_unique_exact_ca_subsequence_pairs(predicted, crystal)
    assert result.reason == "ambiguous_exact_subsequence"
    assert result.unique_hit_count > 1
    assert result.predicted_coords is None


def test_unique_exact_subsequence_fallback_rejects_non_exact_subsequence() -> None:
    predicted = _ordered_ca_residues_from_pdb_text(
        _ca_text(
            ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU"],
            residue_start=1,
        )
    )
    crystal = _ordered_ca_residues_from_pdb_text(
        _ca_text(
            ["ALA", "CYS", "ASP", "GLU", "TYR", "GLY", "HIS", "ILE", "LYS", "LEU"],
            residue_start=101,
        )
    )
    result = _find_unique_exact_ca_subsequence_pairs(predicted, crystal)
    assert result.reason == "no_exact_subsequence"
    assert result.unique_hit_count == 0
    assert result.predicted_coords is None


def test_unique_exact_subsequence_fallback_rejects_non_standard_residues() -> None:
    predicted = _ordered_ca_residues_from_pdb_text(
        _ca_text(
            ["ALA", "CYS", "ASP", "GLU", "MSE", "GLY", "HIS", "ILE", "LYS", "LEU"],
            residue_start=1,
        )
    )
    crystal = _ordered_ca_residues_from_pdb_text(
        _ca_text(
            ["ALA", "CYS", "ASP", "GLU", "MSE", "GLY", "HIS", "ILE", "LYS", "LEU"],
            residue_start=101,
        )
    )
    result = _find_unique_exact_ca_subsequence_pairs(predicted, crystal)
    assert result.reason == "non_standard_residue"
    assert result.predicted_coords is None


def test_unique_exact_gapped_fallback_matches_internal_gap_blocks() -> None:
    predicted = _ordered_ca_residues_from_pdb_text(
        _ca_text(
            [
                "ALA",
                "CYS",
                "ASP",
                "GLU",
                "PHE",
                "GLY",
                "HIS",
                "ILE",
                "LYS",
                "LEU",
                "MET",
                "ASN",
            ],
            residue_start=1,
        )
    )
    crystal = _ordered_ca_residues_from_pdb_text(
        "\n".join(
            [
                _ca_text(["ALA", "CYS", "ASP", "GLU"], residue_start=101),
                _ca_text(["ILE", "LYS", "LEU"], residue_start=110),
                _ca_text(["ASN"], residue_start=120),
            ]
        )
    )
    result = _find_unique_exact_ca_gapped_pairs(predicted, crystal, min_common=3)
    assert result.reason is None
    assert result.matched_len == 8
    assert result.unique_hit_count == 1
    assert result.predicted_coords is not None
    assert result.crystal_coords is not None


def test_unique_exact_gapped_fallback_rejects_ambiguous_repeat() -> None:
    predicted = _ordered_ca_residues_from_pdb_text(
        _ca_text(
            ["ALA", "CYS", "ASP", "ALA", "CYS", "ASP", "PHE", "GLY", "HIS"],
            residue_start=1,
        )
    )
    crystal = _ordered_ca_residues_from_pdb_text(
        "\n".join(
            [
                _ca_text(["ALA", "CYS", "ASP"], residue_start=101),
                _ca_text(["PHE", "GLY", "HIS"], residue_start=110),
            ]
        )
    )
    result = _find_unique_exact_ca_gapped_pairs(predicted, crystal, min_common=3)
    assert result.reason == "ambiguous_exact_gapped_match"
    assert result.predicted_coords is None


def test_unique_exact_gapped_fallback_rejects_block_mismatch() -> None:
    predicted = _ordered_ca_residues_from_pdb_text(
        _ca_text(
            ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS"],
            residue_start=1,
        )
    )
    crystal = _ordered_ca_residues_from_pdb_text(
        "\n".join(
            [
                _ca_text(["ALA", "CYS", "ASP"], residue_start=101),
                _ca_text(["GLU", "TYR", "GLY", "HIS"], residue_start=110),
            ]
        )
    )
    result = _find_unique_exact_ca_gapped_pairs(predicted, crystal, min_common=3)
    assert result.reason == "no_exact_gapped_match"
    assert result.predicted_coords is None


def test_receptor_aligned_uses_sequence_offset_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predicted = tmp_path / "predicted_complex.pdb"
    crystal = tmp_path / "crystal_complex.pdb"
    sequence = ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU"]
    predicted_residues = [
        (idx, "", resname, float(idx), 0.0, 0.0)
        for idx, resname in enumerate(sequence, start=1)
    ]
    crystal_residues = [
        (idx + 100, "", resname, float(idx), 0.0, 0.0)
        for idx, resname in enumerate(sequence, start=1)
    ]
    _write_ca_only_pdb(predicted, predicted_residues)
    _write_ca_only_pdb(crystal, crystal_residues)
    _append_ligand_hetatm(
        crystal,
        [(0.0, 0.0, 0.0, "C"), (1.5, 0.0, 0.0, "C"), (2.1, 1.3, 0.0, "O")],
    )
    ligand_sdf = _write_ethanol_sdf(tmp_path / "ligand.sdf")

    import molforge.docking.boltz_pdb_split as boltz_split

    monkeypatch.setattr(
        boltz_split,
        "split_boltz_pdb",
        lambda *_args, **_kwargs: SimpleNamespace(ligand_sdf=ligand_sdf),
    )

    ligand_rmsd, ca_rmsd, n_common = rmsd_receptor_aligned_symmetry_aware(
        predicted,
        crystal,
        "LIG",
        "CCO",
    )
    assert n_common == 10
    assert ca_rmsd == pytest.approx(0.0, abs=1e-6)
    assert ligand_rmsd < 2.0
