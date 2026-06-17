# pyright: reportMissingImports=false
"""Tests for rmsd_chain_equivalent_symmetry_aware — chain-iteration RMSD metric.

Covers the homomeric/multimeric use-case (1IEP BCR-ABL homodimer) where
DiffDock may place the rank-1 pose in a different chain than the crystal
reference ligand.  The function iterates all crystal chains and returns the
minimum RMSD, so chain-assignment ambiguity does not inflate the reported value.
"""
from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from molforge.docking.pose_rmsd_from_cif import (  # noqa: E402
    rmsd_chain_equivalent_symmetry_aware,
)


# ---------------------------------------------------------------------------
# PDB fixture helpers
# ---------------------------------------------------------------------------

def _write_minimal_homodimer_pdb(
    path: Path,
    *,
    chain_a_lig_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    chain_b_lig_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ligand_code: str = "STI",
) -> None:
    """Write a minimal homodimer crystal PDB with:
    - 15 Cα residues in chain A (GLY x15, resids 1-15)
    - 15 Cα residues in chain B (GLY x15, resids 1-15)
    - One ligand (ligand_code) HETATM in chain A
    - One ligand (ligand_code) HETATM in chain B, displaced by chain_b_lig_offset

    The protein chains are identical in geometry so the Kabsch alignment
    should be near-perfect when aligning predicted chain A onto crystal chain A
    or crystal chain B.
    """
    lines: list[str] = []
    atom_serial = 1

    for chain in ("A", "B"):
        for resid in range(1, 16):
            # Place Cα on a line along X, 3.8 Å spacing — canonical beta strand
            x = (resid - 1) * 3.8
            y = 0.0
            z = 0.0
            lines.append(
                f"ATOM  {atom_serial:5d}  CA  GLY {chain}{resid:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C  "
            )
            atom_serial += 1

    # Ligand HETATM in chain A — 4 heavy atoms forming a simple square
    lig_a_base = [
        (0.5, 0.5, 0.5),
        (1.5, 0.5, 0.5),
        (1.5, 1.5, 0.5),
        (0.5, 1.5, 0.5),
    ]
    ox, oy, oz = chain_a_lig_offset
    for i, (x, y, z) in enumerate(lig_a_base, start=1):
        lines.append(
            f"HETATM{atom_serial:5d}  C{i}  {ligand_code} A   1    "
            f"{x + ox:8.3f}{y + oy:8.3f}{z + oz:8.3f}  1.00  0.00           C  "
        )
        atom_serial += 1

    # Ligand HETATM in chain B — same base geometry displaced by chain_b_lig_offset
    bx, by, bz = chain_b_lig_offset
    for i, (x, y, z) in enumerate(lig_a_base, start=1):
        lines.append(
            f"HETATM{atom_serial:5d}  C{i}  {ligand_code} B   1    "
            f"{x + bx:8.3f}{y + by:8.3f}{z + bz:8.3f}  1.00  0.00           C  "
        )
        atom_serial += 1

    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_predicted_complex_pdb(
    path: Path,
    *,
    lig_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ligand_code: str = "STI",
) -> None:
    """Write a Boltz-style predicted complex PDB.

    Follows Boltz-2 convention: protein ATOM records on chain A, ligand HETATM
    records on chain B.  split_boltz_pdb (called internally by
    rmsd_receptor_aligned_symmetry_aware) hard-codes ligand_chain='B', so the
    predicted complex must have HETATM on chain B.
    """
    lines: list[str] = []
    atom_serial = 1

    # Protein: 15 GLY Cα on chain A
    for resid in range(1, 16):
        x = (resid - 1) * 3.8
        lines.append(
            f"ATOM  {atom_serial:5d}  CA  GLY A{resid:4d}    "
            f"{x:8.3f}   0.000   0.000  1.00  0.00           C  "
        )
        atom_serial += 1

    # Ligand: 4 heavy atoms on chain B (Boltz convention)
    ox, oy, oz = lig_offset
    lig_base = [(0.5, 0.5, 0.5), (1.5, 0.5, 0.5), (1.5, 1.5, 0.5), (0.5, 1.5, 0.5)]
    for i, (x, y, z) in enumerate(lig_base, start=1):
        lines.append(
            f"HETATM{atom_serial:5d}  C{i}  {ligand_code} B   1    "
            f"{x + ox:8.3f}{y + oy:8.3f}{z + oz:8.3f}  1.00  0.00           C  "
        )
        atom_serial += 1

    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_single_chain_crystal_pdb(
    path: Path,
    *,
    lig_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ligand_code: str = "STI",
) -> None:
    """Write a single-chain crystal PDB (chain A protein + chain A HETATM ligand)."""
    lines: list[str] = []
    atom_serial = 1

    for resid in range(1, 16):
        x = (resid - 1) * 3.8
        lines.append(
            f"ATOM  {atom_serial:5d}  CA  GLY A{resid:4d}    "
            f"{x:8.3f}   0.000   0.000  1.00  0.00           C  "
        )
        atom_serial += 1

    ox, oy, oz = lig_offset
    lig_base = [(0.5, 0.5, 0.5), (1.5, 0.5, 0.5), (1.5, 1.5, 0.5), (0.5, 1.5, 0.5)]
    for i, (x, y, z) in enumerate(lig_base, start=1):
        lines.append(
            f"HETATM{atom_serial:5d}  C{i}  {ligand_code} A   1    "
            f"{x + ox:8.3f}{y + oy:8.3f}{z + oz:8.3f}  1.00  0.00           C  "
        )
        atom_serial += 1

    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# Minimal SMILES for a 4-carbon ring-like ligand (cyclobutane as proxy)
_LIGAND_SMILES = "C1CCC1"


# ---------------------------------------------------------------------------
# Test 1 — synthetic homodimer: identical poses → RMSD ≈ 0
# ---------------------------------------------------------------------------

def test_homodimer_identical_poses_near_zero(tmp_path: Path) -> None:
    """Both chains have the same ligand geometry as the predicted pose.

    The minimum RMSD over both crystal chains must be < 0.5 Å (essentially
    zero for identical coordinates, tolerance for floating-point).

    Predicted PDB follows Boltz-2 convention: protein on chain A (ATOM),
    ligand on chain B (HETATM).  Crystal has identical ligand geometry on both
    chain A and chain B.
    """
    predicted = tmp_path / "predicted.pdb"
    crystal = tmp_path / "crystal.pdb"

    # Predicted: Boltz-style complex (protein chain A, ligand chain B)
    _write_predicted_complex_pdb(predicted, ligand_code="STI")
    # Crystal homodimer: both chains have ligand at the same position as predicted
    _write_minimal_homodimer_pdb(crystal, ligand_code="STI")

    min_rmsd, ca_rmsd, n_ca, best_chain = rmsd_chain_equivalent_symmetry_aware(
        predicted,
        crystal,
        ligand_code="STI",
        ligand_smiles=_LIGAND_SMILES,
        predicted_chain="A",
    )
    assert min_rmsd < 0.5, (
        f"Expected near-zero RMSD for identical geometry, got {min_rmsd:.4f} Å"
    )
    assert best_chain in ("A", "B"), f"best_chain should be A or B, got {best_chain!r}"


# ---------------------------------------------------------------------------
# Test 2 — asymmetric homodimer: one chain displaced 10 Å → pick the close one
# ---------------------------------------------------------------------------

def test_asymmetric_homodimer_picks_minimum(tmp_path: Path) -> None:
    """Chain A ligand matches predicted; chain B ligand is displaced 10 Å.

    Function must return the chain-A RMSD (< 5 Å), not chain-B (~10 Å).

    Predicted PDB follows Boltz-2 convention: protein on chain A (ATOM),
    ligand on chain B (HETATM) at position matching the crystal chain-A ligand.
    """
    predicted = tmp_path / "predicted.pdb"
    crystal = tmp_path / "crystal.pdb"

    _write_predicted_complex_pdb(predicted, ligand_code="STI")
    _write_minimal_homodimer_pdb(
        crystal,
        chain_a_lig_offset=(0.0, 0.0, 0.0),   # chain A matches predicted ligand
        chain_b_lig_offset=(10.0, 0.0, 0.0),  # chain B is 10 Å away
        ligand_code="STI",
    )

    min_rmsd, ca_rmsd, n_ca, best_chain = rmsd_chain_equivalent_symmetry_aware(
        predicted,
        crystal,
        ligand_code="STI",
        ligand_smiles=_LIGAND_SMILES,
        predicted_chain="A",
    )
    # The minimum should be the close chain, well below 5 Å
    assert min_rmsd < 5.0, (
        f"Should pick the close chain (chain A), got {min_rmsd:.4f} Å"
    )
    assert best_chain == "A", (
        f"Expected best_chain='A' (close ligand), got {best_chain!r}"
    )


# ---------------------------------------------------------------------------
# Test 3 — auto-detect crystal_chains=None finds both A and B
# ---------------------------------------------------------------------------

def test_auto_detect_finds_both_chains(tmp_path: Path) -> None:
    """crystal_chains=None triggers auto-detection from ATOM records.

    The homodimer crystal has ATOM records for chains A and B; auto-detect
    must enumerate both and find chain B (the close one), rather than only
    checking chain A (the far one).

    Predicted PDB follows Boltz-2 convention: protein on chain A (ATOM),
    ligand on chain B (HETATM) at origin — matching crystal chain B.
    """
    predicted = tmp_path / "predicted.pdb"
    crystal = tmp_path / "crystal.pdb"

    _write_predicted_complex_pdb(predicted, ligand_code="STI")
    # Place the close ligand in crystal chain B; crystal chain A is 10 Å away.
    # If auto-detect misses chain B, the returned RMSD would be large (~10 Å).
    _write_minimal_homodimer_pdb(
        crystal,
        chain_a_lig_offset=(10.0, 0.0, 0.0),  # far
        chain_b_lig_offset=(0.0, 0.0, 0.0),   # matches predicted ligand position
        ligand_code="STI",
    )

    min_rmsd, ca_rmsd, n_ca, best_chain = rmsd_chain_equivalent_symmetry_aware(
        predicted,
        crystal,
        ligand_code="STI",
        ligand_smiles=_LIGAND_SMILES,
        predicted_chain="A",
        crystal_chains=None,  # explicit: use auto-detect
    )
    assert min_rmsd < 5.0, (
        f"Auto-detect should find chain B (close), got {min_rmsd:.4f} Å"
    )
    assert best_chain == "B", (
        f"Expected best_chain='B' (close ligand), got {best_chain!r}"
    )


# ---------------------------------------------------------------------------
# Test 4 — single-chain crystal fallback (no iteration needed)
# ---------------------------------------------------------------------------

def test_single_chain_crystal_fallback(tmp_path: Path) -> None:
    """When the crystal has only one chain, function still works correctly.

    This is the degenerate case — equivalent to calling
    rmsd_receptor_aligned_symmetry_aware directly with crystal_chain='A'.

    Predicted PDB follows Boltz-2 convention: protein on chain A (ATOM),
    ligand on chain B (HETATM).  Crystal is single-chain A only.
    """
    predicted = tmp_path / "predicted.pdb"
    crystal = tmp_path / "crystal.pdb"

    _write_predicted_complex_pdb(predicted, ligand_code="STI")
    _write_single_chain_crystal_pdb(crystal, ligand_code="STI")  # single-chain crystal

    min_rmsd, ca_rmsd, n_ca, best_chain = rmsd_chain_equivalent_symmetry_aware(
        predicted,
        crystal,
        ligand_code="STI",
        ligand_smiles=_LIGAND_SMILES,
        predicted_chain="A",
    )
    assert min_rmsd < 0.5, (
        f"Single-chain fallback should yield near-zero RMSD, got {min_rmsd:.4f} Å"
    )
    assert best_chain == "A", f"Only chain A present; best_chain should be 'A', got {best_chain!r}"


# ---------------------------------------------------------------------------
# Test 5 — docstring lint (ensures test_rmsd_docstring_contract still passes)
# ---------------------------------------------------------------------------

def test_docstring_contract_for_new_function() -> None:
    """rmsd_chain_equivalent_symmetry_aware must satisfy the RMSD contract.

    Verifies the same category checks applied by test_rmsd_docstring_contract.py:
    - Category 1: coordinate-frame declaration
    - Category 2: symmetry handling declaration
    """
    import inspect
    from molforge.docking.pose_rmsd_from_cif import (
        rmsd_chain_equivalent_symmetry_aware as fn,
    )

    doc = (fn.__doc__ or "").lower()

    # Category 1 — coordinate-frame keywords (must hit at least one)
    frame_keywords = {
        "same frame", "crystal frame", "receptor-aligned", "receptor aligned",
        "cα superposition", "ca superposition", "deprecated", "legacy", "kabsch",
    }
    assert any(kw in doc for kw in frame_keywords), (
        f"Docstring missing coordinate-frame keyword. Need one of: {sorted(frame_keywords)}"
    )

    # Category 2 — symmetry handling (must hit at least one)
    symmetry_keywords = {
        "symmetry", "getbestrms", "assumes atom ordering",
        "atom ordering matches", "wrong atom order",
    }
    assert any(kw in doc for kw in symmetry_keywords), (
        f"Docstring missing symmetry keyword. Need one of: {sorted(symmetry_keywords)}"
    )

    # Additional contract phrases required by the task spec
    assert "coordinate frame" in doc, (
        "Docstring must contain phrase 'coordinate frame' (required by task spec)"
    )
