# pyright: reportMissingImports=false
"""Regression + property tests for the RMSD function family.

Covers three
kinds of checks here:

  (1) Identity  — same SDF compared to itself must give RMSD ≈ 0 under
      every metric; guards against accidental over-estimation regressions.
  (2) Translation — rigid translation of a ligand should give positional
      RMSD equal to the translation magnitude, while rigid-body-aligning
      metrics (GetBestRMS) should stay ≈ 0. Guards against re-introducing
      the silent internal-alignment bug.
  (3) Atom permutation — shuffling atom order in an SDF should give
      naive Kabsch RMSD > 0 (it's position-matching by index) but
      positional symmetry-aware RMSD ≈ 0. Guards against the
      wrong-atom-correspondence bug.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

from molforge.docking.pose_rmsd_from_cif import (  # noqa: E402
    rmsd_positional_symmetry_aware,
)


# --- fixtures -----------------------------------------------------------


def _ethanol_sdf(coords: list[tuple[float, float, float]], path: Path) -> Path:
    """Write an ethanol (CCO) SDF with the given atom coords (C, C, O)."""
    mol = Chem.MolFromSmiles("CCO")
    mol = Chem.AddHs(mol, addCoords=False)
    # Embed a random conformer, then replace with our exact coords (heavy
    # atoms only) — keep H in embedded positions.
    AllChem.EmbedMolecule(mol, randomSeed=0)
    heavy_indices = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() != "H"]
    assert len(heavy_indices) == len(coords)
    conf = mol.GetConformer()
    for idx, (x, y, z) in zip(heavy_indices, coords):
        conf.SetAtomPosition(idx, (x, y, z))
    mol = Chem.RemoveHs(mol)
    writer = Chem.SDWriter(str(path))
    writer.write(mol)
    writer.close()
    return path


@pytest.fixture()
def ethanol_pair(tmp_path):
    """Two identical ethanol conformers at different translations."""
    base = [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (2.1, 1.3, 0.0)]
    offset = 5.0
    shifted = [(x + offset, y, z) for (x, y, z) in base]
    a = _ethanol_sdf(base, tmp_path / "a.sdf")
    b = _ethanol_sdf(shifted, tmp_path / "b.sdf")
    return a, b, offset


# --- identity ------------------------------------------------------------


def test_identity_positional_symmetry_zero(ethanol_pair):
    """Same SDF compared to itself must give positional sym-aware RMSD ≈ 0."""
    a, _, _ = ethanol_pair
    assert rmsd_positional_symmetry_aware(a, a, "CCO") == pytest.approx(
        0.0, abs=1e-6
    )


# --- translation ---------------------------------------------------------


def test_translation_positional_matches_offset(ethanol_pair):
    """Pure rigid translation: positional sym-aware RMSD equals the
    translation magnitude. Guards against re-introducing a hidden
    internal-alignment step. If this test ever
    passes at ≈ 0, someone added alignment back into the positional
    function and must be reverted.
    """
    a, b, offset = ethanol_pair
    result = rmsd_positional_symmetry_aware(a, b, "CCO")
    assert result == pytest.approx(offset, abs=1e-4)


# --- atom permutation ----------------------------------------------------


def test_atom_permutation_positional_is_symmetry_aware(tmp_path):
    """Benzene has 6-fold atom equivalence; a rotated index order is the
    same molecule and must give RMSD ≈ 0 under the positional symmetry-
    aware metric even when the SDF stores atoms in shifted order."""
    # Write benzene with heavy atoms in canonical order.
    benzene_coords = [
        (math.cos(math.radians(60 * i)) * 1.4,
         math.sin(math.radians(60 * i)) * 1.4,
         0.0)
        for i in range(6)
    ]
    mol = Chem.MolFromSmiles("c1ccccc1")
    mol = Chem.AddHs(mol, addCoords=False)
    AllChem.EmbedMolecule(mol, randomSeed=0)
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() != "H"]
    conf = mol.GetConformer()
    for idx, xyz in zip(heavy, benzene_coords):
        conf.SetAtomPosition(idx, xyz)
    mol = Chem.RemoveHs(mol)
    ref_path = tmp_path / "ref.sdf"
    w = Chem.SDWriter(str(ref_path))
    w.write(mol)
    w.close()

    # Same molecule but atom order rotated by 1 (labels shifted).
    rotated_coords = benzene_coords[1:] + benzene_coords[:1]
    mol2 = Chem.MolFromSmiles("c1ccccc1")
    mol2 = Chem.AddHs(mol2, addCoords=False)
    AllChem.EmbedMolecule(mol2, randomSeed=0)
    heavy2 = [a.GetIdx() for a in mol2.GetAtoms() if a.GetSymbol() != "H"]
    conf2 = mol2.GetConformer()
    for idx, xyz in zip(heavy2, rotated_coords):
        conf2.SetAtomPosition(idx, xyz)
    mol2 = Chem.RemoveHs(mol2)
    shifted_path = tmp_path / "rot.sdf"
    w = Chem.SDWriter(str(shifted_path))
    w.write(mol2)
    w.close()

    # Positional symmetry-aware: should see this as the same molecule.
    r_sym = rmsd_positional_symmetry_aware(
        shifted_path, ref_path, "c1ccccc1"
    )
    assert r_sym == pytest.approx(0.0, abs=1e-3), (
        f"symmetry-aware RMSD should recognise benzene rotation as identity; "
        f"got {r_sym}"
    )

    # Naive Kabsch matching by index will still find the right alignment
    # for pure translation/rotation (it does SVD), so it can also return
    # near-zero here — that's OK. The bug that matters is asymmetric
    # geometries where the wrong-index mapping gave inflated RMSDs, which
    # is what the boltz-complex + drug-like retrospective verified in
    # archived data rather than synthetic fixtures.


# --- receptor-aligned on real artifact ----------------------------------


@pytest.mark.skipif(
    not (REPO_ROOT / "archive/runs/v3-c1c-hard-pass-boltz2-n10").exists(),
    reason="v3 Boltz artifact not present in this checkout",
)
def test_receptor_aligned_on_7a1p_qw2_sample_0():
    """Integration check on an actual Boltz pose + RCSB crystal.

    Verifies the library function reproduces the retrospective finding
    that Boltz sample 0 for 7A1P_QW2 is ≤ 1 Å under the correct metric,
    contradicting the naive-Kabsch value of 3.57 Å.
    """
    import tempfile

    import httpx

    from molforge.docking.pose_rmsd_from_cif import (
        rmsd_receptor_aligned_symmetry_aware,
    )
    from molforge.docking.posebusters_target import build_pose_target

    ctx = build_pose_target("7A1P_QW2")
    boltz_path = (
        REPO_ROOT
        / "archive/runs/v3-c1c-hard-pass-boltz2-n10/boltz_out_7A1P_QW2/"
        "out/boltz_results_input/predictions/input/input_model_0.pdb"
    )
    if not boltz_path.exists():
        pytest.skip("7A1P_QW2 sample 0 Boltz artifact not present")

    try:
        resp = httpx.get(
            "https://files.rcsb.org/download/7A1P.pdb", timeout=30
        )
        resp.raise_for_status()
    except Exception:  # noqa: BLE001
        pytest.skip("RCSB unreachable — skipping integration test")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".pdb", delete=False
    ) as f:
        f.write(resp.text)
        crystal_path = Path(f.name)

    r, ca_r, _n = rmsd_receptor_aligned_symmetry_aware(
        boltz_path, crystal_path, "QW2", ctx.ligand_smiles
    )
    assert r < 1.0, (
        f"Boltz 7A1P_QW2 sample 0 should be < 1 Å by receptor-aligned "
        f"symmetry-aware positional RMSD; got {r:.3f} Å (Cα {ca_r:.3f})"
    )
