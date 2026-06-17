"""Pose-level rescorer abstraction.

A `PoseRescorer` inspects a single Boltz-2 complex PDB (protein + ligand on
chain B, residue LIG) and reports a `RescoreResult`. Used by
`reselect_via_boltz2` to filter or rerank diffusion samples.

Two concrete implementations:
  - `PoseBustersFilter` — rule-based validity filter (Phase B1).
  - `VinaRescorer` — AutoDock Vina `--score_only` rescoring (Phase B1 Stage 2
     A'). Empirical force-field; different model family from Boltz's learned
     affinity head.

Convention: `RescoreResult.score` is "lower is better" (Vina/Boltz
affinity_pred_value convention). Rescorers whose native output is
"higher = better" (e.g. pKd) must negate before returning.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from .boltz_pdb_split import split_boltz_pdb


@dataclass(frozen=True, slots=True)
class RescoreResult:
    valid: bool
    score: float | None  # lower = better (Vina/Boltz convention)
    fail_reasons: tuple[str, ...] = ()
    auxiliary: dict[str, object] = field(default_factory=dict)


@runtime_checkable
class PoseRescorer(Protocol):
    name: str

    def score(self, *, complex_pdb: Path, ligand_smiles: str) -> RescoreResult:
        ...

    # Stage 2 rescorers (e.g. GNN) may want batch inference. Left as a
    # non-required hint; concrete Protocol only mandates `score`.


class PoseBustersFilter:
    """Rule-based validity filter (19 checks, dock mode).

    Not a scorer — `score=None`. `valid` is True iff all PoseBusters
    binary checks pass. On adapter or PoseBusters failure, returns
    `valid=False` with the reason recorded (silent drop forbidden).
    """

    name: str = "posebusters"

    def __init__(self, *, config: str = "dock") -> None:
        from posebusters import PoseBusters  # heavy import, lazy

        self._pb = PoseBusters(config=config)

    def score(self, *, complex_pdb: Path, ligand_smiles: str) -> RescoreResult:
        with tempfile.TemporaryDirectory() as tmp:
            try:
                split = split_boltz_pdb(complex_pdb, ligand_smiles, Path(tmp))
            except RuntimeError as exc:
                return RescoreResult(
                    valid=False,
                    score=None,
                    fail_reasons=(f"adapter_error: {exc}",),
                )

            try:
                df = self._pb.bust(
                    mol_pred=split.ligand_sdf,
                    mol_cond=split.protein_pdb,
                )
            except Exception as exc:  # noqa: BLE001
                return RescoreResult(
                    valid=False,
                    score=None,
                    fail_reasons=(f"posebusters_error: {type(exc).__name__}: {exc}",),
                )

            checks: dict[str, bool] = {}
            for col in df.columns:
                if col in ("file", "molecule"):
                    continue
                val = df.iloc[0][col]
                checks[col] = bool(val) if val is not None else False
            fails = tuple(k for k, v in checks.items() if not v)
            return RescoreResult(
                valid=len(fails) == 0,
                score=None,
                fail_reasons=fails,
                auxiliary={"checks": checks},
            )


_VINA_AFFINITY_PATTERN = re.compile(
    r"Estimated Free Energy of Binding\s*:\s*(-?\d+(?:\.\d+)?)\s*\(?kcal/mol\)?",
    re.IGNORECASE,
)
# Vina 1.2 prints the energy as a breakdown; recover via the component line
# when the headline string is absent.
_VINA_INTERMOL_PATTERN = re.compile(
    r"Final Intermolecular Energy\s*:\s*(-?\d+(?:\.\d+)?)"
)
_VINA_TORSIONAL_PATTERN = re.compile(
    r"Torsional Free Energy\s*:\s*(-?\d+(?:\.\d+)?)"
)


def _parse_vina_affinity(stdout: str) -> float | None:
    m = _VINA_AFFINITY_PATTERN.search(stdout)
    if m:
        return float(m.group(1))
    # Fallback for Vina 1.2 verbose output: sum intermolecular + torsional.
    intermol = _VINA_INTERMOL_PATTERN.search(stdout)
    torsional = _VINA_TORSIONAL_PATTERN.search(stdout)
    if intermol and torsional:
        return float(intermol.group(1)) + float(torsional.group(1))
    return None


def _ligand_pdbqt_from_sdf(
    sdf_path: Path, out_path: Path
) -> tuple[Path, tuple[float, float, float], tuple[float, float, float]]:
    """Convert an RDKit-readable SDF to pdbqt via meeko.

    Returns (pdbqt_path, centroid, half_extent_xyz). The centroid +
    half_extent are used to size Vina's search box for `--score_only`.
    """
    from meeko import MoleculePreparation, PDBQTWriterLegacy  # lazy
    from rdkit import Chem  # lazy

    supplier = Chem.SDMolSupplier(str(sdf_path), sanitize=True, removeHs=False)
    mol = next(iter(supplier), None)
    if mol is None:
        raise RuntimeError(f"SDF produced no molecule: {sdf_path}")
    mol = Chem.AddHs(mol, addCoords=True)
    prep = MoleculePreparation()
    setups = prep.prepare(mol)
    if not setups:
        raise RuntimeError(f"meeko produced no ligand setup for {sdf_path}")
    pdbqt_str, ok, err = PDBQTWriterLegacy.write_string(setups[0])
    if not ok:
        raise RuntimeError(f"meeko write_string failed: {err}")
    out_path.write_text(pdbqt_str, encoding="utf-8")

    conf = mol.GetConformer()
    xs = [conf.GetAtomPosition(i).x for i in range(mol.GetNumAtoms())]
    ys = [conf.GetAtomPosition(i).y for i in range(mol.GetNumAtoms())]
    zs = [conf.GetAtomPosition(i).z for i in range(mol.GetNumAtoms())]
    centroid = (sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs))
    half = (
        (max(xs) - min(xs)) / 2,
        (max(ys) - min(ys)) / 2,
        (max(zs) - min(zs)) / 2,
    )
    return out_path, centroid, half


class VinaRescorer:
    """AutoDock Vina `--score_only` rescorer.

    Converts a Boltz-2 complex PDB to ligand + receptor pdbqt (via meeko),
    runs Vina with a ligand-centered box, and parses the Estimated Free
    Energy of Binding. Lower = tighter binder (kcal/mol). Empirical force
    field — independent signal from Boltz-2's learned affinity head.

    On any conversion/exec failure, `valid=False` with a specific
    fail_reason (silent drop forbidden).
    """

    name: str = "vina_score_only"

    def __init__(
        self,
        *,
        box_padding_angstrom: float = 4.0,
        timeout_seconds: int = 60,
        vina_binary: str | None = None,
    ) -> None:
        self._box_padding = float(box_padding_angstrom)
        self._timeout = int(timeout_seconds)
        self._binary = vina_binary or shutil.which("vina")
        if self._binary is None:
            raise RuntimeError(
                "vina binary not found on PATH; install AutoDock Vina 1.2+"
            )
        # structure_to_pdbqt relies on meeko's mk_prepare_receptor.py CLI.
        # Import lazily to keep this module importable when meeko is absent.
        from .vina import structure_to_pdbqt  # lazy, avoids cycles

        self._structure_to_pdbqt = structure_to_pdbqt

    def score(self, *, complex_pdb: Path, ligand_smiles: str) -> RescoreResult:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            try:
                split = split_boltz_pdb(complex_pdb, ligand_smiles, tmpdir)
            except RuntimeError as exc:
                return RescoreResult(
                    valid=False,
                    score=None,
                    fail_reasons=(f"adapter_error: {exc}",),
                )

            try:
                ligand_pdbqt, centroid, half_extent = _ligand_pdbqt_from_sdf(
                    split.ligand_sdf, tmpdir / "ligand.pdbqt"
                )
            except Exception as exc:  # noqa: BLE001
                return RescoreResult(
                    valid=False,
                    score=None,
                    fail_reasons=(f"ligand_pdbqt_error: {type(exc).__name__}: {exc}",),
                )

            try:
                protein_pdbqt = self._structure_to_pdbqt(
                    split.protein_pdb, tmpdir / "protein.pdbqt"
                )
            except Exception as exc:  # noqa: BLE001
                return RescoreResult(
                    valid=False,
                    score=None,
                    fail_reasons=(f"protein_pdbqt_error: {type(exc).__name__}: {exc}",),
                )

            box_size = tuple(
                max(2 * h + 2 * self._box_padding, 10.0) for h in half_extent
            )
            cmd = [
                self._binary,
                "--score_only",
                "--receptor", str(protein_pdbqt),
                "--ligand", str(ligand_pdbqt),
                "--center_x", f"{centroid[0]:.3f}",
                "--center_y", f"{centroid[1]:.3f}",
                "--center_z", f"{centroid[2]:.3f}",
                "--size_x", f"{box_size[0]:.3f}",
                "--size_y", f"{box_size[1]:.3f}",
                "--size_z", f"{box_size[2]:.3f}",
            ]
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True,
                    timeout=self._timeout, check=False,
                )
            except subprocess.TimeoutExpired:
                return RescoreResult(
                    valid=False,
                    score=None,
                    fail_reasons=(f"vina_timeout: {self._timeout}s",),
                )

            if result.returncode != 0:
                return RescoreResult(
                    valid=False,
                    score=None,
                    fail_reasons=(
                        f"vina_exit_{result.returncode}: "
                        f"{(result.stderr or result.stdout)[-200:]}",
                    ),
                )

            affinity = _parse_vina_affinity(result.stdout)
            if affinity is None:
                return RescoreResult(
                    valid=False,
                    score=None,
                    fail_reasons=(
                        "vina_parse_failed: no affinity in stdout",
                    ),
                    auxiliary={"stdout_tail": result.stdout[-400:]},
                )

            return RescoreResult(
                valid=True,
                score=float(affinity),
                fail_reasons=(),
                auxiliary={"vina_affinity_kcal_mol": float(affinity)},
            )
