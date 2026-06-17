# pyright: reportMissingImports=false
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from contracts.schema import BindingPocket, ProteinStructure, StructureSource  # noqa: E402
from molforge.generative.filters import (  # noqa: E402
    calculate_novelty,
    calculate_qed,
    calculate_sa_score,
    filter_generated_smiles,
)


def build_pocket() -> BindingPocket:
    return BindingPocket(
        structure=ProteinStructure(
            gene="CXCR4",
            uniprot="P61073",
            pdb_path="/tmp/cxcr4.pdb",
            source=StructureSource.ALPHAFOLD_DB,
            confidence=91.2,
        ),
        center_xyz=(1.0, 2.0, 3.0),
        size_xyz=(10.0, 11.0, 12.0),
        druggability_score=0.9,
        residues=["ASP97", "TYR116"],
    )


def test_filter_generated_smiles_deduplicates_and_applies_thresholds() -> None:
    molecules = filter_generated_smiles(
        ["CCN", "CCN", "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"],
        reference_smiles=["CCO"],
        backend="reinvent4",
        pocket_ref=build_pocket(),
    )

    assert len(molecules) == 1
    assert molecules[0].smiles == "CCN"
    assert molecules[0].backend == "reinvent4"
    assert molecules[0].pocket_ref is not None
    assert molecules[0].qed >= 0.5
    assert molecules[0].sa_score <= 4.0


def test_filter_scores_return_bounded_values() -> None:
    assert 0.0 <= calculate_qed("CCN") <= 1.0
    assert 1.0 <= calculate_sa_score("CCN") <= 10.0
    assert 0.0 <= calculate_novelty("CCN", ["CCO", "CCN"]) <= 1.0
