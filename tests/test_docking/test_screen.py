# pyright: reportMissingImports=false
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from contracts.schema import (
    BindingPocket,
    DockingPose,
    Ligand,
    ProteinStructure,
    StructureSource,
)  # noqa: E402
from molforge.docking import screen  # noqa: E402


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
        druggability_score=0.7,
        residues=["ASP97"],
    )


def test_repurpose_screen_returns_top_ranked_predictions(monkeypatch) -> None:
    pocket = build_pocket()
    library = [
        Ligand(smiles=f"C{i}", source="chembl_fda", chembl_id=f"CHEMBL{i:03d}")
        for i in range(100)
    ]

    def fake_dock(
        bound_pocket: BindingPocket, ligand_smiles: str, output_dir: Path | None = None
    ):
        index = int(ligand_smiles[1:])
        _ = output_dir
        return [
            DockingPose(
                ligand_smiles=ligand_smiles,
                pocket=bound_pocket,
                pose_pdb_path=f"/tmp/{ligand_smiles}.pdbqt",
                vina_score=-float(index),
                rank=1,
            )
        ]

    monkeypatch.setattr(screen, "run_dock", fake_dock)

    predictions = screen.repurpose_screen(
        pocket,
        library,
        top_n=10,
        max_parallel=4,
        output_dir=Path("/tmp"),
    )

    assert len(predictions) == 10
    assert [prediction.vina_score for prediction in predictions] == [
        -99.0,
        -98.0,
        -97.0,
        -96.0,
        -95.0,
        -94.0,
        -93.0,
        -92.0,
        -91.0,
        -90.0,
    ]
    assert all(prediction.target_gene == "CXCR4" for prediction in predictions)


def test_repurpose_screen_skips_failed_ligands(monkeypatch) -> None:
    pocket = build_pocket()
    library = [
        Ligand(smiles="BAD", source="chembl_fda", chembl_id="CHEMBL_BAD"),
        Ligand(smiles="GOOD", source="chembl_fda", chembl_id="CHEMBL_GOOD"),
    ]

    def fake_dock(
        bound_pocket: BindingPocket, ligand_smiles: str, output_dir: Path | None = None
    ):
        _ = output_dir
        if ligand_smiles == "BAD":
            raise RuntimeError("ligand preparation failed")
        return [
            DockingPose(
                ligand_smiles=ligand_smiles,
                pocket=bound_pocket,
                pose_pdb_path=f"/tmp/{ligand_smiles}.pdbqt",
                vina_score=-7.5,
                rank=1,
            )
        ]

    monkeypatch.setattr(screen, "run_dock", fake_dock)

    predictions = screen.repurpose_screen(
        pocket,
        library,
        top_n=10,
        max_parallel=2,
        output_dir=Path("/tmp"),
    )

    assert len(predictions) == 1
    assert predictions[0].ligand_smiles == "GOOD"
