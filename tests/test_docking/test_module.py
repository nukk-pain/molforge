# pyright: reportMissingImports=false
from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from contracts.schema import (
    AffinityPrediction,
    BindingPocket,
    DockingPose,
    Ligand,
    ProteinStructure,
    StructureSource,
    TargetCandidate,
)  # noqa: E402
from molforge.core.store import MolforgeStore  # noqa: E402
from molforge.docking.module import DockingRunner  # noqa: E402


def build_target() -> TargetCandidate:
    return TargetCandidate(
        gene="CXCR4",
        score=0.92,
        disease="ALS",
        ncbi_id=7852,
        uniprot_id="P61073",
        evidence=[],
        pathway=[],
        extra=None,
    )


def test_docking_runner_saves_structure_and_pose(monkeypatch, tmp_path: Path) -> None:
    runner = DockingRunner()
    target = build_target()
    structure = ProteinStructure(
        gene="CXCR4",
        uniprot="P61073",
        pdb_path="/tmp/cxcr4.pdb",
        source=StructureSource.ALPHAFOLD_DB,
        confidence=88.1,
    )
    pocket = BindingPocket(
        structure=structure,
        center_xyz=(1.0, 2.0, 3.0),
        size_xyz=(10.0, 11.0, 12.0),
        druggability_score=0.9,
        residues=["ASP97"],
    )
    pose = DockingPose(
        ligand_smiles="CCO",
        pocket=pocket,
        pose_pdb_path="/tmp/pose.pdbqt",
        vina_score=-8.1,
        rank=1,
    )
    prediction = AffinityPrediction(
        ligand_smiles="CCO",
        target_gene="CXCR4",
        vina_score=-8.1,
        affinity_log_ki=None,
        affinity_confidence=None,
        pose_ref=pose,
    )

    async def fake_fetch_alphafold_structure(uniprot: str, *, cache_dir: Path):
        _ = uniprot, cache_dir
        return structure

    def fake_import_module(name: str):
        if name == "molforge.docking._env_check":
            return SimpleNamespace(
                require_vina_binary=lambda: None,
                require_meeko_available=lambda: None,
            )
        if name == "molforge.docking.chembl":
            return SimpleNamespace(
                load_fda_approved_library=lambda **kwargs: [
                    Ligand(smiles="CCO", source="chembl_fda", chembl_id="CHEMBL1")
                ]
            )
        if name == "molforge.docking.gene_mapping":
            return SimpleNamespace(resolve_uniprot=lambda candidate: "P61073")
        if name == "molforge.docking.structure":
            return SimpleNamespace(
                MissingStructureError=RuntimeError,
                fetch_alphafold_structure=fake_fetch_alphafold_structure,
            )
        if name == "molforge.docking.pocket":
            return SimpleNamespace(
                detect_pocket=lambda bound_structure, use_fpocket=True: pocket
            )
        if name == "molforge.docking.screen":
            return SimpleNamespace(
                repurpose_screen=lambda bound_pocket, library, top_n, max_parallel, output_dir=None: [
                    prediction
                ]
            )
        raise AssertionError(f"Unexpected module import: {name}")

    monkeypatch.setattr(
        "molforge.docking.module.importlib.import_module", fake_import_module
    )

    with MolforgeStore(tmp_path / "dock.db") as store:
        predictions = runner.run([target], store=store, top_n=5)
        structure_rows = store.connection.execute(
            "SELECT structure_json FROM structures"
        ).fetchall()
        pose_rows = store.connection.execute("SELECT pose_json FROM poses").fetchall()

    assert len(predictions) == 1
    assert len(structure_rows) == 1
    assert len(pose_rows) == 1
