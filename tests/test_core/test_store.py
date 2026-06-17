# pyright: reportMissingImports=false
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from contracts.schema import (  # noqa: E402
    ADMETProfile,
    AffinityPrediction,
    BindingPocket,
    BIOCOMPUTE_SCHEMA_VERSION,
    DockingPose,
    EvidenceItem,
    Ligand,
    OffTargetHit,
    ProteinStructure,
    RankedCandidate,
    StructureSource,
    TargetCandidate,
)
from molforge.core.store import MolforgeStore  # noqa: E402


def build_target_candidate() -> TargetCandidate:
    return TargetCandidate(
        gene="TGFB1",
        score=0.87,
        disease="ALS",
        ncbi_id=7040,
        uniprot_id=None,
        evidence=[
            EvidenceItem(
                source="literature",
                description="demo evidence",
                confidence=0.9,
            )
        ],
        pathway=["SMAD3"],
    )


def build_ranked_candidate(target: TargetCandidate) -> RankedCandidate:
    structure = ProteinStructure(
        gene=target.gene,
        uniprot=None,
        pdb_path="cache/tgfb1.pdb",
        source=StructureSource.ALPHAFOLD_DB,
        confidence=81.0,
    )
    pocket = BindingPocket(
        structure=structure,
        center_xyz=(1.0, 2.0, 3.0),
        size_xyz=(10.0, 11.0, 12.0),
        druggability_score=0.8,
        residues=["ASP123"],
    )
    pose = DockingPose(
        ligand_smiles="CCO",
        pocket=pocket,
        pose_pdb_path="poses/cc0.pdb",
        vina_score=-7.4,
        rank=1,
    )
    affinity = AffinityPrediction(
        ligand_smiles="CCO",
        target_gene=target.gene,
        vina_score=-7.4,
        affinity_log_ki=None,
        affinity_confidence=None,
        pose_ref=pose,
    )
    ligand = Ligand(smiles="CCO", source="user")
    admet = ADMETProfile(
        ligand_smiles="CCO",
        endpoints={"hERG": 0.2, "AMES": 0.1},
        liability_flags=["low_ames"],
    )
    off_target = OffTargetHit(
        ligand_smiles="CCO",
        off_target_gene="KCNH2",
        similarity=0.21,
        severity="low",
    )
    return RankedCandidate(
        ligand=ligand,
        target=target,
        affinity=affinity,
        admet=admet,
        off_targets=[off_target],
        composite_score=0.91,
        rank=1,
        provenance={"run_id": "placeholder", "stage_versions": {"phase": "p1"}},
    )


def test_store_creates_tables_and_schema_version() -> None:
    with MolforgeStore("sqlite:///:memory:") as store:
        rows = store.connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
        table_names = {str(row[0]) for row in rows}

        assert {
            "runs",
            "structures",
            "poses",
            "molecules",
            "admet_profiles",
            "rankings",
            "schema_version",
        }.issubset(table_names)

        schema_row = store.connection.execute(
            "SELECT component, version FROM schema_version WHERE component = ?",
            ("contracts",),
        ).fetchone()
        assert schema_row is not None
        assert schema_row[0] == "contracts"
        assert schema_row[1] == BIOCOMPUTE_SCHEMA_VERSION


def test_store_round_trips_pipeline_run() -> None:
    target = build_target_candidate()
    ranking = build_ranked_candidate(target)

    with MolforgeStore("sqlite:///:memory:") as store:
        run_id = store.create_run(target, "config-hash")
        ranking.provenance["run_id"] = run_id
        assert ranking.affinity is not None
        assert ranking.affinity.pose_ref is not None

        structure_id = store.save_structure(
            run_id, ranking.affinity.pose_ref.pocket.structure
        )
        pose_id = store.save_pose(run_id, ranking.affinity.pose_ref)
        molecule_id = store.save_molecule(run_id, ranking.ligand)
        admet_id = store.save_admet_profile(run_id, ranking.admet)
        ranking_id = store.save_ranking(run_id, ranking)

        assert all(
            value > 0
            for value in (structure_id, pose_id, molecule_id, admet_id, ranking_id)
        )

        loaded_run = store.load_run(run_id)

    assert loaded_run.run_id == run_id
    assert loaded_run.input_target == target
    assert loaded_run.config_hash == "config-hash"
    assert loaded_run.schema_version == BIOCOMPUTE_SCHEMA_VERSION
    assert len(loaded_run.candidates) == 1
    assert loaded_run.candidates[0] == ranking
