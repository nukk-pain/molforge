# pyright: reportMissingImports=false
from __future__ import annotations

import json
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
    DockingPose,
    Ligand,
    OffTargetHit,
    PipelineRun,
    ProteinStructure,
    RankedCandidate,
    StructureSource,
    TargetCandidate,
)
from molforge.core.multi_target_batch import run_batch  # noqa: E402


def _write_target_input(path: Path, gene: str, score: float = 0.7) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "gene": {"symbol": gene, "ncbi_id": 123},
                    "score": score,
                    "evidence": [],
                    "pathway": [],
                }
            ]
        ),
        encoding="utf-8",
    )


def _build_run(target_gene: str, smiles_list: list[str]) -> PipelineRun:
    candidates: list[RankedCandidate] = []
    for rank, smiles in enumerate(smiles_list, start=1):
        target = TargetCandidate(gene=target_gene, score=0.7, disease="test")
        structure = ProteinStructure(
            gene=target_gene, uniprot=None, pdb_path="/tmp/x.pdb",
            source=StructureSource.ALPHAFOLD_DB, confidence=80.0,
        )
        pocket = BindingPocket(
            structure=structure, center_xyz=(0.0, 0.0, 0.0),
            size_xyz=(10.0, 10.0, 10.0),
        )
        candidates.append(
            RankedCandidate(
                ligand=Ligand(smiles=smiles, source="generative:mock"),
                target=target,
                affinity=AffinityPrediction(
                    ligand_smiles=smiles, target_gene=target_gene, vina_score=-7.5,
                    pose_ref=DockingPose(
                        ligand_smiles=smiles, pocket=pocket,
                        pose_pdb_path=f"/tmp/{rank}.pdb", vina_score=-7.5, rank=rank,
                    ),
                ),
                admet=ADMETProfile(ligand_smiles=smiles, endpoints={"AMES": 0.1}),
                off_targets=[],
                composite_score=1.0 - 0.1 * rank,
                rank=rank,
                provenance={"mock": True},
            )
        )
    return PipelineRun(
        run_id=f"mock-{target_gene}", input_target=candidates[0].target,
        started_at="2026-04-20T00:00:00Z", completed_at="2026-04-20T00:01:00Z",
        candidates=candidates, config_hash="deadbeef",
    )


def test_run_batch_collects_completed_runs(tmp_path: Path) -> None:
    targets_dir = tmp_path / "inputs"
    targets_dir.mkdir()
    _write_target_input(targets_dir / "A.json", "A")
    _write_target_input(targets_dir / "B.json", "B")

    canned_runs = {"A": _build_run("A", ["CCO", "CCN"]), "B": _build_run("B", ["CCN", "CCC"])}

    def fake_pipeline(targets, *, store, top_n=10, enable_live_chembl=False):
        return canned_runs[targets[0].gene]

    result = run_batch(
        sorted(targets_dir.glob("*.json")),
        disease="demo",
        store_root=tmp_path / "store",
        top_n=5,
        pipeline_runner=fake_pipeline,
    )

    assert len(result.per_target) == 2
    assert all(r.status == "completed" for r in result.per_target)
    assert result.total_candidates == 4
    assert result.unique_smiles_count == 3
    assert result.shared_smiles_across_targets == ["CCN"]
    assert result.per_target_top_smiles["A"][:2] == ["CCO", "CCN"]


def test_run_batch_survives_single_target_failure(tmp_path: Path) -> None:
    targets_dir = tmp_path / "inputs"
    targets_dir.mkdir()
    _write_target_input(targets_dir / "A.json", "A")
    _write_target_input(targets_dir / "B.json", "B")

    def fake_pipeline(targets, *, store, top_n=10, enable_live_chembl=False):
        gene = targets[0].gene
        if gene == "A":
            raise RuntimeError("docking unavailable")
        return _build_run("B", ["CCN"])

    result = run_batch(
        sorted(targets_dir.glob("*.json")),
        disease="demo",
        store_root=tmp_path / "store",
        top_n=5,
        pipeline_runner=fake_pipeline,
    )

    assert len(result.per_target) == 2
    a_row = next(r for r in result.per_target if r.target_gene == "A")
    b_row = next(r for r in result.per_target if r.target_gene == "B")
    assert a_row.status == "failed:RuntimeError"
    assert a_row.error_message == "docking unavailable"
    assert b_row.status == "completed"
    assert result.total_candidates == 1
