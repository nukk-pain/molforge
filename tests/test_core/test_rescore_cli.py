# pyright: reportMissingImports=false
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from contracts.schema import (
    ADMETProfile,
    AffinityPrediction,
    Ligand,
    OffTargetHit,
    PipelineRun,
    RankedCandidate,
    TargetCandidate,
)  # noqa: E402
from molforge.cli import main  # noqa: E402


def build_run() -> PipelineRun:
    target = TargetCandidate(
        gene="CXCR4",
        score=0.9,
        disease="ALS",
        ncbi_id=7852,
        uniprot_id="P61073",
        evidence=[],
        pathway=[],
        extra=None,
    )
    candidate = RankedCandidate(
        ligand=Ligand(smiles="CCO", source="chembl_fda"),
        target=target,
        affinity=AffinityPrediction(
            ligand_smiles="CCO",
            target_gene="CXCR4",
            vina_score=-8.2,
            affinity_log_ki=None,
            affinity_confidence=None,
            pose_ref=None,
        ),
        admet=ADMETProfile(
            ligand_smiles="CCO", endpoints={"hERG": 0.2}, liability_flags=[]
        ),
        off_targets=[
            OffTargetHit(
                ligand_smiles="CCO",
                off_target_gene="KCNH2",
                similarity=0.2,
                severity="low",
            )
        ],
        composite_score=0.8,
        rank=1,
        provenance={"run_id": "old-run", "stage": "phase5_pipeline"},
    )
    return PipelineRun(
        run_id="old-run",
        input_target=target,
        started_at="2026-04-20T00:00:00+00:00",
        completed_at="2026-04-20T00:01:00+00:00",
        candidates=[candidate],
        config_hash="a" * 64,
    )


def test_rescore_cli_writes_new_run_artifact(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    run = build_run()
    input_path = tmp_path / "run.json"
    output_path = tmp_path / "rescored.json"
    input_path.write_text(
        json.dumps(
            {
                "run_id": run.run_id,
                "started_at": run.started_at,
                "completed_at": run.completed_at,
                "config_hash": run.config_hash,
                "schema_version": run.schema_version,
                "input_target": run.input_target.__dict__,
                "candidates": [
                    {
                        "ligand": run.candidates[0].ligand.__dict__,
                        "target": run.candidates[0].target.__dict__,
                        "affinity": {
                            "ligand_smiles": "CCO",
                            "target_gene": "CXCR4",
                            "vina_score": -8.2,
                            "affinity_log_ki": None,
                            "affinity_confidence": None,
                            "pose_ref": None,
                        },
                        "admet": {
                            "ligand_smiles": "CCO",
                            "endpoints": {"hERG": 0.2},
                            "liability_flags": [],
                        },
                        "off_targets": [
                            {
                                "ligand_smiles": "CCO",
                                "off_target_gene": "KCNH2",
                                "similarity": 0.2,
                                "severity": "low",
                            }
                        ],
                        "composite_score": 0.8,
                        "rank": 1,
                        "provenance": {"run_id": "old-run", "stage": "phase5_pipeline"},
                    }
                ],
                "provenance": {"run_id": run.run_id},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_rescore_predictions(predictions, *, backend, batch_size=10):
        _ = backend, batch_size
        rescored = [
            AffinityPrediction(
                ligand_smiles=predictions[0].ligand_smiles,
                target_gene=predictions[0].target_gene,
                vina_score=predictions[0].vina_score,
                affinity_log_ki=-9.1,
                affinity_confidence=0.87,
                pose_ref=predictions[0].pose_ref,
            )
        ]
        return rescored, 0.5

    monkeypatch.setattr("molforge.cli.rescore_predictions", fake_rescore_predictions)
    monkeypatch.setattr("molforge.cli.build_remote_backend", lambda: object())

    exit_code = main(
        [
            "rescore",
            str(input_path),
            "--db-path",
            str(tmp_path / "rescore.db"),
            "--output",
            str(output_path),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["run_id"] != "old-run"
    assert payload["provenance"]["rescored_from"] == "old-run"
    assert payload["candidates"][0]["affinity"]["affinity_log_ki"] == -9.1
    assert payload["candidates"][0]["provenance"]["rescored_from"] == "old-run"
    assert "Created run" in captured.out


def test_rescore_cli_returns_exit_2_for_missing_run_id(
    tmp_path: Path,
    capsys,
) -> None:
    exit_code = main(
        [
            "rescore",
            "missing-run-id",
            "--db-path",
            str(tmp_path / "missing.db"),
            "--output",
            str(tmp_path / "rescored.json"),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "missing-run-id" in captured.err
