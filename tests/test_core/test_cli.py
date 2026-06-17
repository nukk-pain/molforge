# pyright: reportMissingImports=false
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from contracts.schema import (  # noqa: E402
    ADMETProfile,
    AffinityPrediction,
    BindingPocket,
    DockingPose,
    GeneratedMolecule,
    Ligand,
    OffTargetHit,
    ProteinStructure,
    RankedCandidate,
    TargetCandidate,
    StructureSource,
)
from molforge.cli import PHASE_STUB_MESSAGE, load_dotenv_file, main  # noqa: E402


def sample_candidate() -> dict[str, object]:
    return {
        "gene": {"symbol": "TGFB1", "ncbi_id": 7040, "uniprot_id": None},
        "score": 0.87,
        "evidence": [
            {
                "source": "literature",
                "description": "demo evidence",
                "confidence": 0.9,
            }
        ],
        "pathway": ["SMAD3"],
    }


def test_main_version(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["--version"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "molforge 0.1.0" in captured.out


def test_main_run_loads_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(json.dumps([sample_candidate()]) + "\n", encoding="utf-8")

    target = TargetCandidate(
        gene="TGFB1",
        score=0.87,
        disease="ALS",
        ncbi_id=7040,
        uniprot_id=None,
        evidence=[],
        pathway=["SMAD3"],
        extra=None,
    )
    ranked = RankedCandidate(
        ligand=Ligand(smiles="CCO", source="chembl_fda"),
        target=target,
        affinity=AffinityPrediction(
            ligand_smiles="CCO",
            target_gene="TGFB1",
            vina_score=-8.1,
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
        provenance={"run_id": "run-123", "stage": "phase5_pipeline"},
    )

    def fake_run_pipeline(
        candidates,
        *,
        store,
        top_n=10,
        enable_live_chembl=False,
        enable_evebio=False,
    ):
        _ = candidates, store, top_n, enable_live_chembl, enable_evebio
        return type(
            "Run",
            (),
            {
                "run_id": "run-123",
                "input_target": target,
                "started_at": "2026-04-18T00:00:00+00:00",
                "completed_at": "2026-04-18T00:01:00+00:00",
                "candidates": [ranked],
                "config_hash": "a" * 64,
                "schema_version": "2026-04-17",
            },
        )()

    monkeypatch.setattr("molforge.cli.run_pipeline", fake_run_pipeline)

    exit_code = main(["run", str(fixture_path), "--disease", "ALS"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Loaded 1 target candidates" in captured.out
    assert "Created run" in captured.out
    assert "with 1 candidates" in captured.out


def test_main_run_passes_enable_evebio_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(json.dumps([sample_candidate()]) + "\n", encoding="utf-8")
    observed: dict[str, object] = {}

    def fake_run_pipeline(
        candidates,
        *,
        store,
        top_n=10,
        enable_live_chembl=False,
        enable_evebio=False,
    ):
        observed.update({"enable_evebio": enable_evebio})
        _ = candidates, store, top_n, enable_live_chembl
        target = TargetCandidate(
            gene="TGFB1",
            score=0.87,
            disease="ALS",
            ncbi_id=7040,
            uniprot_id=None,
            evidence=[],
            pathway=["SMAD3"],
            extra=None,
        )
        return type(
            "Run",
            (),
            {
                "run_id": "run-evebio",
                "input_target": target,
                "started_at": "2026-04-18T00:00:00+00:00",
                "completed_at": "2026-04-18T00:01:00+00:00",
                "candidates": [],
                "config_hash": "a" * 64,
                "schema_version": "2026-04-17",
            },
        )()

    monkeypatch.setattr("molforge.cli.run_pipeline", fake_run_pipeline)

    exit_code = main(["run", str(fixture_path), "--enable-evebio"])
    _ = capsys.readouterr()

    assert exit_code == 0
    assert observed == {"enable_evebio": True}


def test_admet_stub_subcommands_exit_2(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["admet", "ligands.csv"])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.err.strip()


def test_dock_command_runs_runner_and_writes_predictions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(json.dumps([sample_candidate()]) + "\n", encoding="utf-8")

    class FakeRunner:
        def run(self, targets, *, store, top_n=50):
            _ = targets, store, top_n
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
            return [
                AffinityPrediction(
                    ligand_smiles="CCO",
                    target_gene="CXCR4",
                    vina_score=-8.1,
                    affinity_log_ki=None,
                    affinity_confidence=None,
                    pose_ref=pose,
                )
            ]

    monkeypatch.setattr("molforge.cli.DockingRunner", FakeRunner)
    output_path = tmp_path / "dock.json"

    exit_code = main(["dock", str(fixture_path), "--output", str(output_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload[0]["target_gene"] == "CXCR4"
    assert "Produced 1 predictions" in captured.out


def test_status_command_reads_current_status(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["status"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out.strip()


def test_load_dotenv_file_sets_missing_values_only(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("FOO=bar\nBAR=baz\n", encoding="utf-8")
    os.environ["BAR"] = "existing"

    load_dotenv_file(env_path)

    assert os.environ["FOO"] == "bar"
    assert os.environ["BAR"] == "existing"


def test_cli_generate_runs_pipeline_and_writes_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pocket_path = tmp_path / "pocket.json"
    pocket_path.write_text(
        json.dumps(
            {
                "structure": {
                    "gene": "CXCR4",
                    "uniprot": "P61073",
                    "pdb_path": "/tmp/cxcr4.pdb",
                    "source": "alphafold_db",
                    "confidence": 88.1,
                },
                "center_xyz": [1.0, 2.0, 3.0],
                "size_xyz": [10.0, 11.0, 12.0],
                "druggability_score": 0.9,
                "residues": ["ASP97"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_run_generate_stage(
        pocket: BindingPocket,
        *,
        output_dir: str | Path,
        n: int = 100,
        seed_smiles: str | None = None,
        backend=None,
    ):
        from molforge.generative.backend import GenerationArtifacts, GenerationResult

        _ = seed_smiles, backend
        return GenerationResult(
            backend="reinvent4",
            requested_count=n,
            returned_count=1,
            pocket_gene=pocket.structure.gene,
            artifacts=GenerationArtifacts(
                output_dir=str(output_dir),
                summary_path=str(Path(output_dir) / "summary.json"),
                molecules_path=str(Path(output_dir) / "molecules.json"),
                backend_run_dir=None,
            ),
            molecules=[
                GeneratedMolecule(
                    smiles="CCN",
                    qed=0.8,
                    sa_score=2.0,
                    novelty=0.7,
                    backend="reinvent4",
                    pocket_ref=pocket,
                )
            ],
        )

    monkeypatch.setattr("molforge.cli.run_generate_stage", fake_run_generate_stage)

    exit_code = main(
        [
            "generate",
            str(pocket_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--count",
            "12",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["backend"] == "reinvent4"
    assert payload["requested_count"] == 12
