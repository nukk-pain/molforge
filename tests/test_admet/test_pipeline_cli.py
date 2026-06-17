from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

from contracts.schema import (
    ADMETProfile,
    AffinityPrediction,
    Ligand,
    OffTargetHit,
    RankedCandidate,
    TargetCandidate,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

main = importlib.import_module("molforge.cli").main


def test_cli_run_uses_current_target_candidate_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture_path = tmp_path / "targets.json"
    fixture_path.write_text(
        json.dumps(
            [
                {
                    "gene": {
                        "symbol": "CXCR4",
                        "ncbi_id": 7852,
                        "uniprot_id": "P61073",
                    },
                    "score": 0.9,
                    "evidence": [
                        {
                            "source": "demo",
                            "description": "evidence",
                            "confidence": 0.9,
                        }
                    ],
                    "pathway": ["chemokine"],
                }
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    target = TargetCandidate(
        gene="CXCR4",
        score=0.9,
        disease="pain",
        ncbi_id=7852,
        uniprot_id="P61073",
        evidence=[],
        pathway=["chemokine"],
        extra=None,
    )
    ranked = RankedCandidate(
        ligand=Ligand(smiles="CCO", source="chembl_fda"),
        target=target,
        affinity=AffinityPrediction(
            ligand_smiles="CCO",
            target_gene="CXCR4",
            vina_score=-8.0,
            affinity_log_ki=None,
            affinity_confidence=0.8,
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
        provenance={"run_id": "phase5-test", "stage": "phase5_pipeline"},
    )

    observed: dict[str, object] = {}

    monkeypatch.setattr(
        "molforge.cli.run_pipeline",
        lambda candidates, *, store, top_n=10, enable_live_chembl=False, enable_evebio=False: (
            observed.update(
                {
                    "candidate_count": len(candidates),
                    "top_n": top_n,
                    "enable_live_chembl": enable_live_chembl,
                    "enable_evebio": enable_evebio,
                }
            )
            or type(
                "Run",
                (),
                {
                    "run_id": "phase5-test",
                    "input_target": target,
                    "started_at": "2026-04-18T00:00:00+00:00",
                    "completed_at": "2026-04-18T00:01:00+00:00",
                    "candidates": [ranked],
                    "config_hash": "a" * 64,
                    "schema_version": "2026-04-17",
                },
            )()
        ),
    )

    exit_code = main(
        ["run", str(fixture_path), "--disease", "pain", "--enable-live-chembl"]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Loaded 1 target candidates" in captured.out
    assert observed == {
        "candidate_count": 1,
        "top_n": 10,
        "enable_live_chembl": True,
        "enable_evebio": False,
    }


def test_cli_admet_reads_csv_and_writes_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    csv_path = tmp_path / "ligands.csv"
    csv_path.write_text(
        "smiles,source,vina_score,affinity_confidence,target_gene\nCCO,user,-8.0,0.8,CXCR4\n",
        encoding="utf-8",
    )

    class FakeStore:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

    monkeypatch.setattr("molforge.cli.MolforgeStore", lambda path: FakeStore())
    monkeypatch.setattr(
        "molforge.cli.run_admet_pipeline",
        lambda **kwargs: type(
            "Run",
            (),
            {
                "run_id": "phase4-test",
                "started_at": "2026-04-18T00:00:00+00:00",
                "completed_at": None,
                "config_hash": "abc123",
                "schema_version": "2026-04-17",
                "input_target": kwargs["target"],
                "candidates": [],
            },
        )(),
    )

    output_path = tmp_path / "ranked.json"
    assert main(["admet", str(csv_path), "--output", str(output_path)]) == 0
    captured = capsys.readouterr()
    assert "Created run phase4-test" in captured.out
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["run_id"] == "phase4-test"
    assert payload["input_target"]["gene"] == "CXCR4"


def test_cli_admet_passes_enable_evebio_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path = tmp_path / "ligands.csv"
    csv_path.write_text(
        "smiles,source,vina_score,affinity_confidence,target_gene\nCCO,user,-8.0,0.8,CXCR4\n",
        encoding="utf-8",
    )
    observed: dict[str, object] = {}

    class FakeStore:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

    monkeypatch.setattr("molforge.cli.MolforgeStore", lambda path: FakeStore())
    monkeypatch.setattr(
        "molforge.cli.run_admet_pipeline",
        lambda **kwargs: observed.update({"enable_evebio": kwargs["enable_evebio"]})
        or type(
            "Run",
            (),
            {
                "run_id": "phase4-evebio",
                "started_at": "2026-04-18T00:00:00+00:00",
                "completed_at": None,
                "config_hash": "abc123",
                "schema_version": "2026-04-17",
                "input_target": kwargs["target"],
                "candidates": [],
            },
        )(),
    )

    assert main(["admet", str(csv_path), "--enable-evebio"]) == 0
    assert observed == {"enable_evebio": True}


def test_cli_admet_reports_missing_runtime_cleanly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    csv_path = tmp_path / "ligands.csv"
    csv_path.write_text(
        "smiles,source,vina_score,affinity_confidence,target_gene\nCCO,user,-8.0,0.8,CXCR4\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "molforge.cli.run_admet_command",
        lambda args: (_ for _ in ()).throw(RuntimeError("missing admet runtime")),
    )

    exit_code = main(["admet", str(csv_path)])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "missing admet runtime" in captured.err


def test_cli_admet_rejects_mixed_target_genes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    csv_path = tmp_path / "ligands.csv"
    csv_path.write_text(
        "smiles,source,vina_score,affinity_confidence,target_gene\nCCO,user,-8.0,0.8,CXCR4\nCCN,user,-7.0,0.7,CCR5\n",
        encoding="utf-8",
    )

    exit_code = main(["admet", str(csv_path)])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "single target_gene" in captured.err
