# pyright: reportMissingImports=false
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import molforge.docking.afdb as afdb  # noqa: E402
from molforge.docking.afdb import (  # noqa: E402
    POSEBENCH_SAMPLE_SOURCE_URL,
    normalize_prediction_payload,
    run_smoke_slice,
)

OUTPUT_DIR = REPO_ROOT / "archive" / "runs" / "phase0" / "p0-5-afdb"


def test_normalize_prediction_payload_selects_exact_uniprot_match() -> None:
    payload = [
        {
            "entryId": "AF-P61073-2-F1",
            "uniprotAccession": "P61073-2",
            "pdbUrl": "https://alphafold.ebi.ac.uk/files/AF-P61073-2-F1-model_v6.pdb",
            "cifUrl": "https://alphafold.ebi.ac.uk/files/AF-P61073-2-F1-model_v6.cif",
            "bcifUrl": "https://alphafold.ebi.ac.uk/files/AF-P61073-2-F1-model_v6.bcif",
            "latestVersion": 6,
            "gene": "CXCR4",
        },
        {
            "entryId": "AF-P61073-F1",
            "uniprotAccession": "P61073",
            "pdbUrl": "https://alphafold.ebi.ac.uk/files/AF-P61073-F1-model_v6.pdb",
            "cifUrl": "https://alphafold.ebi.ac.uk/files/AF-P61073-F1-model_v6.cif",
            "bcifUrl": "https://alphafold.ebi.ac.uk/files/AF-P61073-F1-model_v6.bcif",
            "latestVersion": 6,
            "gene": "CXCR4",
        },
    ]

    normalized_prediction = normalize_prediction_payload("P61073", payload)

    assert normalized_prediction == {
        "api_url": "https://alphafold.ebi.ac.uk/api/prediction/P61073",
        "bcif_url": "https://alphafold.ebi.ac.uk/files/AF-P61073-F1-model_v6.bcif",
        "cif_url": "https://alphafold.ebi.ac.uk/files/AF-P61073-F1-model_v6.cif",
        "entry_id": "AF-P61073-F1",
        "gene": "CXCR4",
        "latest_version": 6,
        "pdb_url": "https://alphafold.ebi.ac.uk/files/AF-P61073-F1-model_v6.pdb",
        "requested_uniprot_accession": "P61073",
        "source": "alphafold_db",
        "uniprot_accession": "P61073",
    }


def test_phase0_afdb_smoke_writes_live_artifacts() -> None:
    task_payload = run_smoke_slice("P61073", OUTPUT_DIR)

    raw_prediction_path = OUTPUT_DIR / "raw_prediction.json"
    normalized_prediction_path = OUTPUT_DIR / "normalized_prediction.json"
    posebench_source_path = OUTPUT_DIR / "posebench-source.txt"
    command_log_path = OUTPUT_DIR / "command.log"
    task_path = OUTPUT_DIR / "task.json"
    error_log_path = OUTPUT_DIR / "task-error.log"

    assert task_payload["status"] == "pass"
    assert raw_prediction_path.exists()
    assert normalized_prediction_path.exists()
    assert posebench_source_path.exists()
    assert command_log_path.exists()
    assert task_path.exists()
    assert not error_log_path.exists()

    raw_prediction = json.loads(raw_prediction_path.read_text(encoding="utf-8"))
    normalized_prediction = json.loads(
        normalized_prediction_path.read_text(encoding="utf-8")
    )
    task_file_payload = json.loads(task_path.read_text(encoding="utf-8"))
    command_log = command_log_path.read_text(encoding="utf-8")

    assert isinstance(raw_prediction, list)
    assert len(raw_prediction) >= 1
    assert normalized_prediction["requested_uniprot_accession"] == "P61073"
    assert normalized_prediction["uniprot_accession"] == "P61073"
    assert normalized_prediction["entry_id"].startswith("AF-P61073")
    assert normalized_prediction["pdb_url"].startswith(
        "https://alphafold.ebi.ac.uk/files/"
    )
    assert (
        posebench_source_path.read_text(encoding="utf-8").strip()
        == POSEBENCH_SAMPLE_SOURCE_URL
    )
    assert task_file_payload["command_log_path"].endswith("command.log")
    assert task_file_payload["normalized_prediction_path"].endswith(
        "normalized_prediction.json"
    )
    assert (
        "command=GET https://alphafold.ebi.ac.uk/api/prediction/P61073" in command_log
    )
    assert (
        "command=normalize_prediction_payload('P61073', raw_prediction_payload)"
        in command_log
    )


def test_phase0_afdb_smoke_removes_stale_error_log_on_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "afdb-success"
    output_dir.mkdir(parents=True, exist_ok=True)
    error_log_path = output_dir / "task-error.log"
    error_log_path.write_text("stale failure\n", encoding="utf-8")

    payload = [
        {
            "entryId": "AF-P61073-F1",
            "uniprotAccession": "P61073",
            "pdbUrl": "https://alphafold.ebi.ac.uk/files/AF-P61073-F1-model_v6.pdb",
            "cifUrl": "https://alphafold.ebi.ac.uk/files/AF-P61073-F1-model_v6.cif",
            "bcifUrl": "https://alphafold.ebi.ac.uk/files/AF-P61073-F1-model_v6.bcif",
            "latestVersion": 6,
            "gene": "CXCR4",
        }
    ]

    monkeypatch.setattr(
        afdb, "fetch_prediction_payload", lambda *args, **kwargs: payload
    )

    task_payload = run_smoke_slice("P61073", output_dir)

    assert task_payload["status"] == "pass"
    assert not error_log_path.exists()
    assert (output_dir / "command.log").exists()
    assert (output_dir / "task.json").exists()


def test_invalid_uniprot_writes_explicit_error_log_without_touching_success_artifacts() -> (
    None
):
    tracked_paths = [
        OUTPUT_DIR / "raw_prediction.json",
        OUTPUT_DIR / "normalized_prediction.json",
    ]
    baseline_contents = {
        path.name: path.read_text(encoding="utf-8") if path.exists() else None
        for path in tracked_paths
    }

    with pytest.raises((RuntimeError, ValueError)) as exc_info:
        run_smoke_slice("CXCR4", OUTPUT_DIR)

    error_log_path = OUTPUT_DIR / "task-error.log"
    assert error_log_path.exists()
    error_log = error_log_path.read_text(encoding="utf-8")

    assert "CXCR4" in error_log
    assert type(exc_info.value).__name__ in error_log
    assert "UniProt accession" in str(exc_info.value) or "AlphaFold DB" in str(
        exc_info.value
    )

    for path in tracked_paths:
        current_contents = path.read_text(encoding="utf-8") if path.exists() else None
        assert current_contents == baseline_contents[path.name]
