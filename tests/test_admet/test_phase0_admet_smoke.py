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

import molforge.admet.phase0_admet as phase0_admet  # noqa: E402
from molforge.admet.phase0_admet import (  # noqa: E402
    EXPECTED_ENDPOINT_COUNT,
    SMOKE_SMILES,
    run_smoke_slice,
)

OUTPUT_DIR = REPO_ROOT / "archive" / "runs" / "phase0" / "p0-4-admet"


def test_phase0_admet_smoke_writes_honest_artifacts() -> None:
    task_payload = run_smoke_slice(SMOKE_SMILES, OUTPUT_DIR)

    summary_path = OUTPUT_DIR / "summary.json"
    task_path = OUTPUT_DIR / "task.json"
    error_log_path = OUTPUT_DIR / "task-error.log"
    predictions_path = OUTPUT_DIR / "predictions.json"
    endpoint_keys_path = OUTPUT_DIR / "endpoint-keys.txt"
    command_log_path = OUTPUT_DIR / "command.log"

    assert summary_path.exists()
    assert task_path.exists()
    assert command_log_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    task_file_payload = json.loads(task_path.read_text(encoding="utf-8"))
    command_log = command_log_path.read_text(encoding="utf-8")

    assert task_file_payload == task_payload
    assert summary["task"] == "P0-4 ADMET-AI v2 41-endpoint smoke verification"
    assert summary["source"] == "admet_ai_v2"
    assert summary["smiles"] == SMOKE_SMILES
    assert summary["expected_endpoint_count"] == EXPECTED_ENDPOINT_COUNT
    assert summary["command_log_path"].endswith("command.log")
    assert summary["predictions_path"].endswith("predictions.json")
    assert summary["endpoint_keys_path"].endswith("endpoint-keys.txt")
    assert task_payload["command_log_path"].endswith("command.log")
    assert task_payload["predictions_path"].endswith("predictions.json")
    assert task_payload["endpoint_keys_path"].endswith("endpoint-keys.txt")
    assert (
        "probe_execution_mode=in_process" in command_log
        or "probe_execution_mode=external_python" in command_log
    )
    assert (
        f"planned_probe_path=import admet_ai -> ADMETModel() -> ADMETModel.predict({SMOKE_SMILES!r})"
        in command_log
    )

    if task_payload["status"] == "pass":
        assert predictions_path.exists()
        assert endpoint_keys_path.exists()
        assert not error_log_path.exists()

        predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
        endpoint_keys = [
            line.strip()
            for line in endpoint_keys_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        assert isinstance(predictions, dict)
        assert predictions
        assert endpoint_keys == sorted(endpoint_keys)
        assert summary["status"] == "pass"
        assert summary["blocker"] is False
        assert summary["predictions_available"] is True
        assert summary["endpoint_count_matches_expectation"] is True
        assert len(endpoint_keys) == EXPECTED_ENDPOINT_COUNT
        assert summary["observed_endpoint_count"] == len(endpoint_keys)
        assert task_payload["observed_endpoint_count"] == len(endpoint_keys)
        assert "final_status=pass" in command_log
        assert "step=predict_smiles" in command_log
    else:
        assert task_payload["status"] == "blocked"
        assert summary["status"] == "blocked"
        assert summary["blocker"] is True
        assert error_log_path.exists()

        error_log = error_log_path.read_text(encoding="utf-8")

        if summary["failure_kind"] == "missing_dependency":
            assert summary["predictions_available"] is False
            assert summary["observed_endpoint_count"] == 0
            assert predictions_path.exists()
            assert endpoint_keys_path.exists()

            predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
            endpoint_key_lines = endpoint_keys_path.read_text(
                encoding="utf-8"
            ).splitlines()

            assert predictions["task"] == summary["task"]
            assert predictions["source"] == summary["source"]
            assert predictions["status"] == "blocked"
            assert predictions["failure_kind"] == "missing_dependency"
            assert predictions["smiles"] == SMOKE_SMILES
            assert predictions["predictions_available"] is False
            assert predictions["error_type"] == "RuntimeError"
            assert "admet_ai" in predictions["error_message"]
            assert endpoint_key_lines
            assert all(line.startswith("#") for line in endpoint_key_lines)
            assert any(
                "blocked evidence file: no endpoint keys were produced." in line
                for line in endpoint_key_lines
            )
            assert any(
                "failure_kind=missing_dependency" in line for line in endpoint_key_lines
            )
            assert any(f"smiles={SMOKE_SMILES}" in line for line in endpoint_key_lines)
            assert any("admet_ai" in line for line in endpoint_key_lines)
            assert "error_type=RuntimeError" in error_log
            assert "failure_kind=missing_dependency" in error_log
            assert "admet_ai" in error_log
            assert "final_status=blocked" in command_log
            assert "outcome=error" in command_log
            assert "admet_ai" in command_log
        else:
            assert summary["failure_kind"] == "endpoint_count_mismatch"
            assert summary["predictions_available"] is True
            assert summary["endpoint_count_matches_expectation"] is False
            assert predictions_path.exists()
            assert endpoint_keys_path.exists()
            endpoint_keys = [
                line.strip()
                for line in endpoint_keys_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            assert summary["observed_endpoint_count"] == len(endpoint_keys)
            assert len(endpoint_keys) != EXPECTED_ENDPOINT_COUNT
            assert "failure_kind=endpoint_count_mismatch" in error_log
            assert f"expected_endpoint_count={EXPECTED_ENDPOINT_COUNT}" in error_log
            assert "final_status=blocked" in command_log
            assert "step=collect_endpoint_keys" in command_log


def test_phase0_admet_smoke_requires_exact_endpoint_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mismatch_smiles = "CCN"
    expected_prediction = {"smiles": SMOKE_SMILES} | {
        f"endpoint_{index:02d}": float(index)
        for index in range(EXPECTED_ENDPOINT_COUNT)
    }
    short_prediction = {"smiles": mismatch_smiles} | {
        f"endpoint_{index:02d}": float(index)
        for index in range(EXPECTED_ENDPOINT_COUNT - 1)
    }

    monkeypatch.setattr(
        phase0_admet,
        "predict_smiles",
        lambda smiles: (
            expected_prediction if smiles == SMOKE_SMILES else short_prediction
        ),
    )

    pass_output_dir = tmp_path / "pass-run"
    mismatch_output_dir = tmp_path / "mismatch-run"

    pass_task = run_smoke_slice(SMOKE_SMILES, pass_output_dir)
    mismatch_task = run_smoke_slice(mismatch_smiles, mismatch_output_dir)

    pass_summary = json.loads(
        (pass_output_dir / "summary.json").read_text(encoding="utf-8")
    )
    mismatch_summary = json.loads(
        (mismatch_output_dir / "summary.json").read_text(encoding="utf-8")
    )
    mismatch_error_log = (mismatch_output_dir / "task-error.log").read_text(
        encoding="utf-8"
    )
    pass_command_log = (pass_output_dir / "command.log").read_text(encoding="utf-8")
    mismatch_command_log = (mismatch_output_dir / "command.log").read_text(
        encoding="utf-8"
    )

    assert pass_task["status"] == "pass"
    assert pass_summary["endpoint_count_matches_expectation"] is True
    assert pass_summary["observed_endpoint_count"] == EXPECTED_ENDPOINT_COUNT
    assert not (pass_output_dir / "task-error.log").exists()
    assert pass_task["command_log_path"].endswith("command.log")
    assert pass_summary["command_log_path"].endswith("command.log")
    assert "final_status=pass" in pass_command_log
    assert "step=collect_endpoint_keys" in pass_command_log

    assert mismatch_task["status"] == "blocked"
    assert mismatch_summary["failure_kind"] == "endpoint_count_mismatch"
    assert mismatch_summary["predictions_available"] is True
    assert mismatch_summary["endpoint_count_matches_expectation"] is False
    assert mismatch_summary["observed_endpoint_count"] == EXPECTED_ENDPOINT_COUNT - 1
    assert mismatch_task["command_log_path"].endswith("command.log")
    assert mismatch_summary["command_log_path"].endswith("command.log")
    assert "final_status=blocked" in mismatch_command_log
    assert "failure_kind=endpoint_count_mismatch" in mismatch_command_log
    assert "failure_kind=endpoint_count_mismatch" in mismatch_error_log
    assert (
        f"observed_endpoint_count={EXPECTED_ENDPOINT_COUNT - 1}" in mismatch_error_log
    )
