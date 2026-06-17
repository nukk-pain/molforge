# pyright: reportMissingImports=false
from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from molforge.docking.boltz2_poc import (  # noqa: E402
    PREDICTION_OUTPUT_DIRNAME,
    SMOKE_INPUT_FILENAME,
    SOURCE_NAME,
    TASK_NAME,
    build_inference_command,
    run_smoke_slice,
)

OUTPUT_DIR = REPO_ROOT / "archive" / "runs" / "phase0" / "p0-1-boltz2"


def test_phase0_boltz2_smoke_writes_honest_artifacts() -> None:
    task_payload = run_smoke_slice(OUTPUT_DIR)

    command_log_path = OUTPUT_DIR / "command.log"
    runtime_path = OUTPUT_DIR / "runtime.json"
    decision_path = OUTPUT_DIR / "decision.json"
    task_path = OUTPUT_DIR / "task.json"
    error_log_path = OUTPUT_DIR / "task-error.log"

    assert command_log_path.exists()
    assert runtime_path.exists()
    assert decision_path.exists()
    assert task_path.exists()

    runtime_payload = json.loads(runtime_path.read_text(encoding="utf-8"))
    decision_payload = json.loads(decision_path.read_text(encoding="utf-8"))
    task_file_payload = json.loads(task_path.read_text(encoding="utf-8"))
    command_log = command_log_path.read_text(encoding="utf-8")

    assert task_file_payload == task_payload
    assert runtime_payload["task"] == TASK_NAME
    assert runtime_payload["source"] == SOURCE_NAME
    assert decision_payload["task"] == TASK_NAME
    assert decision_payload["source"] == SOURCE_NAME
    assert task_payload["status"] in {"pass", "blocked", "fallback-approved"}
    assert runtime_payload["status"] == task_payload["status"]
    assert decision_payload["status"] == task_payload["status"]
    assert "pip install boltz[cuda] -U" in command_log

    if task_payload["status"] == "pass":
        assert runtime_payload["failure_class"] is None
        assert runtime_payload["output_files_present"] is True
        assert runtime_payload["output_files"]
        assert not error_log_path.exists()
    else:
        assert error_log_path.exists()
        assert runtime_payload["failure_class"] in {
            "environment-baseline",
            "boltz-runtime",
            "internal-error",
        }
        error_log = error_log_path.read_text(encoding="utf-8")
        assert "error_type=RuntimeError" in error_log

        if task_payload["status"] == "fallback-approved":
            assert runtime_payload["failure_class"] == "environment-baseline"
            assert decision_payload["fallback_candidate"] == "DiffDock-L"
            assert "DiffDock-L" in decision_payload["rationale"]
        elif task_payload["status"] == "blocked":
            assert runtime_payload["failure_class"] in {
                "boltz-runtime",
                "internal-error",
            }


def test_phase0_boltz2_smoke_supports_pass_and_blocked_paths_with_injected_executor(
    tmp_path: Path,
) -> None:
    pass_output_dir = tmp_path / "pass-run"
    blocked_output_dir = tmp_path / "blocked-run"

    def fake_success_executor(
        command: Sequence[str],
        timeout_seconds: float,
        cwd: Path | None,
    ) -> subprocess.CompletedProcess[str]:
        if command[:2] == [sys.executable, "-c"] and "boltz_spec_found" in command[2]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps(
                    {
                        "boltz_spec_found": True,
                        "boltz_origin": "/fake/site-packages/boltz",
                    }
                )
                + "\n",
                stderr="",
            )
        if command[:2] == [sys.executable, "-c"] and "torch_spec_found" in command[2]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps(
                    {
                        "torch_spec_found": True,
                        "torch_version": "2.5.0",
                        "cuda_available": True,
                        "cuda_device_count": 1,
                        "cuda_device_name": "Fake GPU",
                        "cuda_total_memory_bytes": 17179869184,
                    }
                )
                + "\n",
                stderr="",
            )

        out_dir = Path(command[command.index("--out_dir") + 1])
        prediction_dir = out_dir / "predictions" / Path(command[2]).stem
        prediction_dir.mkdir(parents=True, exist_ok=True)
        _ = (prediction_dir / "affinity_minimal-affinity-input.json").write_text(
            json.dumps({"affinity_pred_value": -1.2}) + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

    def fake_blocked_executor(
        command: Sequence[str],
        timeout_seconds: float,
        cwd: Path | None,
    ) -> subprocess.CompletedProcess[str]:
        if command[:2] == [sys.executable, "-c"] and "boltz_spec_found" in command[2]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps(
                    {
                        "boltz_spec_found": True,
                        "boltz_origin": "/fake/site-packages/boltz",
                    }
                )
                + "\n",
                stderr="",
            )
        if command[:2] == [sys.executable, "-c"] and "torch_spec_found" in command[2]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps(
                    {
                        "torch_spec_found": True,
                        "torch_version": "2.5.0",
                        "cuda_available": True,
                        "cuda_device_count": 1,
                        "cuda_device_name": "Fake GPU",
                        "cuda_total_memory_bytes": 17179869184,
                    }
                )
                + "\n",
                stderr="",
            )
        return subprocess.CompletedProcess(
            command,
            1,
            stdout="",
            stderr="MSA server request failed during Boltz predict",
        )

    fake_cli_path = "/fake/bin/boltz"
    pass_task = run_smoke_slice(
        pass_output_dir,
        command_executor=fake_success_executor,
        which_resolver=lambda _: fake_cli_path,
    )
    blocked_task = run_smoke_slice(
        blocked_output_dir,
        command_executor=fake_blocked_executor,
        which_resolver=lambda _: fake_cli_path,
    )

    assert pass_task["status"] == "pass"
    assert blocked_task["status"] == "blocked"

    pass_runtime = json.loads(
        (pass_output_dir / "runtime.json").read_text(encoding="utf-8")
    )
    blocked_runtime = json.loads(
        (blocked_output_dir / "runtime.json").read_text(encoding="utf-8")
    )
    blocked_decision = json.loads(
        (blocked_output_dir / "decision.json").read_text(encoding="utf-8")
    )
    pass_command_log = (pass_output_dir / "command.log").read_text(encoding="utf-8")

    assert pass_runtime["output_files_present"] is True
    assert pass_runtime["output_files"] == [
        "predictions/minimal-affinity-input/affinity_minimal-affinity-input.json"
    ]
    assert blocked_runtime["failure_class"] == "boltz-runtime"
    assert blocked_runtime["failure_kind"] == "msa_server_failure"
    assert blocked_decision["fallback_candidate"] is None
    assert SMOKE_INPUT_FILENAME in pass_command_log
    assert (
        build_inference_command(
            boltz_cli_path=fake_cli_path,
            input_path=pass_output_dir / SMOKE_INPUT_FILENAME,
            output_dir=pass_output_dir / PREDICTION_OUTPUT_DIRNAME,
        )[0]
        in pass_command_log
    )
