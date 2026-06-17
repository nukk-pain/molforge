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

from molforge.docking.openfold3_poc import (  # noqa: E402
    OFFICIAL_QUERY_RELATIVE_PATH,
    SOURCE_NAME,
    TASK_NAME,
    build_inference_command,
    run_smoke_slice,
)

OUTPUT_DIR = REPO_ROOT / "archive" / "runs" / "phase0" / "p0-2-openfold3"


def test_phase0_openfold3_smoke_writes_honest_artifacts() -> None:
    task_payload = run_smoke_slice(OUTPUT_DIR)

    env_check_path = OUTPUT_DIR / "env-check.txt"
    output_summary_path = OUTPUT_DIR / "output-summary.json"
    comparison_path = OUTPUT_DIR / "comparison.json"
    command_log_path = OUTPUT_DIR / "command.log"
    task_path = OUTPUT_DIR / "task.json"
    error_log_path = OUTPUT_DIR / "task-error.log"

    assert env_check_path.exists()
    assert output_summary_path.exists()
    assert comparison_path.exists()
    assert command_log_path.exists()
    assert task_path.exists()

    output_summary = json.loads(output_summary_path.read_text(encoding="utf-8"))
    comparison_payload = json.loads(comparison_path.read_text(encoding="utf-8"))
    task_file_payload = json.loads(task_path.read_text(encoding="utf-8"))
    env_check = env_check_path.read_text(encoding="utf-8")
    command_log = command_log_path.read_text(encoding="utf-8")

    assert task_file_payload == task_payload
    assert output_summary["task"] == TASK_NAME
    assert output_summary["source"] == SOURCE_NAME
    assert comparison_payload["task"] == TASK_NAME
    assert comparison_payload["source"] == SOURCE_NAME
    assert task_payload["status"] in {"pass", "blocked"}
    assert "openfold3_probe.openfold3_spec_found=" in env_check
    assert comparison_payload["comparison_method"] == "contract-level/non-rmsd"
    assert comparison_payload["comparison_result"] == "not_comparable_target_mismatch"
    assert comparison_payload["baseline_available"] is True
    assert output_summary["command_log_path"].endswith("command.log")
    assert task_payload["command_log_path"].endswith("command.log")
    assert (
        "authoritative_predict_command=run_openfold predict --query_json=examples/example_inference_inputs/query_ubiquitin.json"
        in command_log
    )
    assert "planned_inference_command=" in command_log

    if task_payload["status"] == "pass":
        assert output_summary["failure_class"] is None
        assert output_summary["produced_structure_artifacts_present"] is True
        assert output_summary["produced_structure_artifact_paths"]
        assert not error_log_path.exists()
    else:
        assert error_log_path.exists()
        assert output_summary["failure_class"] in {
            "environment-prerequisite",
            "openfold3-runtime",
            "internal-error",
        }
        error_log = error_log_path.read_text(encoding="utf-8")
        assert "error_type=RuntimeError" in error_log
        if output_summary["failure_class"] == "environment-prerequisite":
            assert output_summary["failure_kind"] in {
                "missing_openfold3_package",
                "platform_unsatisfiable_mkl_dependency",
            }


def test_phase0_openfold3_smoke_supports_pass_and_runtime_blocked_paths(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "fake-openfold3"
    package_dir = package_root / "openfold3"
    package_dir.mkdir(parents=True)
    _ = (package_dir / "__init__.py").write_text(
        "__version__ = '0.1'\n", encoding="utf-8"
    )

    example_query_path = package_root / OFFICIAL_QUERY_RELATIVE_PATH
    example_query_path.parent.mkdir(parents=True, exist_ok=True)
    _ = example_query_path.write_text(
        json.dumps(
            {"name": "ubiquitin", "sequences": [{"protein": {"sequence": "MSEQN"}}]}
        )
        + "\n",
        encoding="utf-8",
    )

    weights_dir = package_root / "weights"
    weights_dir.mkdir()
    _ = (weights_dir / "manifest.json").write_text("{}\n", encoding="utf-8")

    fake_paths = {
        "run_openfold": "/fake/bin/run_openfold",
        "setup_openfold": "/fake/bin/setup_openfold",
        "kalign": "/fake/bin/kalign",
        "nvcc": "/fake/bin/nvcc",
        "nvidia-smi": "/fake/bin/nvidia-smi",
    }

    def fake_success_executor(
        command: Sequence[str],
        timeout_seconds: float,
        cwd: Path | None,
    ) -> subprocess.CompletedProcess[str]:
        if (
            command[:2] == [sys.executable, "-c"]
            and "openfold3_spec_found" in command[2]
        ):
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps(
                    {
                        "openfold3_spec_found": True,
                        "openfold3_origin": str(package_dir / "__init__.py"),
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
                        "torch_version": "2.6.0",
                        "cuda_available": True,
                        "cuda_device_count": 1,
                    }
                )
                + "\n",
                stderr="",
            )

        if cwd is None:
            raise AssertionError(
                "OpenFold3 predict command should run in the staged workspace."
            )
        prediction_dir = cwd / "predictions" / "query_ubiquitin"
        prediction_dir.mkdir(parents=True, exist_ok=True)
        _ = (prediction_dir / "prediction.cif").write_text(
            "data_prediction\n", encoding="utf-8"
        )
        return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

    def fake_runtime_blocked_executor(
        command: Sequence[str],
        timeout_seconds: float,
        cwd: Path | None,
    ) -> subprocess.CompletedProcess[str]:
        if command[:2] == [sys.executable, "-c"]:
            return fake_success_executor(command, timeout_seconds, cwd)
        return subprocess.CompletedProcess(
            command,
            1,
            stdout="",
            stderr="weights manifest missing during run_openfold predict",
        )

    pass_output_dir = tmp_path / "pass-run"
    blocked_output_dir = tmp_path / "blocked-run"
    pass_task = run_smoke_slice(
        pass_output_dir,
        command_executor=fake_success_executor,
        which_resolver=lambda name: fake_paths.get(name),
        environment={"OPENFOLD3_WEIGHTS_DIR": str(weights_dir)},
    )
    blocked_task = run_smoke_slice(
        blocked_output_dir,
        command_executor=fake_runtime_blocked_executor,
        which_resolver=lambda name: fake_paths.get(name),
        environment={"OPENFOLD3_WEIGHTS_DIR": str(weights_dir)},
    )

    assert pass_task["status"] == "pass"
    assert blocked_task["status"] == "blocked"

    pass_summary = json.loads(
        (pass_output_dir / "output-summary.json").read_text(encoding="utf-8")
    )
    blocked_summary = json.loads(
        (blocked_output_dir / "output-summary.json").read_text(encoding="utf-8")
    )
    blocked_error_log = (blocked_output_dir / "task-error.log").read_text(
        encoding="utf-8"
    )
    pass_command_log = (pass_output_dir / "command.log").read_text(encoding="utf-8")
    blocked_command_log = (blocked_output_dir / "command.log").read_text(
        encoding="utf-8"
    )

    assert pass_summary["produced_structure_artifacts_present"] is True
    assert pass_summary["produced_structure_artifact_paths"] == [
        "predictions/query_ubiquitin/prediction.cif"
    ]
    assert blocked_summary["failure_class"] == "openfold3-runtime"
    assert blocked_summary["failure_kind"] == "missing_setup_or_weights"
    assert build_inference_command(fake_paths["run_openfold"]) == [
        "/fake/bin/run_openfold",
        "predict",
        "--query_json=examples/example_inference_inputs/query_ubiquitin.json",
    ]
    assert "weights manifest missing" in blocked_error_log
    assert "inference_command_executed=True" in pass_command_log
    assert (
        "executed_inference_command=/fake/bin/run_openfold predict --query_json=examples/example_inference_inputs/query_ubiquitin.json"
        in pass_command_log
    )
    assert "failure_kind=missing_setup_or_weights" in blocked_command_log
