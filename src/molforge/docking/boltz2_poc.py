from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TASK_NAME = "P0-1 Boltz-2 local runtime feasibility and fallback decision logging"
SOURCE_NAME = "boltz2"
AUTHORITATIVE_INSTALL_COMMAND = "pip install boltz[cuda] -U"
DEFAULT_TIMEOUT_SECONDS = 180.0
SMOKE_INPUT_FILENAME = "minimal-affinity-input.yaml"
PREDICTION_OUTPUT_DIRNAME = "prediction-output"
SMOKE_PROTEIN_SEQUENCE = "MKTAYIAKQRQISFVKSHFSRQDILD"
SMOKE_LIGAND_SMILES = "CCO"
REPO_ROOT = Path(__file__).resolve().parents[3]
EXTERNAL_BOLTZ_PYTHON = REPO_ROOT / ".uv" / "phase0-boltz-311" / "bin" / "python"
EXTERNAL_BOLTZ_CLI = REPO_ROOT / ".uv" / "phase0-boltz-311" / "bin" / "boltz"

CommandExecutor = Callable[..., subprocess.CompletedProcess[str]]
WhichResolver = Callable[[str], str | None]


@dataclass(frozen=True, slots=True)
class CommandRecord:
    step: str
    command: list[str]
    duration_seconds: float
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "command": self.command,
            "command_text": format_command(self.command),
            "duration_seconds": round(self.duration_seconds, 6),
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "timed_out": self.timed_out,
            "note": self.note,
        }


def run_smoke_slice(
    output_dir: str | Path,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    command_executor: CommandExecutor | None = None,
    which_resolver: WhichResolver = shutil.which,
) -> dict[str, Any]:
    executor = command_executor or execute_command
    use_external_runtime = command_executor is None
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    command_log_path = output_path / "command.log"
    runtime_path = output_path / "runtime.json"
    decision_path = output_path / "decision.json"
    task_path = output_path / "task.json"
    error_log_path = output_path / "task-error.log"
    input_path = output_path / SMOKE_INPUT_FILENAME
    prediction_output_dir = output_path / PREDICTION_OUTPUT_DIRNAME

    reset_path(prediction_output_dir)
    unlink_if_exists(error_log_path)
    unlink_if_exists(input_path)

    started_at = time.perf_counter()
    command_records: list[CommandRecord] = []
    probes: dict[str, Any] = {}
    python_executable = resolve_boltz_python(use_external_runtime)
    planned_inference_command = build_inference_command(
        boltz_cli_path=resolve_boltz_cli(which_resolver, use_external_runtime)
        or "boltz",
        input_path=input_path,
        output_dir=prediction_output_dir,
    )
    inference_command: list[str] | None = None
    inference_record: CommandRecord | None = None
    output_files: list[str] = []

    try:
        boltz_probe, boltz_record = run_json_probe(
            step="probe-boltz-package",
            script=(
                "import importlib.util, json; "
                "spec = importlib.util.find_spec('boltz'); "
                "print(json.dumps({'boltz_spec_found': spec is not None, "
                "'boltz_origin': None if spec is None else spec.origin}))"
            ),
            command_executor=executor,
            python_executable=python_executable,
        )
        command_records.append(boltz_record)
        probes["boltz_package"] = boltz_probe

        torch_probe, torch_record = run_json_probe(
            step="probe-torch-cuda",
            script=(
                "import importlib.util, json; "
                "payload = {'torch_spec_found': False, 'torch_version': None, "
                "'cuda_available': None, 'cuda_device_count': 0, "
                "'cuda_device_name': None, 'cuda_total_memory_bytes': None}; "
                "spec = importlib.util.find_spec('torch'); "
                "payload['torch_spec_found'] = spec is not None; "
                "\nif spec is not None:\n"
                " import torch\n"
                " payload['torch_version'] = getattr(torch, '__version__', None)\n"
                " payload['cuda_available'] = bool(torch.cuda.is_available())\n"
                " if payload['cuda_available']:\n"
                "  payload['cuda_device_count'] = int(torch.cuda.device_count())\n"
                "  props = torch.cuda.get_device_properties(0)\n"
                "  payload['cuda_device_name'] = props.name\n"
                "  payload['cuda_total_memory_bytes'] = int(props.total_memory)\n"
                "print(json.dumps(payload))"
            ),
            command_executor=executor,
            python_executable=python_executable,
        )
        command_records.append(torch_record)
        probes["torch_cuda"] = torch_probe

        boltz_cli_path = resolve_boltz_cli(which_resolver, use_external_runtime)
        probes["boltz_cli"] = {"cli_path": boltz_cli_path}

        environment_failures = collect_environment_failures(
            boltz_probe=boltz_probe,
            torch_probe=torch_probe,
            boltz_cli_path=boltz_cli_path,
        )

        failure_class: str | None = None
        failure_kind: str | None = None
        error_message: str | None = None
        rationale: str
        status: str

        if environment_failures:
            status = "fallback-approved"
            failure_class = "environment-baseline"
            failure_kind = environment_failures[0]
            error_message = build_environment_error_message(environment_failures)
            rationale = build_environment_rationale(environment_failures)
        else:
            if boltz_cli_path is None:
                raise RuntimeError(
                    "Boltz CLI path was unexpectedly unavailable after environment probes passed."
                )
            _ = input_path.write_text(build_smoke_input_yaml(), encoding="utf-8")
            inference_command = build_inference_command(
                boltz_cli_path=boltz_cli_path,
                input_path=input_path,
                output_dir=prediction_output_dir,
            )
            inference_record = run_command(
                step="boltz-predict",
                command=inference_command,
                command_executor=executor,
                timeout_seconds=timeout_seconds,
            )
            command_records.append(inference_record)
            output_files = collect_output_files(prediction_output_dir)

            if inference_record.timed_out:
                status = "blocked"
                failure_class = "boltz-runtime"
                failure_kind = "prediction_timeout"
                error_message = f"Boltz prediction exceeded the bounded timeout of {timeout_seconds:.0f}s."
                rationale = (
                    "Boltz prerequisites were present, but the bounded local inference did not "
                    "complete in time, so the result remains blocked pending a faster or better "
                    "provisioned runtime."
                )
            elif inference_record.exit_code != 0:
                status = "blocked"
                failure_class = "boltz-runtime"
                failure_kind = classify_inference_failure(inference_record)
                error_message = (
                    f"Boltz prediction exited with code {inference_record.exit_code}."
                )
                rationale = (
                    "Boltz reached the bounded inference step, but the runtime failed after the "
                    "environment probes passed, so the result is recorded as a Boltz-specific blocker."
                )
            elif not output_files:
                status = "blocked"
                failure_class = "boltz-runtime"
                failure_kind = "missing_output_artifacts"
                error_message = (
                    "Boltz prediction returned success but did not produce any output files in the "
                    "prediction directory."
                )
                rationale = (
                    "Boltz returned without a hard error, but the expected prediction artifacts were "
                    "absent, so the PoC remains blocked rather than falsely marked as pass."
                )
            else:
                status = "pass"
                rationale = (
                    "Boltz completed one bounded protein-ligand affinity run and produced output "
                    "artifacts within the local canary harness."
                )

        wall_time_seconds = round(time.perf_counter() - started_at, 6)
        output_files = output_files or collect_output_files(prediction_output_dir)

        runtime_payload = {
            "task": TASK_NAME,
            "source": SOURCE_NAME,
            "status": status,
            "failure_class": failure_class,
            "failure_kind": failure_kind,
            "wall_time_seconds": wall_time_seconds,
            "vram_observation": build_vram_observation(
                torch_probe=probes["torch_cuda"]
            ),
            "input_path": str(input_path) if input_path.exists() else None,
            "prediction_output_dir": str(prediction_output_dir),
            "output_files": output_files,
            "output_files_present": bool(output_files),
            "authoritative_install_command": AUTHORITATIVE_INSTALL_COMMAND,
            "install_command_executed": False,
            "inference_command": None
            if inference_command is None
            else format_command(inference_command),
            "probes": probes,
            "commands": [record.to_dict() for record in command_records],
        }
        decision_payload = {
            "task": TASK_NAME,
            "source": SOURCE_NAME,
            "status": status,
            "failure_class": failure_class,
            "failure_kind": failure_kind,
            "rationale": rationale,
            "fallback_candidate": "DiffDock-L"
            if status == "fallback-approved"
            else None,
            "authoritative_install_command": AUTHORITATIVE_INSTALL_COMMAND,
            "inference_command": None
            if inference_command is None
            else format_command(inference_command),
        }
        task_payload = {
            "task": TASK_NAME,
            "source": SOURCE_NAME,
            "status": status,
            "failure_class": failure_class,
            "failure_kind": failure_kind,
            "command_log_path": str(command_log_path),
            "runtime_path": str(runtime_path),
            "decision_path": str(decision_path),
            "task_path": str(task_path),
            "error_log_path": str(error_log_path),
        }

        write_command_log(
            path=command_log_path,
            command_records=command_records,
            planned_inference_command=planned_inference_command,
            inference_command=inference_command,
        )
        _ = runtime_path.write_text(
            json.dumps(runtime_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _ = decision_path.write_text(
            json.dumps(decision_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _ = task_path.write_text(
            json.dumps(task_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        if status == "pass":
            unlink_if_exists(error_log_path)
        else:
            write_task_error(
                error_path=error_log_path,
                status=status,
                failure_class=failure_class,
                failure_kind=failure_kind,
                error_message=error_message,
            )

        return task_payload
    except Exception as exc:
        wall_time_seconds = round(time.perf_counter() - started_at, 6)
        write_command_log(
            path=command_log_path,
            command_records=command_records,
            planned_inference_command=planned_inference_command,
            inference_command=inference_command,
        )
        output_files = collect_output_files(prediction_output_dir)
        _ = runtime_path.write_text(
            json.dumps(
                {
                    "task": TASK_NAME,
                    "source": SOURCE_NAME,
                    "status": "blocked",
                    "failure_class": "internal-error",
                    "failure_kind": type(exc).__name__,
                    "wall_time_seconds": wall_time_seconds,
                    "vram_observation": None,
                    "prediction_output_dir": str(prediction_output_dir),
                    "output_files": output_files,
                    "output_files_present": bool(output_files),
                    "authoritative_install_command": AUTHORITATIVE_INSTALL_COMMAND,
                    "install_command_executed": False,
                    "inference_command": None
                    if inference_command is None
                    else format_command(inference_command),
                    "probes": probes,
                    "commands": [record.to_dict() for record in command_records],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        _ = decision_path.write_text(
            json.dumps(
                {
                    "task": TASK_NAME,
                    "source": SOURCE_NAME,
                    "status": "blocked",
                    "failure_class": "internal-error",
                    "failure_kind": type(exc).__name__,
                    "rationale": "The harness itself failed before reaching a trustworthy Boltz result.",
                    "fallback_candidate": None,
                    "authoritative_install_command": AUTHORITATIVE_INSTALL_COMMAND,
                    "inference_command": None
                    if inference_command is None
                    else format_command(inference_command),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        task_payload = {
            "task": TASK_NAME,
            "source": SOURCE_NAME,
            "status": "blocked",
            "failure_class": "internal-error",
            "failure_kind": type(exc).__name__,
            "command_log_path": str(command_log_path),
            "runtime_path": str(runtime_path),
            "decision_path": str(decision_path),
            "task_path": str(task_path),
            "error_log_path": str(error_log_path),
        }
        _ = task_path.write_text(
            json.dumps(task_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_task_error(
            error_path=error_log_path,
            status="blocked",
            failure_class="internal-error",
            failure_kind=type(exc).__name__,
            error_message=str(exc),
        )
        return task_payload


def build_smoke_input_yaml() -> str:
    return (
        "version: 1\n"
        "sequences:\n"
        "  - protein:\n"
        "      id: A\n"
        f"      sequence: {SMOKE_PROTEIN_SEQUENCE}\n"
        "  - ligand:\n"
        "      id: B\n"
        f"      smiles: '{SMOKE_LIGAND_SMILES}'\n"
        "properties:\n"
        "  - affinity:\n"
        "      binder: B\n"
    )


def build_inference_command(
    boltz_cli_path: str,
    input_path: Path,
    output_dir: Path,
) -> list[str]:
    return [
        boltz_cli_path,
        "predict",
        str(input_path),
        "--use_msa_server",
        "--out_dir",
        str(output_dir),
        "--override",
        "--recycling_steps",
        "1",
        "--sampling_steps",
        "1",
        "--diffusion_samples",
        "1",
        "--sampling_steps_affinity",
        "1",
        "--diffusion_samples_affinity",
        "1",
        "--max_parallel_samples",
        "1",
        "--num_workers",
        "0",
        "--preprocessing-threads",
        "1",
    ]


def collect_environment_failures(
    boltz_probe: dict[str, Any],
    torch_probe: dict[str, Any],
    boltz_cli_path: str | None,
) -> list[str]:
    failures: list[str] = []
    if not boltz_probe.get("boltz_spec_found"):
        failures.append("missing_boltz_package")
    if not torch_probe.get("torch_spec_found"):
        failures.append("missing_torch_package")
    if boltz_probe.get("boltz_spec_found") and not boltz_cli_path:
        failures.append("boltz_cli_unavailable")
    return failures


def build_environment_error_message(environment_failures: Sequence[str]) -> str:
    joined_failures = ", ".join(environment_failures)
    return f"Boltz local runtime prerequisites were not met: {joined_failures}."


def build_environment_rationale(environment_failures: Sequence[str]) -> str:
    joined_failures = ", ".join(environment_failures)
    return (
        "The Wave 2 Boltz canary failed on shared local runtime prerequisites "
        f"({joined_failures}), so a bounded Boltz inference could not start. This is enough "
        "evidence to approve the planned DiffDock-L fallback without broadening the scope into "
        "installation or CUDA remediation work."
    )


def build_vram_observation(torch_probe: dict[str, Any]) -> dict[str, Any] | None:
    if not torch_probe.get("torch_spec_found"):
        return None
    return {
        "torch_version": torch_probe.get("torch_version"),
        "cuda_available": torch_probe.get("cuda_available"),
        "cuda_device_count": torch_probe.get("cuda_device_count"),
        "cuda_device_name": torch_probe.get("cuda_device_name"),
        "cuda_total_memory_bytes": torch_probe.get("cuda_total_memory_bytes"),
    }


def classify_inference_failure(record: CommandRecord) -> str:
    failure_text = (record.stderr + "\n" + record.stdout).lower()
    if "linalg.svd" in failure_text or "algorithm failed to converge" in failure_text:
        return "mps_linalg_runtime_failure"
    if "mps" in failure_text and "not supported" in failure_text:
        return "mps_runtime_failure"
    if "checkpoint" in failure_text or "weight" in failure_text:
        return "missing_weights"
    if "msa" in failure_text or "colabfold" in failure_text:
        return "msa_server_failure"
    if "cuda" in failure_text or "cudnn" in failure_text:
        return "cuda_runtime_failure"
    if (
        "yaml" in failure_text
        or "validation" in failure_text
        or "schema" in failure_text
    ):
        return "input_validation_failure"
    return "boltz_runtime_failure"


def run_json_probe(
    step: str,
    script: str,
    command_executor: CommandExecutor,
    python_executable: str,
) -> tuple[dict[str, Any], CommandRecord]:
    record = run_command(
        step=step,
        command=build_python_probe_command(script, python_executable),
        command_executor=command_executor,
        timeout_seconds=30.0,
    )
    if record.timed_out or record.exit_code != 0:
        raise RuntimeError(f"{step} did not complete successfully.")
    try:
        payload = json.loads(record.stdout.strip())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{step} returned invalid JSON probe output.") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{step} returned a non-object JSON payload.")
    return payload, record


def build_python_probe_command(script: str, python_executable: str) -> list[str]:
    return [python_executable, "-c", script]


def resolve_boltz_python(use_external_runtime: bool) -> str:
    if use_external_runtime and EXTERNAL_BOLTZ_PYTHON.exists():
        return str(EXTERNAL_BOLTZ_PYTHON)
    return sys.executable


def resolve_boltz_cli(
    which_resolver: WhichResolver, use_external_runtime: bool
) -> str | None:
    if use_external_runtime and EXTERNAL_BOLTZ_CLI.exists():
        return str(EXTERNAL_BOLTZ_CLI)
    return which_resolver("boltz")


def run_command(
    step: str,
    command: Sequence[str],
    command_executor: CommandExecutor,
    timeout_seconds: float,
) -> CommandRecord:
    started_at = time.perf_counter()
    try:
        completed_process = command_executor(
            list(command),
            timeout_seconds=timeout_seconds,
            cwd=None,
        )
        return CommandRecord(
            step=step,
            command=list(command),
            duration_seconds=time.perf_counter() - started_at,
            exit_code=completed_process.returncode,
            stdout=completed_process.stdout,
            stderr=completed_process.stderr,
            timed_out=False,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandRecord(
            step=step,
            command=list(command),
            duration_seconds=time.perf_counter() - started_at,
            exit_code=None,
            stdout=coerce_timeout_output(exc.stdout),
            stderr=coerce_timeout_output(exc.stderr),
            timed_out=True,
        )


def execute_command(
    command: Sequence[str],
    timeout_seconds: float,
    cwd: Path | None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_seconds,
        cwd=cwd,
    )


def collect_output_files(output_dir: Path) -> list[str]:
    if not output_dir.exists():
        return []
    return sorted(
        str(path.relative_to(output_dir))
        for path in output_dir.rglob("*")
        if path.is_file()
    )


def write_command_log(
    path: Path,
    command_records: Sequence[CommandRecord],
    planned_inference_command: Sequence[str],
    inference_command: Sequence[str] | None,
) -> None:
    lines = [
        f"authoritative_install_command={AUTHORITATIVE_INSTALL_COMMAND}",
        "install_command_executed=false",
        f"planned_inference_command={format_command(planned_inference_command)}",
        f"inference_command_executed={'true' if inference_command is not None else 'false'}",
        (
            "executed_inference_command=skipped"
            if inference_command is None
            else f"executed_inference_command={format_command(inference_command)}"
        ),
        "",
    ]
    for record in command_records:
        lines.extend(
            [
                f"step={record.step}",
                f"command={format_command(record.command)}",
                f"duration_seconds={record.duration_seconds:.6f}",
                f"exit_code={record.exit_code}",
                f"timed_out={str(record.timed_out).lower()}",
                f"stdout={record.stdout.strip()}",
                f"stderr={record.stderr.strip()}",
                "",
            ]
        )
    _ = path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_task_error(
    error_path: Path,
    status: str,
    failure_class: str | None,
    failure_kind: str | None,
    error_message: str | None,
) -> None:
    _ = error_path.write_text(
        "\n".join(
            [
                f"task={TASK_NAME}",
                f"status={status}",
                f"failure_class={failure_class}",
                f"failure_kind={failure_kind}",
                f"error_type=RuntimeError",
                f"error_message={error_message}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def format_command(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def reset_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def unlink_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def coerce_timeout_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
