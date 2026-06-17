from __future__ import annotations

import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

TASK_NAME = "P0-2 OpenFold3 fallback runtime verification against AF-DB baseline"
SOURCE_NAME = "openfold3"
AUTHORITATIVE_INSTALL_COMMAND = "pip install openfold3"
AUTHORITATIVE_SETUP_COMMAND = "setup_openfold"
AUTHORITATIVE_PREDICT_COMMAND = "run_openfold predict --query_json=examples/example_inference_inputs/query_ubiquitin.json"
DEFAULT_TIMEOUT_SECONDS = 300.0
OFFICIAL_QUERY_RELATIVE_PATH = (
    Path("examples") / "example_inference_inputs" / "query_ubiquitin.json"
)
AFDB_BASELINE_PATH = (
    Path("archive") / "runs" / "phase0" / "p0-5-afdb" / "normalized_prediction.json"
)
REPO_ROOT = Path(__file__).resolve().parents[3]
WEIGHTS_ENV_VARS = (
    "OPENFOLD3_WEIGHTS_DIR",
    "OPENFOLD_WEIGHTS_DIR",
    "OPENFOLD3_MODEL_DIR",
    "OPENFOLD_MODEL_DIR",
)
WEIGHTS_DIR_NAMES = ("weights", "checkpoints", "params", "model_weights")

CommandExecutor = Callable[
    [Sequence[str], float, Path | None], subprocess.CompletedProcess[str]
]
WhichResolver = Callable[[str], str | None]


def run_smoke_slice(
    output_dir: str | Path,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    command_executor: CommandExecutor | None = None,
    which_resolver: WhichResolver = shutil.which,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    executor = command_executor or execute_command
    env = dict(os.environ if environment is None else environment)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    env_check_path = output_path / "env-check.txt"
    output_summary_path = output_path / "output-summary.json"
    comparison_path = output_path / "comparison.json"
    command_log_path = output_path / "command.log"
    task_path = output_path / "task.json"
    error_log_path = output_path / "task-error.log"
    workspace_dir = output_path / "workspace"

    reset_path(workspace_dir)
    unlink_if_exists(error_log_path)

    started_at = time.perf_counter()
    run_record: dict[str, Any] = {
        "command": None,
        "cwd": str(workspace_dir),
        "duration_seconds": 0.0,
        "exit_code": None,
        "stdout": "",
        "stderr": "",
        "timed_out": False,
    }
    output_files: list[str] = []
    openfold3_probe_script = (
        "import importlib.util, json; "
        "spec = importlib.util.find_spec('openfold3'); "
        "print(json.dumps({'openfold3_spec_found': spec is not None, "
        "'openfold3_origin': None if spec is None else spec.origin}))"
    )
    torch_probe_script = (
        "import importlib.util, json; payload = {'torch_spec_found': False, "
        "'torch_version': None, 'cuda_available': None, 'cuda_device_count': 0}; "
        "spec = importlib.util.find_spec('torch'); "
        "payload['torch_spec_found'] = spec is not None; "
        "payload['torch_version'] = None if spec is None else __import__('torch').__version__; "
        "payload['cuda_available'] = None if spec is None else bool(__import__('torch').cuda.is_available()); "
        "payload['cuda_device_count'] = 0 if spec is None or not payload['cuda_available'] else int(__import__('torch').cuda.device_count()); "
        "print(json.dumps(payload))"
    )
    cli_paths: dict[str, str | None] = {
        "run_openfold": None,
        "setup_openfold": None,
        "kalign": None,
        "nvcc": None,
        "nvidia_smi": None,
    }
    failure_class: str | None = None
    failure_kind: str | None = None
    status = "blocked"

    try:
        openfold3_probe = run_json_probe(
            script=openfold3_probe_script,
            command_executor=executor,
        )
        torch_probe = run_json_probe(
            script=torch_probe_script,
            command_executor=executor,
        )

        cli_paths = {
            "run_openfold": which_resolver("run_openfold"),
            "setup_openfold": which_resolver("setup_openfold"),
            "kalign": which_resolver("kalign"),
            "nvcc": which_resolver("nvcc"),
            "nvidia_smi": which_resolver("nvidia-smi"),
        }
        package_root = detect_package_root(openfold3_probe.get("openfold3_origin"))
        example_query_source = find_official_query_source(package_root)
        weights_probe = detect_weights_probe(package_root=package_root, environment=env)
        prerequisite_failures = collect_prerequisite_failures(
            openfold3_probe=openfold3_probe,
            torch_probe=torch_probe,
            cli_paths=cli_paths,
            example_query_source=example_query_source,
            weights_probe=weights_probe,
        )

        env_payload = {
            "openfold3_probe": openfold3_probe,
            "torch_probe": torch_probe,
            "cli_paths": cli_paths,
            "package_root": None if package_root is None else str(package_root),
            "example_query_source": None
            if example_query_source is None
            else str(example_query_source),
            "weights_probe": weights_probe,
            "detected_failures": prerequisite_failures,
        }
        write_env_check(env_check_path, env_payload)

        error_message: str | None = None

        if prerequisite_failures:
            status = "blocked"
            failure_class = "environment-prerequisite"
            failure_kind = normalize_prerequisite_failure(prerequisite_failures[0])
            error_message = (
                "OpenFold3 local runtime prerequisites were not met: "
                + ", ".join(prerequisite_failures)
                + "."
            )
        else:
            if example_query_source is None or cli_paths["run_openfold"] is None:
                raise RuntimeError(
                    "OpenFold3 probes passed but the official example could not be staged."
                )
            stage_official_query(example_query_source, workspace_dir)
            run_record = run_command(
                command=build_inference_command(cli_paths["run_openfold"]),
                timeout_seconds=timeout_seconds,
                cwd=workspace_dir,
                command_executor=executor,
            )
            output_files = collect_output_files(workspace_dir)
            if run_record["timed_out"]:
                status = "blocked"
                failure_class = "openfold3-runtime"
                failure_kind = "prediction_timeout"
                error_message = f"OpenFold3 prediction exceeded the bounded timeout of {timeout_seconds:.0f}s."
            elif run_record["exit_code"] != 0:
                status = "blocked"
                failure_class = "openfold3-runtime"
                failure_kind = classify_runtime_failure(run_record)
                error_detail = summarize_runtime_error(run_record)
                error_message = (
                    f"OpenFold3 prediction exited with code {run_record['exit_code']}: "
                    f"{error_detail}"
                )
            elif not output_files:
                status = "blocked"
                failure_class = "openfold3-runtime"
                failure_kind = "missing_output_artifacts"
                error_message = "OpenFold3 prediction returned success but did not produce any output artifacts."
            else:
                status = "pass"

        comparison_payload = build_comparison_payload(REPO_ROOT, output_files)
        output_summary_payload = {
            "task": TASK_NAME,
            "source": SOURCE_NAME,
            "status": status,
            "failure_class": failure_class,
            "failure_kind": failure_kind,
            "command_log_path": str(command_log_path),
            "authoritative_predict_command": AUTHORITATIVE_PREDICT_COMMAND,
            "workspace_dir": str(workspace_dir),
            "produced_structure_artifact_paths": output_files,
            "produced_structure_artifacts_present": bool(output_files),
            "run_record": run_record,
        }
        task_payload = {
            "task": TASK_NAME,
            "source": SOURCE_NAME,
            "status": status,
            "failure_class": failure_class,
            "failure_kind": failure_kind,
            "command_log_path": str(command_log_path),
            "env_check_path": str(env_check_path),
            "output_summary_path": str(output_summary_path),
            "comparison_path": str(comparison_path),
            "task_path": str(task_path),
            "error_log_path": str(error_log_path),
            "wall_time_seconds": round(time.perf_counter() - started_at, 6),
        }

        write_command_log(
            path=command_log_path,
            cli_paths=cli_paths,
            openfold3_probe_script=openfold3_probe_script,
            torch_probe_script=torch_probe_script,
            run_record=run_record,
            status=status,
            failure_kind=failure_kind,
        )
        write_json(output_summary_path, output_summary_payload)
        write_json(comparison_path, comparison_payload)
        write_json(task_path, task_payload)
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
        write_command_log(
            path=command_log_path,
            cli_paths=cli_paths,
            openfold3_probe_script=openfold3_probe_script,
            torch_probe_script=torch_probe_script,
            run_record=run_record,
            status="blocked",
            failure_kind=type(exc).__name__,
        )
        write_env_check(env_check_path, {"internal_error": str(exc)})
        write_json(
            output_summary_path,
            {
                "task": TASK_NAME,
                "source": SOURCE_NAME,
                "status": "blocked",
                "failure_class": "internal-error",
                "failure_kind": type(exc).__name__,
                "command_log_path": str(command_log_path),
                "produced_structure_artifact_paths": output_files,
                "produced_structure_artifacts_present": bool(output_files),
                "run_record": run_record,
            },
        )
        write_json(comparison_path, build_comparison_payload(REPO_ROOT, output_files))
        task_payload = {
            "task": TASK_NAME,
            "source": SOURCE_NAME,
            "status": "blocked",
            "failure_class": "internal-error",
            "failure_kind": type(exc).__name__,
            "command_log_path": str(command_log_path),
            "env_check_path": str(env_check_path),
            "output_summary_path": str(output_summary_path),
            "comparison_path": str(comparison_path),
            "task_path": str(task_path),
            "error_log_path": str(error_log_path),
            "wall_time_seconds": round(time.perf_counter() - started_at, 6),
        }
        write_json(task_path, task_payload)
        write_task_error(
            error_path=error_log_path,
            status="blocked",
            failure_class="internal-error",
            failure_kind=type(exc).__name__,
            error_message=str(exc),
        )
        return task_payload


def build_inference_command(run_openfold_path: str) -> list[str]:
    return [
        run_openfold_path,
        "predict",
        "--query_json=examples/example_inference_inputs/query_ubiquitin.json",
    ]


def detect_package_root(openfold3_origin: Any) -> Path | None:
    if not isinstance(openfold3_origin, str) or not openfold3_origin:
        return None
    origin_path = Path(openfold3_origin).resolve()
    candidates = [
        origin_path.parent,
        origin_path.parent.parent,
        origin_path.parent.parent.parent,
    ]
    for candidate in candidates:
        if (candidate / OFFICIAL_QUERY_RELATIVE_PATH).exists():
            return candidate
    return (
        origin_path.parent.parent
        if origin_path.name == "__init__.py"
        else origin_path.parent
    )


def find_official_query_source(package_root: Path | None) -> Path | None:
    if package_root is None:
        return None
    query_path = package_root / OFFICIAL_QUERY_RELATIVE_PATH
    return query_path if query_path.exists() else None


def detect_weights_probe(
    package_root: Path | None,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    detected_locations: list[str] = []
    env_locations: dict[str, str] = {}
    for env_var in WEIGHTS_ENV_VARS:
        value = environment.get(env_var)
        if value:
            env_locations[env_var] = value
            if Path(value).exists():
                detected_locations.append(str(Path(value).resolve()))
    if package_root is not None:
        for dirname in WEIGHTS_DIR_NAMES:
            candidate = package_root / dirname
            if candidate.is_dir() and any(candidate.iterdir()):
                detected_locations.append(str(candidate.resolve()))
    return {
        "weights_detected": bool(detected_locations),
        "environment_variables": env_locations,
        "detected_locations": sorted(set(detected_locations)),
    }


def collect_prerequisite_failures(
    openfold3_probe: Mapping[str, Any],
    torch_probe: Mapping[str, Any],
    cli_paths: Mapping[str, str | None],
    example_query_source: Path | None,
    weights_probe: Mapping[str, Any],
) -> list[str]:
    failures: list[str] = []
    if not openfold3_probe.get("openfold3_spec_found"):
        failures.append(detect_platform_openfold_failure())
    if (
        cli_paths.get("run_openfold") is None
        or cli_paths.get("setup_openfold") is None
        or example_query_source is None
        or not weights_probe.get("weights_detected")
    ):
        failures.append("missing_setup_or_weights")
    if cli_paths.get("kalign") is None:
        failures.append("missing_kalign_binary")
    if not torch_probe.get("torch_spec_found"):
        failures.append("missing_torch_package")
    elif not torch_probe.get("cuda_available"):
        failures.append("cuda_unavailable")
    return failures


def detect_platform_openfold_failure() -> str:
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "platform_unsatisfiable_mkl_dependency"
    return "missing_openfold3_package"


def normalize_prerequisite_failure(failure_kind: str) -> str:
    return failure_kind


def build_comparison_payload(
    repo_root: Path,
    output_files: Sequence[str],
) -> dict[str, Any]:
    baseline_path = repo_root / AFDB_BASELINE_PATH
    baseline_payload = (
        json.loads(baseline_path.read_text(encoding="utf-8"))
        if baseline_path.exists()
        else None
    )
    return {
        "task": TASK_NAME,
        "source": SOURCE_NAME,
        "baseline_reference_path": str(baseline_path),
        "baseline_available": baseline_payload is not None,
        "baseline_summary": baseline_payload,
        "openfold3_example_query": str(OFFICIAL_QUERY_RELATIVE_PATH),
        "openfold3_example_target": "ubiquitin",
        "produced_structure_artifact_paths": list(output_files),
        "comparison_method": "contract-level/non-rmsd",
        "comparable": False,
        "comparison_result": "not_comparable_target_mismatch",
        "rationale": (
            "The official OpenFold3 minimal example targets ubiquitin, while the AF-DB baseline "
            "artifact in this repo is the CXCR4 accession P61073. A direct RMSD claim would be "
            "invalid, so this artifact records only target identity and baseline availability."
        ),
    }


def stage_official_query(example_query_source: Path, workspace_dir: Path) -> Path:
    staged_query_path = workspace_dir / OFFICIAL_QUERY_RELATIVE_PATH
    staged_query_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(example_query_source, staged_query_path)
    return staged_query_path


def collect_output_files(workspace_dir: Path) -> list[str]:
    excluded_path = workspace_dir / OFFICIAL_QUERY_RELATIVE_PATH
    return sorted(
        str(path.relative_to(workspace_dir))
        for path in workspace_dir.rglob("*")
        if path.is_file() and path != excluded_path
    )


def classify_runtime_failure(run_record: Mapping[str, Any]) -> str:
    error_text = (
        f"{run_record.get('stdout', '')}\n{run_record.get('stderr', '')}".lower()
    )
    if "kalign" in error_text:
        return "missing_kalign_binary"
    if any(
        token in error_text for token in ("checkpoint", "weights", "model", "setup")
    ):
        return "missing_setup_or_weights"
    if any(token in error_text for token in ("cuda", "cudnn", "nvidia")):
        return "cuda_toolchain_failure"
    return "openfold3_runtime_error"


def summarize_runtime_error(run_record: Mapping[str, Any]) -> str:
    stderr_text = str(run_record.get("stderr", "")).strip()
    if stderr_text:
        return stderr_text
    stdout_text = str(run_record.get("stdout", "")).strip()
    return (
        stdout_text or "OpenFold3 returned a non-zero exit code without stderr output."
    )


def write_command_log(
    path: Path,
    cli_paths: Mapping[str, str | None],
    openfold3_probe_script: str,
    torch_probe_script: str,
    run_record: Mapping[str, Any],
    status: str,
    failure_kind: str | None,
) -> None:
    planned_inference_command = build_inference_command(
        cli_paths.get("run_openfold") or "run_openfold"
    )
    executed_command = run_record.get("command")
    lines = [
        f"task={TASK_NAME}",
        f"source={SOURCE_NAME}",
        f"authoritative_install_command={AUTHORITATIVE_INSTALL_COMMAND}",
        f"authoritative_setup_command={AUTHORITATIVE_SETUP_COMMAND}",
        f"authoritative_predict_command={AUTHORITATIVE_PREDICT_COMMAND}",
        f"probe_openfold3_package_command={format_command([sys.executable, '-c', openfold3_probe_script])}",
        f"probe_torch_cuda_command={format_command([sys.executable, '-c', torch_probe_script])}",
        f"planned_inference_command={format_command(planned_inference_command)}",
        f"inference_command_executed={executed_command is not None}",
        "executed_inference_command="
        + (
            "skipped"
            if executed_command is None
            else format_command([str(part) for part in executed_command])
        ),
        f"run_openfold_path={cli_paths.get('run_openfold')}",
        f"setup_openfold_path={cli_paths.get('setup_openfold')}",
        f"kalign_path={cli_paths.get('kalign')}",
        f"final_status={status}",
        f"failure_kind={failure_kind}",
    ]
    if executed_command is not None:
        lines.extend(
            [
                f"exit_code={run_record.get('exit_code')}",
                f"timed_out={run_record.get('timed_out')}",
                f"cwd={run_record.get('cwd')}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_env_check(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [f"task={TASK_NAME}", f"source={SOURCE_NAME}"]
    for key, value in flatten_mapping(payload).items():
        lines.append(f"{key}={value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def flatten_mapping(
    payload: Mapping[str, Any],
    prefix: str = "",
) -> dict[str, str]:
    flattened: dict[str, str] = {}
    for key, value in payload.items():
        namespaced_key = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flattened.update(flatten_mapping(value, prefix=f"{namespaced_key}."))
        elif isinstance(value, Path):
            flattened[namespaced_key] = str(value)
        elif isinstance(value, list):
            flattened[namespaced_key] = ",".join(str(item) for item in value) or "none"
        else:
            flattened[namespaced_key] = "none" if value is None else str(value)
    return flattened


def run_json_probe(
    script: str,
    command_executor: CommandExecutor,
) -> dict[str, Any]:
    result = command_executor(
        [sys.executable, "-c", script], DEFAULT_TIMEOUT_SECONDS, None
    )
    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.strip() or result.stdout.strip() or "probe failed"
        )
    return json.loads(result.stdout)


def run_command(
    command: Sequence[str],
    timeout_seconds: float,
    cwd: Path,
    command_executor: CommandExecutor,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    try:
        result = command_executor(command, timeout_seconds, cwd)
        return {
            "command": list(command),
            "cwd": str(cwd),
            "duration_seconds": round(time.perf_counter() - started_at, 6),
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": list(command),
            "cwd": str(cwd),
            "duration_seconds": round(time.perf_counter() - started_at, 6),
            "exit_code": None,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "timed_out": True,
        }


def execute_command(
    command: Sequence[str],
    timeout_seconds: float,
    cwd: Path | None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        cwd=cwd,
    )


def format_command(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def write_task_error(
    error_path: Path,
    status: str,
    failure_class: str | None,
    failure_kind: str | None,
    error_message: str | None,
) -> None:
    error_path.write_text(
        "\n".join(
            [
                f"task={TASK_NAME}",
                f"status={status}",
                f"failure_class={failure_class}",
                f"failure_kind={failure_kind}",
                "error_type=RuntimeError",
                f"error_message={error_message}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def reset_path(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def unlink_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()
