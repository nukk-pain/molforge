from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

TASK_NAME = (
    "P0-3 REINVENT4 pocket-constraint format confirmation and bounded sample run"
)
SOURCE_NAME = "reinvent4"
AUTHORITATIVE_INSTALL_COMMAND = "python install.py cu126"
AUTHORITATIVE_CLI_COMMAND = "reinvent -l sampling.log sampling.toml"
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_DOCUMENTATION_REFERENCES = (
    "REINVENT 4 sampling CLI",
    "molforge REINVENT runtime configuration",
)
REPO_ROOT = Path(__file__).resolve().parents[3]
EXTERNAL_REINVENT_PYTHON = REPO_ROOT / ".uv" / "phase0-reinvent4-mac" / "bin" / "python"
EXTERNAL_REINVENT_CLI = REPO_ROOT / ".uv" / "phase0-reinvent4-mac" / "bin" / "reinvent"
BOOKKEEPING_FILENAMES = {
    "install.log",
    "config.toml",
    "decision.json",
    "command.log",
    "task.json",
    "task-error.log",
    "sampling.log",
}

CommandExecutor = Callable[
    [Sequence[str], float, Path | None], subprocess.CompletedProcess[str]
]
WhichResolver = Callable[[str], str | None]


def run_smoke_slice(
    output_dir: str | Path,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    command_executor: CommandExecutor | None = None,
    which_resolver: WhichResolver = shutil.which,
    official_config_probe: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    executor = command_executor or execute_command
    use_external_runtime = command_executor is None
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    install_log_path = output_path / "install.log"
    config_path = output_path / "config.toml"
    decision_path = output_path / "decision.json"
    command_log_path = output_path / "command.log"
    task_path = output_path / "task.json"
    error_log_path = output_path / "task-error.log"
    sampling_log_path = output_path / "sampling.log"

    unlink_if_exists(error_log_path)
    unlink_if_exists(sampling_log_path)

    started_at = time.perf_counter()
    runtime_record: dict[str, Any] | None = None
    output_files: list[str] = []
    package_probe_script = (
        "import importlib.util, json; "
        "spec = importlib.util.find_spec('reinvent'); "
        "print(json.dumps({'reinvent_spec_found': spec is not None, "
        "'reinvent_origin': None if spec is None else spec.origin}))"
    )
    status = "blocked"
    failure_class: str | None = None
    failure_kind: str | None = None

    try:
        package_probe = run_json_probe(
            script=package_probe_script,
            command_executor=executor,
            python_executable=resolve_reinvent_python(use_external_runtime),
        )
        cli_path = resolve_reinvent_cli(which_resolver, use_external_runtime)
        config_probe = resolve_official_config_probe(official_config_probe)
        config_text = build_config_toml(config_probe)
        _ = config_path.write_text(config_text, encoding="utf-8")

        blocking_reasons = collect_blocking_reasons(
            package_probe=package_probe,
            cli_path=cli_path,
            config_probe=config_probe,
        )

        sampling_command: list[str] | None = None
        error_message: str | None = None

        if config_probe["status"] != "confirmed":
            status = "blocked"
            failure_class = "documentation-ambiguity"
            failure_kind = "unresolved_pocket_constraint_format"
            error_message = str(config_probe["summary"])
        elif not package_probe["reinvent_spec_found"]:
            status = "blocked"
            failure_class = "environment-prerequisite"
            failure_kind = "missing_reinvent_package"
            error_message = "REINVENT4 package import probe failed before any bounded sampling run could start."
        elif cli_path is None:
            status = "blocked"
            failure_class = "environment-prerequisite"
            failure_kind = "missing_reinvent_cli"
            error_message = (
                "REINVENT4 CLI entrypoint `reinvent` was not available in PATH."
            )
        else:
            sampling_command = build_sampling_command(
                cli_path=cli_path, config_path=config_path
            )
            runtime_record = run_command(
                command=sampling_command,
                timeout_seconds=timeout_seconds,
                cwd=output_path,
                command_executor=executor,
            )
            output_files = collect_output_files(output_path)

            if runtime_record["timed_out"]:
                status = "blocked"
                failure_class = "reinvent4-runtime"
                failure_kind = "sampling_timeout"
                error_message = f"REINVENT4 bounded sample run exceeded the timeout of {timeout_seconds:.0f}s."
            elif runtime_record["exit_code"] != 0:
                status = "blocked"
                failure_class = "reinvent4-runtime"
                failure_kind = classify_runtime_failure(runtime_record)
                error_message = (
                    f"REINVENT4 sampling exited with code {runtime_record['exit_code']}: "
                    f"{summarize_runtime_error(runtime_record)}"
                )
            elif not output_files:
                status = "blocked"
                failure_class = "reinvent4-runtime"
                failure_kind = "missing_output_artifacts"
                error_message = (
                    "REINVENT4 sampling returned success but did not produce any non-bookkeeping "
                    "artifacts in the bounded workspace."
                )
            else:
                status = "pass"

        output_files = output_files or collect_output_files(output_path)
        decision_payload = {
            "task": TASK_NAME,
            "source": SOURCE_NAME,
            "status": status,
            "failure_class": failure_class,
            "failure_kind": failure_kind,
            "blocking_reasons": blocking_reasons,
            "command_log_path": str(command_log_path),
            "official_config_status": config_probe["status"],
            "official_config_summary": config_probe["summary"],
            "official_config_references": list(config_probe["references"]),
            "authoritative_install_command": AUTHORITATIVE_INSTALL_COMMAND,
            "authoritative_cli_command": AUTHORITATIVE_CLI_COMMAND,
            "sampling_command": None
            if sampling_command is None
            else format_command(sampling_command),
            "config_path": str(config_path),
            "sampling_log_path": str(sampling_log_path),
            "produced_output_files": output_files,
        }
        task_payload = {
            "task": TASK_NAME,
            "source": SOURCE_NAME,
            "status": status,
            "failure_class": failure_class,
            "failure_kind": failure_kind,
            "command_log_path": str(command_log_path),
            "install_log_path": str(install_log_path),
            "config_path": str(config_path),
            "decision_path": str(decision_path),
            "task_path": str(task_path),
            "error_log_path": str(error_log_path),
            "sampling_log_path": str(sampling_log_path),
            "wall_time_seconds": round(time.perf_counter() - started_at, 6),
        }

        write_command_log(
            path=command_log_path,
            package_probe_script=package_probe_script,
            cli_path=cli_path,
            runtime_record=runtime_record,
            config_probe=config_probe,
            status=status,
            failure_kind=failure_kind,
            config_path=config_path,
        )
        write_install_log(
            path=install_log_path,
            package_probe=package_probe,
            cli_path=cli_path,
            config_probe=config_probe,
            runtime_record=runtime_record,
        )
        write_json(decision_path, decision_payload)
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
        config_probe = resolve_official_config_probe(official_config_probe)
        _ = config_path.write_text(build_config_toml(config_probe), encoding="utf-8")
        write_command_log(
            path=command_log_path,
            package_probe_script=package_probe_script,
            cli_path=None,
            runtime_record=runtime_record,
            config_probe=config_probe,
            status="blocked",
            failure_kind=type(exc).__name__,
            config_path=config_path,
        )
        write_install_log(
            path=install_log_path,
            package_probe={"reinvent_spec_found": None, "reinvent_origin": None},
            cli_path=None,
            config_probe=config_probe,
            runtime_record=runtime_record,
            internal_error=str(exc),
        )
        write_json(
            decision_path,
            {
                "task": TASK_NAME,
                "source": SOURCE_NAME,
                "status": "blocked",
                "failure_class": "internal-error",
                "failure_kind": type(exc).__name__,
                "blocking_reasons": [type(exc).__name__],
                "command_log_path": str(command_log_path),
                "official_config_status": config_probe["status"],
                "official_config_summary": config_probe["summary"],
                "official_config_references": list(config_probe["references"]),
                "authoritative_install_command": AUTHORITATIVE_INSTALL_COMMAND,
                "authoritative_cli_command": AUTHORITATIVE_CLI_COMMAND,
                "sampling_command": None,
                "config_path": str(config_path),
                "sampling_log_path": str(sampling_log_path),
                "produced_output_files": collect_output_files(output_path),
            },
        )
        task_payload = {
            "task": TASK_NAME,
            "source": SOURCE_NAME,
            "status": "blocked",
            "failure_class": "internal-error",
            "failure_kind": type(exc).__name__,
            "command_log_path": str(command_log_path),
            "install_log_path": str(install_log_path),
            "config_path": str(config_path),
            "decision_path": str(decision_path),
            "task_path": str(task_path),
            "error_log_path": str(error_log_path),
            "sampling_log_path": str(sampling_log_path),
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


def resolve_official_config_probe(
    official_config_probe: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if official_config_probe is not None:
        references = (
            official_config_probe.get("references") or DEFAULT_DOCUMENTATION_REFERENCES
        )
        return {
            "status": official_config_probe.get("status", "ambiguous"),
            "summary": official_config_probe.get(
                "summary",
                "Official pocket-aware REINVENT4 config support was not confirmed.",
            ),
            "references": tuple(str(item) for item in references),
            "config_text": official_config_probe.get("config_text"),
        }

    return {
        "status": "ambiguous",
        "summary": (
            "Wave 2 authoritative guidance confirms the REINVENT4 install path and CLI entrypoint, "
            "but the exact pocket-aware config shape remains unresolved for this repository's intended "
            "use. Current local references mention `3DChem` or `Mol2Mol` without a trustworthy runnable "
            "schema, so Phase 1 generative work must stay blocked rather than inventing a local format."
        ),
        "references": DEFAULT_DOCUMENTATION_REFERENCES,
        "config_text": None,
    }


def build_config_toml(config_probe: Mapping[str, Any]) -> str:
    config_text = config_probe.get("config_text")
    if isinstance(config_text, str) and config_text.strip():
        return config_text if config_text.endswith("\n") else config_text + "\n"

    lines = [
        "# REINVENT4 pocket-aware config probe artifact",
        "# No trustworthy runnable pocket-constraint TOML was confirmed for this repository.",
        f'# official_config_status = "{config_probe["status"]}"',
        '# summary = "' + escape_comment_text(str(config_probe["summary"])) + '"',
    ]
    for reference in config_probe["references"]:
        lines.append(f'# reference = "{escape_comment_text(str(reference))}"')
    return "\n".join(lines) + "\n"


def collect_blocking_reasons(
    package_probe: Mapping[str, Any],
    cli_path: str | None,
    config_probe: Mapping[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if not package_probe.get("reinvent_spec_found"):
        reasons.append("missing_reinvent_package")
    if cli_path is None:
        reasons.append("missing_reinvent_cli")
    if config_probe.get("status") != "confirmed":
        reasons.append("unresolved_pocket_constraint_format")
    return reasons


def build_sampling_command(cli_path: str, config_path: Path) -> list[str]:
    return [cli_path, "-l", "sampling.log", config_path.name]


def run_json_probe(
    script: str,
    command_executor: CommandExecutor,
    python_executable: str,
) -> dict[str, Any]:
    completed = command_executor([python_executable, "-c", script], 30.0, None)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "json probe failed")
    return json.loads(completed.stdout.strip())


def run_command(
    command: Sequence[str],
    timeout_seconds: float,
    cwd: Path,
    command_executor: CommandExecutor,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    try:
        completed = command_executor(command, timeout_seconds, cwd)
        return {
            "command": list(command),
            "command_text": format_command(command),
            "cwd": str(cwd),
            "duration_seconds": round(time.perf_counter() - started_at, 6),
            "exit_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return {
            "command": list(command),
            "command_text": format_command(command),
            "cwd": str(cwd),
            "duration_seconds": round(time.perf_counter() - started_at, 6),
            "exit_code": None,
            "stdout": stdout,
            "stderr": stderr,
            "timed_out": True,
        }


def classify_runtime_failure(runtime_record: Mapping[str, Any]) -> str:
    stderr = str(runtime_record.get("stderr") or "").lower()
    stdout = str(runtime_record.get("stdout") or "").lower()
    combined = f"{stdout}\n{stderr}"
    if "pocket" in combined and "constraint" in combined:
        return "pocket_constraint_runtime_failure"
    if "config" in combined or "toml" in combined:
        return "config_validation_failure"
    return "sampling_runtime_failure"


def summarize_runtime_error(runtime_record: Mapping[str, Any]) -> str:
    stderr = str(runtime_record.get("stderr") or "").strip()
    stdout = str(runtime_record.get("stdout") or "").strip()
    if stderr:
        return stderr.splitlines()[0]
    if stdout:
        return stdout.splitlines()[0]
    return "no stderr/stdout detail captured"


def collect_output_files(output_dir: Path) -> list[str]:
    output_files: list[str] = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name in BOOKKEEPING_FILENAMES:
            continue
        output_files.append(path.relative_to(output_dir).as_posix())
    return output_files


def write_command_log(
    path: Path,
    package_probe_script: str,
    cli_path: str | None,
    runtime_record: Mapping[str, Any] | None,
    config_probe: Mapping[str, Any],
    status: str,
    failure_kind: str | None,
    config_path: Path,
) -> None:
    runtime_python = resolve_reinvent_python(False)
    planned_sampling_command = build_sampling_command(
        cli_path or "reinvent",
        config_path,
    )
    lines = [
        f"task={TASK_NAME}",
        f"source={SOURCE_NAME}",
        f"authoritative_install_command={AUTHORITATIVE_INSTALL_COMMAND}",
        f"authoritative_cli_command={AUTHORITATIVE_CLI_COMMAND}",
        f"probe_reinvent_package_command={format_command([runtime_python, '-c', package_probe_script])}",
        f"planned_sampling_command={format_command(planned_sampling_command)}",
        f"sampling_command_executed={runtime_record is not None}",
        "executed_sampling_command="
        + (
            "skipped"
            if runtime_record is None
            else str(runtime_record.get("command_text"))
        ),
        f"config_path={config_path}",
        f"official_config_status={config_probe['status']}",
        f"final_status={status}",
        f"failure_kind={failure_kind}",
    ]
    if runtime_record is not None:
        lines.extend(
            [
                f"exit_code={runtime_record.get('exit_code')}",
                f"timed_out={runtime_record.get('timed_out')}",
                f"cwd={runtime_record.get('cwd')}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_install_log(
    path: Path,
    package_probe: Mapping[str, Any],
    cli_path: str | None,
    config_probe: Mapping[str, Any],
    runtime_record: Mapping[str, Any] | None,
    internal_error: str | None = None,
) -> None:
    lines = [
        f"task={TASK_NAME}",
        f"source={SOURCE_NAME}",
        f"authoritative_install_command={AUTHORITATIVE_INSTALL_COMMAND}",
        f"authoritative_cli_command={AUTHORITATIVE_CLI_COMMAND}",
        f"reinvent_spec_found={package_probe.get('reinvent_spec_found')}",
        f"reinvent_origin={package_probe.get('reinvent_origin')}",
        f"reinvent_cli_path={cli_path}",
        f"official_config_status={config_probe['status']}",
        f"official_config_summary={config_probe['summary']}",
    ]
    for reference in config_probe["references"]:
        lines.append(f"official_config_reference={reference}")
    if runtime_record is None:
        lines.append("sampling_command=null")
    else:
        lines.extend(
            [
                f"sampling_command={runtime_record['command_text']}",
                f"sampling_exit_code={runtime_record['exit_code']}",
                f"sampling_timed_out={runtime_record['timed_out']}",
            ]
        )
    if internal_error is not None:
        lines.append(f"internal_error={internal_error}")
    _ = path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_task_error(
    error_path: Path,
    status: str,
    failure_class: str | None,
    failure_kind: str | None,
    error_message: str | None,
) -> None:
    lines = [
        f"task={TASK_NAME}",
        f"status={status}",
        f"failure_class={failure_class}",
        f"failure_kind={failure_kind}",
        "error_type=RuntimeError",
        f"error_message={error_message}",
    ]
    _ = error_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _ = path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def execute_command(
    command: Sequence[str],
    timeout_seconds: float,
    cwd: Path | None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=None if cwd is None else str(cwd),
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )


def resolve_reinvent_python(use_external_runtime: bool) -> str:
    if use_external_runtime and EXTERNAL_REINVENT_PYTHON.exists():
        return str(EXTERNAL_REINVENT_PYTHON)
    return sys.executable


def resolve_reinvent_cli(
    which_resolver: WhichResolver, use_external_runtime: bool
) -> str | None:
    if use_external_runtime and EXTERNAL_REINVENT_CLI.exists():
        return str(EXTERNAL_REINVENT_CLI)
    return which_resolver("reinvent")


def unlink_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def format_command(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def escape_comment_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', "'")
