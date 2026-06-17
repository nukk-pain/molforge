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

from molforge.generative.reinvent4_poc import (  # noqa: E402
    AUTHORITATIVE_CLI_COMMAND,
    AUTHORITATIVE_INSTALL_COMMAND,
    SOURCE_NAME,
    TASK_NAME,
    build_sampling_command,
    run_smoke_slice,
)

OUTPUT_DIR = REPO_ROOT / "archive" / "runs" / "phase0" / "p0-3-reinvent4"


def test_phase0_reinvent4_smoke_writes_honest_artifacts() -> None:
    task_payload = run_smoke_slice(OUTPUT_DIR)

    install_log_path = OUTPUT_DIR / "install.log"
    config_path = OUTPUT_DIR / "config.toml"
    decision_path = OUTPUT_DIR / "decision.json"
    command_log_path = OUTPUT_DIR / "command.log"
    task_path = OUTPUT_DIR / "task.json"
    error_log_path = OUTPUT_DIR / "task-error.log"

    assert install_log_path.exists()
    assert config_path.exists()
    assert decision_path.exists()
    assert command_log_path.exists()
    assert task_path.exists()

    decision_payload = json.loads(decision_path.read_text(encoding="utf-8"))
    task_file_payload = json.loads(task_path.read_text(encoding="utf-8"))
    install_log = install_log_path.read_text(encoding="utf-8")
    config_text = config_path.read_text(encoding="utf-8")
    command_log = command_log_path.read_text(encoding="utf-8")

    assert task_file_payload == task_payload
    assert decision_payload["task"] == TASK_NAME
    assert decision_payload["source"] == SOURCE_NAME
    assert task_payload["status"] in {"pass", "blocked"}
    assert decision_payload["status"] == task_payload["status"]
    assert AUTHORITATIVE_INSTALL_COMMAND in install_log
    assert AUTHORITATIVE_CLI_COMMAND in install_log
    assert config_text.strip()
    assert decision_payload["command_log_path"].endswith("command.log")
    assert task_payload["command_log_path"].endswith("command.log")
    assert (
        "authoritative_cli_command=reinvent -l sampling.log sampling.toml"
        in command_log
    )
    assert "planned_sampling_command=" in command_log

    if task_payload["status"] == "pass":
        assert decision_payload["failure_class"] is None
        assert decision_payload["produced_output_files"]
        assert "sampling.log" not in decision_payload["produced_output_files"]
        assert not error_log_path.exists()
    else:
        assert error_log_path.exists()
        assert decision_payload["failure_class"] in {
            "documentation-ambiguity",
            "environment-prerequisite",
            "reinvent4-runtime",
            "internal-error",
        }
        error_log = error_log_path.read_text(encoding="utf-8")
        assert "error_type=RuntimeError" in error_log

        if decision_payload["failure_class"] == "documentation-ambiguity":
            assert (
                decision_payload["failure_kind"]
                == "unresolved_pocket_constraint_format"
            )
            assert (
                "unresolved_pocket_constraint_format"
                in decision_payload["blocking_reasons"]
            )
            assert 'official_config_status = "ambiguous"' in config_text
            assert "reinvent_spec_found=" in install_log


def test_phase0_reinvent4_smoke_requires_non_bookkeeping_outputs_for_pass(
    tmp_path: Path,
) -> None:
    confirmed_probe = {
        "status": "confirmed",
        "summary": "Official REINVENT4 sampling TOML confirmed for bounded smoke testing.",
        "references": ["reinvent4-sampling-runtime-note"],
        "config_text": 'run_type = "sampling"\n[parameters]\nnum_smiles = 1\n',
    }

    fake_cli_path = "/fake/bin/reinvent"

    def fake_success_executor(
        command: Sequence[str],
        timeout_seconds: float,
        cwd: Path | None,
    ) -> subprocess.CompletedProcess[str]:
        if (
            command[:2] == [sys.executable, "-c"]
            and "reinvent_spec_found" in command[2]
        ):
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps(
                    {
                        "reinvent_spec_found": True,
                        "reinvent_origin": "/fake/site-packages/reinvent/__init__.py",
                    }
                )
                + "\n",
                stderr="",
            )

        if cwd is None:
            raise AssertionError(
                "REINVENT4 sampling command should run inside the output directory."
            )
        _ = (cwd / "sampling.log").write_text("sampling complete\n", encoding="utf-8")
        _ = (cwd / "generated.smi").write_text("CCO generated-1\n", encoding="utf-8")
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
            2,
            stdout="",
            stderr="Runtime rejected the sampling config after startup",
        )

    def fake_bookkeeping_only_executor(
        command: Sequence[str],
        timeout_seconds: float,
        cwd: Path | None,
    ) -> subprocess.CompletedProcess[str]:
        if command[:2] == [sys.executable, "-c"]:
            return fake_success_executor(command, timeout_seconds, cwd)
        if cwd is None:
            raise AssertionError(
                "REINVENT4 sampling command should run inside the output directory."
            )
        _ = (cwd / "sampling.log").write_text("sampling complete\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

    pass_output_dir = tmp_path / "pass-run"
    bookkeeping_only_output_dir = tmp_path / "bookkeeping-only-run"
    blocked_output_dir = tmp_path / "blocked-run"
    pass_task = run_smoke_slice(
        pass_output_dir,
        command_executor=fake_success_executor,
        which_resolver=lambda _: fake_cli_path,
        official_config_probe=confirmed_probe,
    )
    bookkeeping_only_task = run_smoke_slice(
        bookkeeping_only_output_dir,
        command_executor=fake_bookkeeping_only_executor,
        which_resolver=lambda _: fake_cli_path,
        official_config_probe=confirmed_probe,
    )
    blocked_task = run_smoke_slice(
        blocked_output_dir,
        command_executor=fake_runtime_blocked_executor,
        which_resolver=lambda _: fake_cli_path,
        official_config_probe=confirmed_probe,
    )

    assert pass_task["status"] == "pass"
    assert bookkeeping_only_task["status"] == "blocked"
    assert blocked_task["status"] == "blocked"

    pass_decision = json.loads(
        (pass_output_dir / "decision.json").read_text(encoding="utf-8")
    )
    bookkeeping_only_decision = json.loads(
        (bookkeeping_only_output_dir / "decision.json").read_text(encoding="utf-8")
    )
    blocked_decision = json.loads(
        (blocked_output_dir / "decision.json").read_text(encoding="utf-8")
    )
    bookkeeping_only_error_log = (
        bookkeeping_only_output_dir / "task-error.log"
    ).read_text(encoding="utf-8")
    blocked_error_log = (blocked_output_dir / "task-error.log").read_text(
        encoding="utf-8"
    )
    pass_install_log = (pass_output_dir / "install.log").read_text(encoding="utf-8")
    pass_config = (pass_output_dir / "config.toml").read_text(encoding="utf-8")
    pass_command_log = (pass_output_dir / "command.log").read_text(encoding="utf-8")
    bookkeeping_only_command_log = (
        bookkeeping_only_output_dir / "command.log"
    ).read_text(encoding="utf-8")

    assert pass_decision["produced_output_files"] == ["generated.smi"]
    assert bookkeeping_only_decision["failure_class"] == "reinvent4-runtime"
    assert bookkeeping_only_decision["failure_kind"] == "missing_output_artifacts"
    assert bookkeeping_only_decision["produced_output_files"] == []
    assert blocked_decision["failure_class"] == "reinvent4-runtime"
    assert blocked_decision["failure_kind"] == "config_validation_failure"
    assert build_sampling_command(fake_cli_path, pass_output_dir / "config.toml") == [
        "/fake/bin/reinvent",
        "-l",
        "sampling.log",
        "config.toml",
    ]
    assert AUTHORITATIVE_CLI_COMMAND in pass_install_log
    assert pass_config == confirmed_probe["config_text"]
    assert "failure_kind=missing_output_artifacts" in bookkeeping_only_error_log
    assert "Runtime rejected the sampling config" in blocked_error_log
    assert "sampling_command_executed=True" in pass_command_log
    assert (
        "executed_sampling_command=/fake/bin/reinvent -l sampling.log config.toml"
        in pass_command_log
    )
    assert "failure_kind=missing_output_artifacts" in bookkeeping_only_command_log
