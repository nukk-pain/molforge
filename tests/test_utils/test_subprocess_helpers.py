# pyright: reportMissingImports=false
"""Regression shield for subprocess timeout/logging bugs."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from molforge.utils.subprocess_helpers import (  # noqa: E402
    run_streaming_subprocess,
    safe_env,
)


def test_safe_env_restricts_to_allowlist(monkeypatch):
    monkeypatch.setenv("PATH", "/bin:/usr/bin")
    monkeypatch.setenv("MODAL_TOKEN_ID", "secret-should-not-leak")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret-should-not-leak")
    env = safe_env()
    assert "PATH" in env
    assert "MODAL_TOKEN_ID" not in env, (
        "safe_env must exclude non-allowlisted env vars so subprocesses "
        "don't inherit molforge's Modal/Anthropic secrets"
    )
    assert "ANTHROPIC_API_KEY" not in env


def test_safe_env_accepts_extras(monkeypatch):
    monkeypatch.setenv("PATH", "/bin")
    env = safe_env(extra={"MY_VAR": "value"})
    assert env["MY_VAR"] == "value"
    assert env["PATH"] == "/bin"


def test_streaming_subprocess_captures_stdout_and_stderr(tmp_path):
    stdout_path = tmp_path / "stdout.log"
    stderr_path = tmp_path / "stderr.log"
    result = run_streaming_subprocess(
        ["sh", "-c", "echo hello-stdout; echo hello-stderr 1>&2; exit 0"],
        cwd=tmp_path,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        env=safe_env(),
    )
    assert result.returncode == 0
    assert "hello-stdout" in result.stdout_text
    assert "hello-stderr" in result.stderr_text
    # Log files persisted to disk (not just returned in memory) — the whole
    # point of the streaming helper.
    assert stdout_path.exists() and "hello-stdout" in stdout_path.read_text()
    assert stderr_path.exists() and "hello-stderr" in stderr_path.read_text()


def test_streaming_subprocess_timeout_preserves_partial_stdout(tmp_path):
    """subprocess.run(capture_output=True,
    timeout=N) drops the buffer on timeout; Popen with file streams keeps it.
    This test emits output, then sleeps past the timeout so we're forced
    to kill it. The output produced before the sleep must survive in the
    log file even though the process was killed."""
    stdout_path = tmp_path / "stdout.log"
    stderr_path = tmp_path / "stderr.log"
    result = run_streaming_subprocess(
        [
            "sh", "-c",
            "echo partial-before-timeout && sleep 10 && echo should-not-reach",
        ],
        cwd=tmp_path,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        env=safe_env(),
        timeout=2,
    )
    assert result.timed_out is True
    assert "partial-before-timeout" in result.stdout_text, (
        "partial output emitted before the timeout must survive the kill"
    )
    assert "should-not-reach" not in result.stdout_text
    # Timeout marker in stderr for traceability.
    assert "TIMEOUT" in result.stderr_text


def test_streaming_subprocess_check_raises_on_nonzero_exit(tmp_path):
    import subprocess

    with pytest.raises(subprocess.CalledProcessError):
        run_streaming_subprocess(
            ["sh", "-c", "exit 42"],
            cwd=tmp_path,
            stdout_path=tmp_path / "out.log",
            stderr_path=tmp_path / "err.log",
            env=safe_env(),
            check=True,
        )


def test_streaming_subprocess_respects_absolute_cwd(tmp_path):
    """Regression: external tools invoked with a subprocess cwd will
    interpret any relative paths they receive relative to THEIR cwd,
    not the caller's. The helper does not re-resolve paths (by design),
    but its cwd honour is tested here as a canary — a subprocess that
    writes a file must create it under the given cwd."""
    work_dir = tmp_path / "elsewhere"
    work_dir.mkdir()
    result = run_streaming_subprocess(
        ["sh", "-c", "touch created-by-child.txt"],
        cwd=work_dir,
        stdout_path=tmp_path / "out.log",
        stderr_path=tmp_path / "err.log",
        env=safe_env(),
    )
    assert result.returncode == 0
    assert (work_dir / "created-by-child.txt").exists()
