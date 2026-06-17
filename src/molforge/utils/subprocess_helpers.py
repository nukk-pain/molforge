"""Subprocess helpers for external-tool orchestration.

Key guarantees:
  * Stdout/stderr streamed to log files in real time. `subprocess.run(
    capture_output=True, timeout=N)` drops the entire buffer when
    TimeoutExpired fires — a 1-hour biocompute hang left us with 0 bytes
    of logs to diagnose. This helper uses `Popen` with file handles so
    the partial output survives kills.
  * Relative-path caller cwd gotcha: if the caller passes a relative
    output path and the subprocess cwd is set elsewhere, the external
    tool writes into *its own* tree instead of the caller's. Helper
    does NOT resolve paths — that's the caller's responsibility — but
    docs the contract clearly.
  * Env allowlist: by default, subprocesses inherit the minimal
    PATH / HOME / USER / LANG / TMPDIR from the parent. Prevents
    molforge's own Modal tokens / API keys from leaking into external
    tool execution contexts.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


_ENV_ALLOWLIST = ("PATH", "HOME", "USER", "LANG", "LC_ALL", "TMPDIR")


def safe_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Build a subprocess env limited to an allowlist of parent vars,
    plus optional `extra` overrides.

    Do NOT pass `os.environ` directly to subprocess.run / Popen — it
    leaks API tokens and secrets (Modal, Anthropic, etc.) into child
    processes that have no business seeing them.
    """
    env = {k: os.environ[k] for k in _ENV_ALLOWLIST if k in os.environ}
    if extra:
        env.update({str(k): str(v) for k, v in extra.items()})
    return env


@dataclass(frozen=True, slots=True)
class StreamingSubprocessResult:
    returncode: int
    stdout_text: str
    stderr_text: str
    stdout_path: Path
    stderr_path: Path
    elapsed_seconds: float
    timed_out: bool


def run_streaming_subprocess(
    args: list[str],
    *,
    cwd: Path | str,
    stdout_path: Path,
    stderr_path: Path,
    env: dict[str, str] | None = None,
    timeout: int | None = None,
    check: bool = False,
) -> StreamingSubprocessResult:
    """Run `args` with real-time log streaming.

    Unlike `subprocess.run(capture_output=True)`, the stdout/stderr
    output is written to disk as the process emits it, so a TimeoutExpired
    kill does NOT destroy the partial buffer — the logs on disk are the
    authoritative record of what the child did before we terminated it.

    Arguments:
      args: exec argv
      cwd: subprocess working directory (absolute path recommended —
           the helper does not resolve paths for you)
      stdout_path / stderr_path: files created and streamed to
      env: explicit env dict. Pass `safe_env()` for the default
           allowlist-restricted view. If None, inherits the full parent
           env — that's usually wrong for external tools.
      timeout: seconds before SIGKILL (None disables)
      check: if True, raise CalledProcessError on non-zero exit

    Returns a StreamingSubprocessResult with the full log text already
    read back from disk, so callers don't need to re-open the files.
    """
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    timed_out = False
    with open(stdout_path, "w", encoding="utf-8") as out_f, \
         open(stderr_path, "w", encoding="utf-8") as err_f:
        proc = subprocess.Popen(
            args,
            cwd=str(cwd),
            env=env,
            stdout=out_f,
            stderr=err_f,
            text=True,
        )
        try:
            returncode = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            returncode = proc.wait()
            timed_out = True
            err_f.write(
                f"\n-- TIMEOUT after {timeout}s — process killed --\n"
            )

    elapsed = round(time.perf_counter() - started, 2)
    stdout_text = stdout_path.read_text(encoding="utf-8") if stdout_path.exists() else ""
    stderr_text = stderr_path.read_text(encoding="utf-8") if stderr_path.exists() else ""

    if check and returncode != 0 and not timed_out:
        raise subprocess.CalledProcessError(
            returncode=returncode,
            cmd=args,
            output=stdout_text,
            stderr=stderr_text,
        )

    return StreamingSubprocessResult(
        returncode=returncode,
        stdout_text=stdout_text,
        stderr_text=stderr_text,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        elapsed_seconds=elapsed,
        timed_out=timed_out,
    )
