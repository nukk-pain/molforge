from __future__ import annotations

import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class JobSpec:
    image: str
    args: list[str]
    input_files: dict[str, bytes]
    timeout_seconds: int
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class JobHandle:
    handle_id: str
    provider: str


@dataclass(frozen=True, slots=True)
class JobResult:
    success: bool
    stdout: str
    stderr: str
    output_files: dict[str, bytes]
    elapsed: float
    cost_estimate_usd: float


class RemoteGPUBackend(Protocol):
    name: str

    def submit(self, job: JobSpec) -> JobHandle: ...

    def fetch_result(self, handle: JobHandle) -> JobResult: ...


class LocalMockBackend:
    name = "local-mock"

    def __init__(self, *, cost_per_second_usd: float = 0.0) -> None:
        self.cost_per_second_usd = cost_per_second_usd
        self._jobs: dict[str, JobSpec] = {}

    def submit(self, job: JobSpec) -> JobHandle:
        handle = JobHandle(handle_id=str(uuid4()), provider=self.name)
        self._jobs[handle.handle_id] = job
        return handle

    def fetch_result(self, handle: JobHandle) -> JobResult:
        job = self._jobs.pop(handle.handle_id, None)
        if job is None:
            raise ValueError(f"Unknown local mock job handle: {handle.handle_id}")

        with tempfile.TemporaryDirectory(prefix="molforge-local-mock-") as tmpdir:
            workdir = Path(tmpdir)
            for relative_path, payload in job.input_files.items():
                destination = workdir / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(payload)

            started_at = time.perf_counter()
            completed = subprocess.run(
                job.args,
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=job.timeout_seconds,
                check=False,
                env={**os.environ, **job.env},
            )
            elapsed = round(time.perf_counter() - started_at, 6)
            output_files = _collect_output_files(
                workdir, exclude_paths=set(job.input_files.keys())
            )
            return JobResult(
                success=completed.returncode == 0,
                stdout=completed.stdout,
                stderr=completed.stderr,
                output_files=output_files,
                elapsed=elapsed,
                cost_estimate_usd=round(elapsed * self.cost_per_second_usd, 6),
            )


def _collect_output_files(root: Path, *, exclude_paths: set[str]) -> dict[str, bytes]:
    outputs: dict[str, bytes] = {}
    normalized_excludes = {Path(path).as_posix() for path in exclude_paths}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative_path = path.relative_to(root).as_posix()
        if relative_path in normalized_excludes:
            continue
        outputs[relative_path] = path.read_bytes()
    return outputs
