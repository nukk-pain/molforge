from __future__ import annotations

import base64
import time
from typing import Any, Callable
from uuid import uuid4

from .backend import JobHandle, JobResult, JobSpec


class ModalBackend:
    name = "modal"

    def __init__(
        self,
        *,
        app_name: str = "molforge-remote-gpu",
        function_name: str = "run_job",
        cost_per_second_usd: float = 2.10 / 3600.0,
        runner: Callable[[JobSpec], dict[str, Any]] | None = None,
    ) -> None:
        self.app_name = app_name
        self.function_name = function_name
        self.cost_per_second_usd = cost_per_second_usd
        self._runner = runner or self._run_job_via_modal
        self._jobs: dict[str, JobSpec] = {}

    def submit(self, job: JobSpec) -> JobHandle:
        handle = JobHandle(handle_id=str(uuid4()), provider=self.name)
        self._jobs[handle.handle_id] = job
        return handle

    def fetch_result(self, handle: JobHandle) -> JobResult:
        job = self._jobs.pop(handle.handle_id, None)
        if job is None:
            raise ValueError(f"Unknown Modal job handle: {handle.handle_id}")
        started_at = time.perf_counter()
        payload = self._runner(job)
        elapsed = round(time.perf_counter() - started_at, 6)
        output_files = {
            str(path): base64.b64decode(encoded)
            for path, encoded in _expect_dict(
                payload.get("output_files"), "output_files"
            ).items()
        }
        provider_elapsed = _optional_float(payload.get("elapsed"))
        effective_elapsed = (
            provider_elapsed if provider_elapsed is not None else elapsed
        )
        cost_estimate_usd = _optional_float(payload.get("cost_estimate_usd"))
        if cost_estimate_usd is None:
            cost_estimate_usd = round(effective_elapsed * self.cost_per_second_usd, 6)
        return JobResult(
            success=bool(payload.get("success")),
            stdout=str(payload.get("stdout") or ""),
            stderr=str(payload.get("stderr") or ""),
            output_files=output_files,
            elapsed=effective_elapsed,
            cost_estimate_usd=cost_estimate_usd,
        )

    def _run_job_via_modal(self, job: JobSpec) -> dict[str, Any]:
        import modal

        try:
            function = modal.Function.from_name(self.app_name, self.function_name)
        except (modal.exception.NotFoundError, modal.exception.InvalidError) as exc:
            raise RuntimeError(
                f"Modal app '{self.app_name}' (function '{self.function_name}') is "
                f"not deployed. Run `scripts/deploy_modal_remote_gpu.sh` first.\n"
                f"Underlying error: {type(exc).__name__}: {exc}"
            ) from exc
        except modal.exception.AuthError as exc:
            raise RuntimeError(
                "Modal authentication failed. Run `modal token new` (the token "
                "is stored in ~/.modal.toml) and retry.\n"
                f"Underlying error: {type(exc).__name__}: {exc}"
            ) from exc
        remote_call = function.spawn(
            {
                "image": job.image,
                "args": list(job.args),
                "input_files": {
                    path: base64.b64encode(payload).decode("ascii")
                    for path, payload in job.input_files.items()
                },
                "timeout_seconds": job.timeout_seconds,
                "env": dict(job.env),
            }
        )
        result = remote_call.get(timeout=job.timeout_seconds)
        if not isinstance(result, dict):
            raise ValueError("Modal backend returned a non-dict job result payload.")
        return {str(key): value for key, value in result.items()}


def _expect_dict(payload: object, label: str) -> dict[str, str]:
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected Modal {label} payload to be a dict.")
    return {str(key): str(value) for key, value in payload.items()}


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(
        f"Expected optional float-like Modal field, got {type(value).__name__}."
    )
