# pyright: reportMissingImports=false
from __future__ import annotations

import base64
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from molforge.remote.backend import JobSpec  # noqa: E402
from molforge.remote.modal_backend import ModalBackend  # noqa: E402


def test_modal_backend_round_trips_runner_payload() -> None:
    observed: dict[str, object] = {}

    def fake_runner(job: JobSpec):
        observed["image"] = job.image
        observed["args"] = job.args
        return {
            "success": True,
            "stdout": "ok",
            "stderr": "",
            "elapsed": 1.5,
            "output_files": {
                "result.json": base64.b64encode(b'{"status":"done"}').decode("ascii")
            },
        }

    backend = ModalBackend(runner=fake_runner, cost_per_second_usd=0.25)
    handle = backend.submit(
        JobSpec(
            image="ghcr.io/example/boltz:latest",
            args=["python", "job.py"],
            input_files={"input.json": b"{}"},
            timeout_seconds=30,
        )
    )

    result = backend.fetch_result(handle)

    assert observed["image"] == "ghcr.io/example/boltz:latest"
    assert observed["args"] == ["python", "job.py"]
    assert result.success is True
    assert result.elapsed == 1.5
    assert result.cost_estimate_usd == 0.375
    assert result.output_files["result.json"] == b'{"status":"done"}'


def test_modal_backend_raises_helpful_error_when_app_not_deployed() -> None:
    """v3 A1-4: Modal.Function.from_name 404 must surface as a deploy-hint
    RuntimeError rather than an opaque modal.exception.NotFoundError."""
    import pytest
    import types
    import sys as _sys

    # Build a minimal stand-in for the `modal` module so we don't require a
    # live modal installation for this unit test.
    fake_modal = types.ModuleType("modal")

    class _NotFoundError(Exception):
        pass

    class _InvalidError(Exception):
        pass

    fake_modal.exception = types.SimpleNamespace(
        NotFoundError=_NotFoundError, InvalidError=_InvalidError
    )

    class _Function:
        @staticmethod
        def from_name(app_name: str, function_name: str):
            raise _NotFoundError(f"no app {app_name!r} / function {function_name!r}")

    fake_modal.Function = _Function
    _sys.modules["modal"] = fake_modal
    try:
        backend = ModalBackend()
        handle = backend.submit(
            JobSpec(
                image="molforge-remote-gpu",
                args=["boltz", "--help"],
                input_files={},
                timeout_seconds=30,
            )
        )
        with pytest.raises(RuntimeError) as exc_info:
            backend.fetch_result(handle)
        message = str(exc_info.value)
        assert "molforge-remote-gpu" in message
        assert "deploy_modal_remote_gpu.sh" in message
    finally:
        _sys.modules.pop("modal", None)
