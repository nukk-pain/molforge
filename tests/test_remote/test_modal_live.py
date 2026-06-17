# pyright: reportMissingImports=false
"""Live Modal smoke for v3 A1.

Skipped unless `REMOTE_GPU_LIVE=1` is exported AND `modal` is installed
AND the `molforge-remote-gpu` app has been deployed via
`scripts/deploy_modal_remote_gpu.sh`. The test submits a tiny JobSpec
(`boltz --version`) against the deployed function and asserts the
round-trip payload surface.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from molforge.remote import JobSpec, ModalBackend  # noqa: E402


@pytest.mark.skipif(
    os.environ.get("REMOTE_GPU_LIVE") != "1",
    reason="REMOTE_GPU_LIVE=1 not set — live Modal smoke is opt-in.",
)
def test_modal_live_boltz_version_roundtrip() -> None:
    pytest.importorskip("modal", reason="modal package not installed")

    backend = ModalBackend()
    spec = JobSpec(
        image="molforge-remote-gpu",
        args=["boltz", "--help"],
        input_files={},
        timeout_seconds=120,
        env={},
    )
    handle = backend.submit(spec)
    result = backend.fetch_result(handle)

    combined = (result.stdout + result.stderr).lower()
    assert result.success is True, f"stderr: {result.stderr}"
    assert "boltz" in combined
    assert "predict" in combined  # boltz --help enumerates subcommands
    assert result.cost_estimate_usd is not None
    assert 0.0 < result.cost_estimate_usd < 0.20  # well under our $0.50 ceiling


@pytest.mark.skipif(
    os.environ.get("REMOTE_GPU_LIVE") != "1",
    reason="REMOTE_GPU_LIVE=1 not set — live Modal smoke is opt-in.",
)
def test_modal_live_input_files_roundtrip() -> None:
    """v3 A2-1 (A1 follow-up Q7): input_files base64 → subprocess cwd → output_files."""
    pytest.importorskip("modal", reason="modal package not installed")

    backend = ModalBackend()
    spec = JobSpec(
        image="molforge-remote-gpu",
        args=["sh", "-c", "cat in.txt > out.txt && echo wrote"],
        input_files={"in.txt": b"hello-molforge"},
        timeout_seconds=120,
        env={},
    )
    handle = backend.submit(spec)
    result = backend.fetch_result(handle)

    assert result.success is True, f"stderr: {result.stderr}"
    assert "wrote" in result.stdout
    assert "out.txt" in result.output_files
    assert result.output_files["out.txt"] == b"hello-molforge"
    assert result.cost_estimate_usd is not None
    assert result.cost_estimate_usd < 0.20


@pytest.mark.skipif(
    os.environ.get("REMOTE_GPU_LIVE") != "1",
    reason="REMOTE_GPU_LIVE=1 not set — live Modal smoke is opt-in.",
)
def test_modal_live_env_allowlist_blocks_modal_secrets() -> None:
    """v3 A2-1 (A1 follow-up Q1): MODAL_* env vars must NOT leak into subprocess."""
    pytest.importorskip("modal", reason="modal package not installed")

    backend = ModalBackend()
    spec = JobSpec(
        image="molforge-remote-gpu",
        # printenv + grep returns non-zero exit when no match, so pipe to cat to
        # get zero exit regardless. The assertion is that no MODAL_ vars print.
        args=["sh", "-c", "printenv | grep -c '^MODAL_' || true"],
        input_files={},
        timeout_seconds=120,
        env={},
    )
    handle = backend.submit(spec)
    result = backend.fetch_result(handle)

    assert result.success is True, f"stderr: {result.stderr}"
    count = int(result.stdout.strip() or "0")
    assert count == 0, f"MODAL_ env vars leaked into subprocess: {count} found"
