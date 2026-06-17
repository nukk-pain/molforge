"""Modal app for the molforge RemoteGPUBackend.

Deployed via: `modal deploy src/molforge/remote/modal_app.py`
ModalBackend binds to this via `modal.Function.from_name("molforge-remote-gpu", "run_job")`.

The `run_job` function is a generic JobSpec runner:
payload shape (produced by `ModalBackend._run_job_via_modal`):
    {
        "image": str (informational only — the deployed function image is fixed below),
        "args": list[str],
        "input_files": dict[str, base64_str],
        "timeout_seconds": int,
        "env": dict[str, str],
    }
return shape (consumed by `ModalBackend.fetch_result`):
    {
        "success": bool,
        "returncode": int,
        "stdout": str,
        "stderr": str,
        "output_files": dict[str, base64_str],
        "elapsed": float,
        "cost_estimate_usd": float,
    }
"""

from __future__ import annotations

import modal

app = modal.App("molforge-remote-gpu")

# Image layer caches the V4-0.5 boltz build. Anyone calling run_job gets a
# container with boltz CLI, rdkit, and torch already installed.
image = (
    # Use an NVIDIA-official CUDA 13 runtime base so libnvrtc-builtins.so.13.0
    # is present (Modal's GPU hosts report driver 580.x == CUDA 13). Boltz-2
    # 2.2.1 + torch compiled against cu124/cu121 fails during JIT because
    # nvrtc tries to open the driver's NVRTC 13 library. With the official
    # CUDA 13 image, the matching builtins ship in the container.
    modal.Image.from_registry(
        "nvidia/cuda:13.0.0-devel-ubuntu24.04",
        add_python="3.11",
    )
    .apt_install("build-essential", "git")
    .pip_install(
        "boltz[cuda]==2.2.1",
    )
)

# Persistent volume so Boltz-2 weights survive across invocations
# (carried over from V4-0.5).
weights_volume = modal.Volume.from_name(
    "molforge-boltz2-weights", create_if_missing=True
)

# Informational estimate only. Modal's real billing is authoritative —
# see the Modal dashboard / `modal.functions.FunctionStats`. Keep this
# value conservative; callers should not treat `cost_estimate_usd` as
# invoicing truth.
A100_RATE_USD_PER_SECOND = 2.10 / 3600.0


# NOTE (v3 A1 follow-ups, tracked in PROGRESS):
#   - Q1 env allowlist: run_job currently inherits os.environ; a stricter
#     allowlist (PATH/HOME/CUDA/BOLTZ_CACHE) would reduce the chance of
#     Modal-injected vars (MODAL_TASK_ID etc.) leaking into subprocesses.
#   - Q2 output diff: pre_existing path-set ignores overwritten or
#     deleted-and-recreated files; switching to (mtime, size) tuples would
#     surface in-place mutation. Revisit when A2 flows real Boltz inputs.
#   - Q3 multi-function deploy: if A2/A3 needs a different boltz version,
#     add a sibling `@app.function` rather than re-pinning this one, so
#     callers can select via ModalBackend(function_name="run_job_boltz_0_5_x").


@app.function(
    image=image,
    gpu="A100",
    timeout=1800,
    volumes={"/root/.boltz": weights_volume},
)
def run_job(payload: dict) -> dict:
    """Generic JobSpec runner on an A100 container.

    Writes input_files to a temp dir, shells out to `args`, captures stdout /
    stderr / returncode / output artifacts, and reports GPU-invocation cost.
    """
    import base64
    import os
    import subprocess
    import tempfile
    import time
    from pathlib import Path

    args = [str(a) for a in (payload.get("args") or [])]
    if not args:
        raise ValueError("JobSpec.args must contain at least one element.")

    # v3 A2-1 (A1 follow-up Q1): env allowlist — don't inherit Modal-injected
    # secrets (MODAL_TASK_ID/MODAL_TOKEN_*) into the subprocess. Start from a
    # minimal base and let user_env override on top.
    _ENV_ALLOWLIST = (
        "PATH",
        "HOME",
        "LANG",
        "LC_ALL",
        "TMPDIR",
        "BOLTZ_CACHE",
        "CUDA_VISIBLE_DEVICES",
        "LD_LIBRARY_PATH",
        "NVIDIA_VISIBLE_DEVICES",
        "NVIDIA_DRIVER_CAPABILITIES",
    )
    env: dict[str, str] = {
        key: os.environ[key] for key in _ENV_ALLOWLIST if key in os.environ
    }
    env.setdefault("BOLTZ_CACHE", "/root/.boltz")
    for key, value in (payload.get("env") or {}).items():
        env[str(key)] = str(value)

    timeout_seconds = int(payload.get("timeout_seconds") or 600)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        # Write input_files into the tempdir, preserving relative paths.
        for relative_path, encoded in (payload.get("input_files") or {}).items():
            destination = tmp_path / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(base64.b64decode(encoded))

        # v3 A2-1 (A1 follow-up Q2): snapshot (relpath, size) so files that are
        # rewritten in place or deleted-and-recreated are detected as outputs.
        def _snapshot(root: Path) -> dict[str, int]:
            snap: dict[str, int] = {}
            for path in root.rglob("*"):
                if path.is_file():
                    snap[str(path.relative_to(root))] = path.stat().st_size
            return snap

        pre_snapshot = _snapshot(tmp_path)

        started_at = time.perf_counter()
        completed = subprocess.run(
            args,
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        elapsed = round(time.perf_counter() - started_at, 6)

        post_snapshot = _snapshot(tmp_path)

        # File is an output if (a) new path, or (b) existing path with changed
        # size. Size is a cheap proxy for "content differs"; mtime is
        # untrustworthy inside container runtimes.
        output_files: dict[str, str] = {}
        for path in tmp_path.rglob("*"):
            if not path.is_file():
                continue
            rel = str(path.relative_to(tmp_path))
            if rel in pre_snapshot and pre_snapshot[rel] == post_snapshot.get(rel):
                continue
            output_files[rel] = base64.b64encode(path.read_bytes()).decode("ascii")

        return {
            "success": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "output_files": output_files,
            "elapsed": elapsed,
            "cost_estimate_usd": round(elapsed * A100_RATE_USD_PER_SECOND, 6),
        }


if __name__ == "__main__":
    # Informational guard: Modal apps are normally deployed via
    # `modal deploy src/molforge/remote/modal_app.py`, not by executing
    # this module directly.
    print(
        "This module defines a Modal app. Deploy with:\n"
        "  modal deploy src/molforge/remote/modal_app.py\n"
        "Then call via molforge.remote.ModalBackend()."
    )
