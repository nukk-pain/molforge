"""Modal app for DiffDock-L inference (v4 Track D1).

Separate from `modal_app.py` (Boltz) because DiffDock-L requires
Python 3.9 + torch 1.13 + CUDA 11.7 — incompatible with Boltz-2's
CUDA 13 / torch 2.x stack.

Deployed via:
  modal deploy src/molforge/remote/diffdock_modal_app.py

Invoked via `molforge.docking.diffdock_runner.run_diffdock_l(...)`,
which calls `modal.Function.from_name("molforge-diffdock-l",
"run_diffdock_l")` directly (no generic JobSpec dispatch — the
interface is DiffDock-specific).

The `run_diffdock_l` function takes protein PDB bytes + ligand SMILES
and returns a result dict matching the Boltz JobResult contract:
  {"success", "stdout", "stderr", "output_files": {rel: b64}, ...}

Weights cached in Modal Volume `molforge-diffdock-weights`, mounted at
`/workdir` and referenced via `--model_dir` + `--confidence_model_dir`
overrides on the inference CLI.
"""

from __future__ import annotations

import modal

app = modal.App("molforge-diffdock-l")

# Build DiffDock-L image from CUDA 11.7 base with Python 3.10 injected.
# Reason: the author-published rbgcsail/diffdock image uses Conda Python 3.9
# which Modal cannot introspect (fails with "unable to determine Python
# version"). Building from scratch gives Modal a Python it recognises AND
# lets us pin exact PyG wheel versions for cu117.
#
# DiffDock-L originally targets Python 3.9 + torch 1.13.1+cu117; cp310
# wheels for both torch 1.13.1+cu117 and the PyG `+pt113cu117` extensions
# are published, so Python 3.10 is a safe upgrade. All downstream libs
# (fair-esm, e3nn, prody, etc.) are Python-version-agnostic in this range.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:11.7.1-devel-ubuntu22.04",
        add_python="3.10",
    )
    # clang — prody source-build fallback.
    # ninja-build — openfold CUDA extensions prefer ninja over distutils.
    .apt_install("git", "build-essential", "clang", "ninja-build", "wget", "unzip")
    # Torch first (PyG wheels hard-pin against torch 1.13).
    .pip_install(
        "torch==1.13.1+cu117",
        extra_index_url="https://download.pytorch.org/whl/cu117",
    )
    # PyG CUDA extensions (prebuilt against torch 1.13 cu117, Python 3.10).
    .pip_install(
        "torch-scatter==2.1.0+pt113cu117",
        "torch-sparse==0.6.16+pt113cu117",
        "torch-cluster==1.6.0+pt113cu117",
        "torch-spline-conv==1.2.1+pt113cu117",
        find_links="https://data.pyg.org/whl/torch-1.13.1+cu117.html",
    )
    .pip_install(
        "torch-geometric==2.2.0",
        # DiffDock-L environment.yml pins (biopython intentionally omitted —
        # DiffDock doesn't import it; some prebuilt wheels fall back to
        # source build which requires clang, breaking the layer).
        "fair-esm==2.0.0",
        "e3nn==0.5.1",
        "prody==2.4.1",  # 2.4+ ships manylinux wheels; 2.2 was source-only.
        "rdkit==2022.3.3",
        "pytorch-lightning==1.9.5",
        "pybind11==2.11.1",
        "pyyaml",
        "networkx==2.8.4",
        "pandas==1.5.1",
        "scipy==1.12.0",
        "scikit-learn==1.1.0",
        "torchmetrics==0.11.0",
    )
    # dllogger is a tiny pure-Python logger that openfold uses — install it
    # normally (build-isolation is fine).
    .pip_install("git+https://github.com/NVIDIA/dllogger.git")
    # openfold pinned to DiffDock's target commit. Requirements:
    # - `--no-build-isolation` (setup.py imports torch + CUDA_HOME)
    # - `wheel`/`ninja`/`cmake` pre-installed so bdist_wheel works
    # - `CC=gcc CXX=g++` forces gcc (not clang); torch's _check_cuda_version
    #   rejects clang++ in the image because its reported version parses as
    #   "0.0.0" and fails the >=6.0.0 gate. gcc from build-essential passes.
    .pip_install("wheel", "ninja", "cmake")
    .run_commands(
        "CC=gcc CXX=g++ pip install --no-build-isolation "
        "git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307",
    )
    # Clone the DiffDock repo to /app — config + inference.py expected there.
    .run_commands(
        "git clone https://github.com/gcorso/DiffDock.git /app",
    )
)

# Persistent volume for DiffDock-L `workdir/v1.1/{score_model,
# confidence_model}` plus cached ESM-2 checkpoints. `inference.py`
# auto-downloads weights from GitHub releases on first run when the
# model_dir paths don't exist — after the first call completes, the
# volume retains them for all subsequent invocations.
weights_volume = modal.Volume.from_name(
    "molforge-diffdock-weights", create_if_missing=True
)

A100_RATE_USD_PER_SECOND = 2.10 / 3600.0


@app.function(
    image=image,
    gpu="A100",
    timeout=1800,
    # Single volume holds: (a) `/workdir/v1.1/{score,confidence}_model` —
    # DiffDock weights auto-downloaded from GitHub releases;
    # (b) ESM-2 checkpoints cached under `/workdir/.cache/torch/hub/` so
    # the first-call download cost (~650MB) pays back on every subsequent
    # call.
    volumes={"/workdir": weights_volume},
)
def run_diffdock_l(
    protein_pdb_b64: str,
    ligand_smiles: str,
    num_samples: int = 10,
    complex_name: str = "query",
    inference_steps: int = 20,
) -> dict:
    """Run DiffDock-L on a (protein, ligand) pair.

    Arguments:
      protein_pdb_b64: base64-encoded PDB bytes of the rigid receptor.
      ligand_smiles:   SMILES string of the ligand to dock.
      num_samples:     N poses to sample (default 10, matches config).
      complex_name:    Output subdirectory name inside DiffDock's out/.
      inference_steps: Diffusion steps; 20 matches default config.

    Returns a dict matching the Boltz `JobResult` contract so the
    downstream pose parser can be symmetric:
      {
        "success": bool,
        "returncode": int,
        "stdout": str,
        "stderr": str,
        "output_files": {rel_path: b64_str, ...},
        "elapsed": float,
        "cost_estimate_usd": float,
      }

    Weights are resolved at `/workdir/v1.1/{score_model,confidence_model}`.
    On first invocation these paths are empty, so DiffDock's inline
    downloader pulls `diffdock_models.zip` from GitHub releases. The
    `modal.Volume` persists the result, eliminating the cold-start
    cost on all subsequent runs.
    """
    import base64
    import os
    import subprocess
    import tempfile
    import time
    from pathlib import Path

    # Route fair-esm / torch.hub caches into the persistent volume so
    # ESM-2 weights download once and survive across invocations.
    torch_cache = Path("/workdir/.cache/torch")
    torch_cache.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(torch_cache)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        protein_path = tmp_path / "protein.pdb"
        protein_path.write_bytes(base64.b64decode(protein_pdb_b64))

        out_dir = tmp_path / "out"
        out_dir.mkdir()

        cmd = [
            "python",
            "-m",
            "inference",
            "--config",
            "/app/default_inference_args.yaml",
            # Override the config's relative paths so cwd doesn't matter.
            "--model_dir",
            "/workdir/v1.1/score_model",
            "--confidence_model_dir",
            "/workdir/v1.1/confidence_model",
            "--protein_path",
            str(protein_path),
            # NB: DiffDock-L's flag is `--ligand_description`, NOT `--ligand`
            # (accepts a SMILES string or a path to a molfile).
            "--ligand_description",
            ligand_smiles,
            "--out_dir",
            str(out_dir),
            "--samples_per_complex",
            str(num_samples),
            "--inference_steps",
            str(inference_steps),
            "--complex_name",
            complex_name,
            # `--save_visualisation` is action='store_true'; we want SDFs only,
            # so we omit the flag entirely (default False).
        ]

        started_at = time.perf_counter()
        try:
            completed = subprocess.run(
                cmd,
                cwd="/app",
                capture_output=True,
                text=True,
                timeout=1800,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "success": False,
                "returncode": -1,
                "stdout": (exc.stdout or "") if isinstance(exc.stdout, str) else "",
                "stderr": f"DiffDock-L timed out after 1800s",
                "output_files": {},
                "elapsed": 1800.0,
                "cost_estimate_usd": round(1800.0 * A100_RATE_USD_PER_SECOND, 6),
            }
        elapsed = round(time.perf_counter() - started_at, 6)

        output_files: dict[str, str] = {}
        if out_dir.exists():
            for f in out_dir.rglob("*"):
                if f.is_file():
                    rel = str(f.relative_to(out_dir))
                    output_files[rel] = base64.b64encode(f.read_bytes()).decode(
                        "ascii"
                    )

        # Commit the volume so auto-downloaded weights survive across
        # invocations. Safe even when nothing changed.
        try:
            weights_volume.commit()
        except Exception:  # noqa: BLE001 — best-effort; don't mask inference result
            pass

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
    print(
        "This module defines a Modal app. Deploy with:\n"
        "  modal deploy src/molforge/remote/diffdock_modal_app.py\n"
        "Then call via molforge.docking.diffdock_runner.run_diffdock_l()."
    )
