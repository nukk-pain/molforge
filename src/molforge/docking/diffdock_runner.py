"""DiffDock-L runner (v4 Track D1).

Wraps the Modal-deployed `run_diffdock_l` function. Same structural
pattern as `reselect_via_boltz2` in `reselect.py`:

  1. Caller provides protein PDB bytes + ligand SMILES.
  2. Modal function runs DiffDock-L inference on A100.
  3. Output SDF files staged into a local output_dir.
  4. `parse_diffdock_samples()` extracts per-pose metadata into
     `DiffDockPose` dataclasses.

DiffDock-L writes per-pose SDFs with filenames encoding rank and
confidence: `rank{N}.sdf` or `rank{N}_confidence-{score}.sdf`
(the latter when `--save_visualisation true`, but we leave it off
and read confidence from `confidence.npy` / the `ranks/` side files
when present). Since exact filename conventions can shift between
DiffDock versions, the parser is defensive: it globs all `*.sdf`
under output_dir and parses the rank / confidence from the stem
with regex, falling back to deterministic order when confidence is
absent.
"""

from __future__ import annotations

import base64
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


_RANK_PATTERN = re.compile(r"rank(\d+)(?:_confidence-?(-?\d+(?:\.\d+)?))?", re.I)


@dataclass(frozen=True, slots=True)
class DiffDockPose:
    """A single DiffDock-L diffusion sample.

    rank:       1-based rank reported by DiffDock's confidence model
                (lower rank = higher confidence).
    confidence: DiffDock confidence score (higher = more confident).
                None when filename / side files don't expose it.
    """

    rank: int
    structure_path: Path  # SDF emitted by `python -m inference`
    confidence: float | None = None
    sampler: str = "diffdock_l"
    auxiliary: dict[str, object] = field(default_factory=dict)

    @property
    def sample_index(self) -> int:
        """Expose rank as 0-indexed sample_index for cross-sampler joins."""
        return self.rank - 1


@dataclass(frozen=True, slots=True)
class DiffDockResult:
    samples: list[DiffDockPose]
    top_rank_sample: DiffDockPose | None
    output_dir: Path
    cost_estimate_usd: float
    stdout_tail: str
    stderr_tail: str


def parse_diffdock_samples(output_dir: Path) -> list[DiffDockPose]:
    """Walk output_dir for DiffDock SDFs + confidence metadata.

    Accepts both flat-layout (all SDFs at `output_dir/*.sdf`) and
    per-complex subdirectory layout (`output_dir/{complex}/*.sdf`).

    DiffDock-L writes each pose twice when confidence is available:
    `rank{N}.sdf` (without confidence score) and
    `rank{N}_confidence-{score}.sdf` (with score embedded in filename).
    We prefer the `_confidence-` variant so we retain the score, and
    drop the plain `rank{N}.sdf` duplicate.
    """
    sdf_paths: list[Path] = []
    for sdf in output_dir.rglob("*.sdf"):
        if sdf.is_file():
            sdf_paths.append(sdf)
    if not sdf_paths:
        return []

    # Dedupe: when `rank{N}_confidence-*.sdf` exists, drop `rank{N}.sdf`.
    ranked_with_confidence: set[int] = set()
    for sdf in sdf_paths:
        m = _RANK_PATTERN.search(sdf.stem)
        if m and m.group(2) is not None:
            ranked_with_confidence.add(int(m.group(1)))
    filtered: list[Path] = []
    for sdf in sdf_paths:
        m = _RANK_PATTERN.search(sdf.stem)
        if m and m.group(2) is None and int(m.group(1)) in ranked_with_confidence:
            continue  # drop plain rank{N}.sdf when _confidence variant exists
        filtered.append(sdf)
    sdf_paths = filtered

    # Look for a sidecar `confidences.npy` / `confidence.txt` if present;
    # otherwise extract from filename.
    sidecar_confidence: dict[int, float] = {}
    for txt in output_dir.rglob("confidence*.txt"):
        try:
            for line_no, line in enumerate(txt.read_text().splitlines(), start=1):
                try:
                    sidecar_confidence[line_no] = float(line.strip())
                except ValueError:
                    continue
        except OSError:
            continue

    samples: list[DiffDockPose] = []
    for sdf in sorted(sdf_paths):
        stem = sdf.stem
        match = _RANK_PATTERN.search(stem)
        if match:
            rank = int(match.group(1))
            confidence_str = match.group(2)
            confidence = (
                float(confidence_str) if confidence_str is not None else None
            )
        else:
            # Fallback: assign rank from sorted order (1-based).
            rank = len(samples) + 1
            confidence = None

        if confidence is None and rank in sidecar_confidence:
            confidence = sidecar_confidence[rank]

        samples.append(
            DiffDockPose(
                rank=rank,
                structure_path=sdf,
                confidence=confidence,
            )
        )

    samples.sort(key=lambda s: s.rank)
    return samples


# ---------------------------------------------------------------------------
# Modal dispatch
# ---------------------------------------------------------------------------


# Type alias for an injectable runner (used by tests to bypass Modal).
# The runner receives the payload that would otherwise be sent to the
# Modal function and returns the same JobResult-compatible dict.
DiffDockModalRunner = Callable[[dict], dict]


def _run_diffdock_via_modal(payload: dict) -> dict:
    """Default runner — calls the deployed Modal function."""
    import modal

    try:
        fn = modal.Function.from_name(
            "molforge-diffdock-l", "run_diffdock_l"
        )
    except (modal.exception.NotFoundError, modal.exception.InvalidError) as exc:
        raise RuntimeError(
            "Modal app 'molforge-diffdock-l' not deployed. Run\n"
            "  modal deploy src/molforge/remote/diffdock_modal_app.py\n"
            f"Underlying error: {type(exc).__name__}: {exc}"
        ) from exc
    except modal.exception.AuthError as exc:
        raise RuntimeError(
            "Modal auth failed. Run `modal token new` and retry.\n"
            f"Underlying error: {type(exc).__name__}: {exc}"
        ) from exc

    call = fn.spawn(
        payload["protein_pdb_b64"],
        payload["ligand_smiles"],
        payload.get("num_samples", 10),
        payload.get("complex_name", "query"),
        payload.get("inference_steps", 20),
    )
    result = call.get(timeout=payload.get("timeout_seconds", 1800))
    if not isinstance(result, dict):
        raise ValueError("DiffDock Modal returned a non-dict result.")
    return {str(k): v for k, v in result.items()}


def run_diffdock_l(
    *,
    protein_pdb_bytes: bytes,
    ligand_smiles: str,
    num_samples: int = 10,
    complex_name: str = "query",
    inference_steps: int = 20,
    timeout_seconds: int = 1800,
    output_dir: Path | None = None,
    runner: DiffDockModalRunner | None = None,
) -> DiffDockResult:
    """Run DiffDock-L and return parsed samples.

    `runner` is injectable so tests can substitute a mock that returns
    a deterministic output_files payload (avoiding Modal in CI).
    """
    payload = {
        "protein_pdb_b64": base64.b64encode(protein_pdb_bytes).decode("ascii"),
        "ligand_smiles": ligand_smiles,
        "num_samples": num_samples,
        "complex_name": complex_name,
        "inference_steps": inference_steps,
        "timeout_seconds": timeout_seconds,
    }

    runner = runner or _run_diffdock_via_modal
    result = runner(payload)

    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="molforge-diffdock-"))
    output_dir.mkdir(parents=True, exist_ok=True)

    for rel_path, b64_str in (result.get("output_files") or {}).items():
        destination = output_dir / rel_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(base64.b64decode(b64_str))

    samples = parse_diffdock_samples(output_dir)
    top = samples[0] if samples else None

    return DiffDockResult(
        samples=samples,
        top_rank_sample=top,
        output_dir=output_dir,
        cost_estimate_usd=float(result.get("cost_estimate_usd") or 0.0),
        stdout_tail=str(result.get("stdout") or "")[-500:],
        stderr_tail=str(result.get("stderr") or "")[-500:],
    )
