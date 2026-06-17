"""Multi-sample Boltz-2 reselection (v3 A2).

Invokes the deployed `molforge-remote-gpu.run_job` Modal function with
`boltz predict --diffusion_samples N`, parses per-sample outputs, and picks
the top-affinity sample.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from molforge.remote import JobSpec, RemoteGPUBackend

from .multi_pose import PoseWithAffinity, parse_boltz_samples, write_boltz_input_yaml
from .pose_rescorer import PoseRescorer, RescoreResult


class _BackendLike(Protocol):
    def submit(self, spec: JobSpec): ...
    def fetch_result(self, handle): ...


@dataclass(frozen=True, slots=True)
class ReselectResult:
    samples: list[PoseWithAffinity]
    chosen_sample_index: int
    chosen_structure_path: Path
    method: str
    cost_estimate_usd: float
    stdout_tail: str
    stderr_tail: str
    rescore_results: tuple[RescoreResult, ...] = ()


def _select_from_candidates(
    candidates: list[PoseWithAffinity],
) -> tuple[PoseWithAffinity, str]:
    """Apply the existing 4-tier preference order to a candidate list."""
    by_pred = [s for s in candidates if s.affinity_pred_value is not None]
    if by_pred:
        chosen = min(by_pred, key=lambda s: s.affinity_pred_value or 0.0)
        return chosen, "affinity_pred_value"

    by_prob = [
        s for s in candidates if s.affinity_probability_binary is not None
    ]
    if by_prob:
        chosen = max(
            by_prob, key=lambda s: s.affinity_probability_binary or 0.0
        )
        return chosen, "affinity_probability_binary"

    by_iptm = [s for s in candidates if s.ligand_iptm is not None]
    if by_iptm:
        chosen = max(by_iptm, key=lambda s: s.ligand_iptm or 0.0)
        return chosen, "ligand_iptm_structural_fallback"

    by_conf = [s for s in candidates if s.confidence_score is not None]
    if by_conf:
        chosen = max(by_conf, key=lambda s: s.confidence_score or 0.0)
        return chosen, "confidence_score_fallback"

    return candidates[0], "fallback_sample_0"


def select_best_pose(
    samples: list[PoseWithAffinity],
    *,
    rescore_results: list[RescoreResult] | tuple[RescoreResult, ...] | None = None,
    rescorer_name: str | None = None,
) -> tuple[int, str]:
    """Pick the best sample index + method label.

    Rules:
      - rescorer_results=None → existing 4-tier on all samples.
      - rescorer_results with ≥1 valid + numeric score → argmin(score)
        on valid subset. Method: `{rescorer_name}_score` (Phase 5 — A').
      - rescorer_results with ≥1 valid but all scores None → 4-tier on
        valid subset, method prefixed with `{rescorer_name}+` (AC-2a,
        binary filter path used by PoseBustersFilter).
      - rescorer_results with 0 valid → 4-tier on *full* sample set
        (silent drop forbidden), method prefixed with
        `no_valid_pose_{rescorer_name}+` (AC-2b).

    Score convention: lower = better (Vina / Boltz affinity_pred_value).
    """
    if not samples:
        return 0, "fallback_sample_0"

    if rescore_results is None or rescorer_name is None:
        chosen, method = _select_from_candidates(samples)
        return samples.index(chosen), method

    valid_pairs = [
        (s, r) for s, r in zip(samples, rescore_results) if r.valid
    ]
    if not valid_pairs:
        chosen, method = _select_from_candidates(samples)
        return samples.index(chosen), f"no_valid_pose_{rescorer_name}+{method}"

    scored_pairs = [(s, r.score) for s, r in valid_pairs if r.score is not None]
    if scored_pairs:
        chosen, _ = min(scored_pairs, key=lambda pair: pair[1])
        return samples.index(chosen), f"{rescorer_name}_score"

    valid_samples = [s for s, _ in valid_pairs]
    chosen, method = _select_from_candidates(valid_samples)
    return samples.index(chosen), f"{rescorer_name}+{method}"


def reselect_via_boltz2(
    *,
    protein_sequence: str,
    ligand_smiles: str,
    backend: RemoteGPUBackend,
    diffusion_samples: int = 3,
    timeout_seconds: int = 1800,
    output_dir: Path | None = None,
    rescorer: PoseRescorer | None = None,
) -> ReselectResult:
    """Generate `diffusion_samples` Boltz-2 poses, pick best affinity.

    The caller passes a RemoteGPUBackend so tests and CI can inject a mock
    backend that returns deterministic output files (matching Boltz's file
    layout).

    If `rescorer` is provided, each pose is passed through
    `rescorer.score(...)` and the returned `RescoreResult` list is used to
    filter candidates before the 4-tier preference is applied. See
    `select_best_pose` for the precise rules.
    """
    yaml_body = write_boltz_input_yaml(
        protein_sequence=protein_sequence,
        ligand_smiles=ligand_smiles,
    )

    spec = JobSpec(
        image="molforge-remote-gpu",
        args=[
            "sh",
            "-c",
            "boltz predict input.yaml --out_dir out --diffusion_samples "
            f"{diffusion_samples} --use_msa_server --output_format pdb --no_kernels",
        ],
        input_files={"input.yaml": yaml_body.encode("utf-8")},
        timeout_seconds=timeout_seconds,
        env={},
    )

    handle = backend.submit(spec)
    result = backend.fetch_result(handle)

    if output_dir is None:
        import tempfile

        output_dir = Path(tempfile.mkdtemp(prefix="molforge-boltz2-reselect-"))
    output_dir.mkdir(parents=True, exist_ok=True)
    for relative_path, content in result.output_files.items():
        destination = output_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)

    samples = parse_boltz_samples(output_dir)

    rescore_results: tuple[RescoreResult, ...] = ()
    if rescorer is not None and samples:
        rescore_results = tuple(
            rescorer.score(
                complex_pdb=s.structure_path,
                ligand_smiles=ligand_smiles,
            )
            for s in samples
        )

    chosen_index, method = select_best_pose(
        samples,
        rescore_results=rescore_results or None,
        rescorer_name=rescorer.name if rescorer is not None else None,
    )

    chosen_path = (
        samples[chosen_index].structure_path if samples else output_dir
    )

    return ReselectResult(
        samples=samples,
        chosen_sample_index=chosen_index,
        chosen_structure_path=chosen_path,
        method=method,
        cost_estimate_usd=result.cost_estimate_usd or 0.0,
        stdout_tail=(result.stdout or "")[-500:],
        stderr_tail=(result.stderr or "")[-500:],
        rescore_results=rescore_results,
    )
