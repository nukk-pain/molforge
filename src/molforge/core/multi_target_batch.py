"""Multi-target batch runner for v3 C2.

Runs the v1/v2 full pipeline on N biocompute TargetCandidate JSON inputs and
aggregates cross-target metrics (unique SMILES, shared SMILES, top-N per
target). Per-target failures are recorded rather than aborting the batch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from contracts.schema import PipelineRun

from .input import load_target_candidates
from .pipeline import run_pipeline
from .store import MolforgeStore


@dataclass(frozen=True, slots=True)
class BatchTargetResult:
    target_gene: str
    status: str  # "completed" | f"failed:{ExcType}"
    run: PipelineRun | None = None
    error_message: str | None = None
    run_id: str | None = None
    candidate_count: int = 0


@dataclass(frozen=True, slots=True)
class BatchResult:
    per_target: list[BatchTargetResult]
    total_candidates: int
    unique_smiles_count: int
    shared_smiles_across_targets: list[str] = field(default_factory=list)
    per_target_top_smiles: dict[str, list[str]] = field(default_factory=dict)

    def completed_targets(self) -> list[BatchTargetResult]:
        return [r for r in self.per_target if r.status == "completed"]


def run_batch(
    input_paths: list[Path],
    *,
    disease: str,
    store_root: Path,
    top_n: int = 10,
    enable_live_chembl: bool = False,
    pipeline_runner: Any = run_pipeline,
) -> BatchResult:
    """Run the full pipeline for each input JSON.

    Per-target failures are caught and recorded; the batch continues with the
    remaining targets. `pipeline_runner` is injectable for tests.
    """
    store_root.mkdir(parents=True, exist_ok=True)

    per_target: list[BatchTargetResult] = []
    all_smiles: set[str] = set()
    target_smiles_sets: list[set[str]] = []
    per_target_top: dict[str, list[str]] = {}

    for input_path in input_paths:
        target_gene = input_path.stem
        try:
            targets = load_target_candidates(input_path, disease=disease)
            target_gene = targets[0].gene if targets else target_gene
            db_path = store_root / f"{target_gene}.db"
            with MolforgeStore(db_path) as store:
                run = pipeline_runner(
                    targets,
                    store=store,
                    top_n=top_n,
                    enable_live_chembl=enable_live_chembl,
                )
        except Exception as exc:  # noqa: BLE001 — honest per-target catch
            per_target.append(
                BatchTargetResult(
                    target_gene=target_gene,
                    status=f"failed:{type(exc).__name__}",
                    error_message=str(exc),
                )
            )
            continue

        smiles_set = {
            candidate.ligand.smiles for candidate in run.candidates
        }
        all_smiles.update(smiles_set)
        target_smiles_sets.append(smiles_set)
        per_target_top[target_gene] = [c.ligand.smiles for c in run.candidates[:5]]
        per_target.append(
            BatchTargetResult(
                target_gene=target_gene,
                status="completed",
                run=run,
                run_id=run.run_id,
                candidate_count=len(run.candidates),
            )
        )

    shared = _intersect_sets(target_smiles_sets) if target_smiles_sets else set()

    return BatchResult(
        per_target=per_target,
        total_candidates=sum(r.candidate_count for r in per_target),
        unique_smiles_count=len(all_smiles),
        shared_smiles_across_targets=sorted(shared),
        per_target_top_smiles=per_target_top,
    )


def _intersect_sets(sets: list[set[str]]) -> set[str]:
    if not sets:
        return set()
    result = set(sets[0])
    for subsequent in sets[1:]:
        result &= subsequent
    return result
