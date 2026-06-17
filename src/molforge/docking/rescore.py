from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterable

from contracts.schema import AffinityPrediction

from molforge.remote import RemoteGPUBackend

from .boltz2 import build_boltz2_job_spec, parse_boltz2_job_result


def rescore_predictions(
    predictions: list[AffinityPrediction],
    *,
    backend: RemoteGPUBackend,
    batch_size: int = 10,
) -> tuple[list[AffinityPrediction], float]:
    if not predictions:
        return [], 0.0

    grouped_predictions = _group_predictions_by_protein(predictions)
    rescored_predictions: list[AffinityPrediction] = []
    total_cost_estimate_usd = 0.0
    for protein_pdb_path, protein_predictions in grouped_predictions:
        for batch in _batched(protein_predictions, batch_size=batch_size):
            ligands = [prediction.ligand_smiles for prediction in batch]
            handle = backend.submit(
                build_boltz2_job_spec(
                    protein_pdb_path=protein_pdb_path,
                    ligands=ligands,
                )
            )
            result = backend.fetch_result(handle)
            if not result.success:
                raise RuntimeError(
                    f"Boltz-2 rescoring failed for {protein_pdb_path}: {result.stderr or result.stdout}"
                )
            score_map = parse_boltz2_job_result(result)
            rescored_predictions.extend(_merge_scores(batch, score_map=score_map))
            total_cost_estimate_usd += result.cost_estimate_usd
    return rescored_predictions, round(total_cost_estimate_usd, 6)


def _group_predictions_by_protein(
    predictions: Iterable[AffinityPrediction],
) -> list[tuple[Path, list[AffinityPrediction]]]:
    grouped: dict[Path, list[AffinityPrediction]] = {}
    for prediction in predictions:
        if prediction.pose_ref is None:
            raise ValueError(
                f"Cannot rescore '{prediction.ligand_smiles}' without a docking pose reference."
            )
        protein_pdb_path = Path(prediction.pose_ref.pocket.structure.pdb_path)
        grouped.setdefault(protein_pdb_path, []).append(prediction)
    return list(grouped.items())


def _batched(
    predictions: list[AffinityPrediction], *, batch_size: int
) -> list[list[AffinityPrediction]]:
    return [
        predictions[index : index + batch_size]
        for index in range(0, len(predictions), batch_size)
    ]


def _merge_scores(
    predictions: list[AffinityPrediction],
    *,
    score_map: dict[str, tuple[float, float]],
) -> list[AffinityPrediction]:
    rescored: list[AffinityPrediction] = []
    for prediction in predictions:
        if prediction.ligand_smiles not in score_map:
            raise ValueError(
                f"Boltz-2 output did not contain ligand '{prediction.ligand_smiles}'."
            )
        affinity_log_ki, affinity_confidence = score_map[prediction.ligand_smiles]
        rescored.append(
            replace(
                prediction,
                affinity_log_ki=affinity_log_ki,
                affinity_confidence=affinity_confidence,
            )
        )
    return rescored
