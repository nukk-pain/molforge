from __future__ import annotations

import asyncio
import importlib
import os
import sys
from pathlib import Path

from contracts.schema import AffinityPrediction, BindingPocket, Ligand


def repurpose_screen(
    pocket: BindingPocket,
    library: list[Ligand],
    *,
    top_n: int = 50,
    max_parallel: int | None = None,
    output_dir: Path | None = None,
) -> list[AffinityPrediction]:
    if not library:
        return []
    effective_parallel = max_parallel or max(1, (os.cpu_count() or 2) - 1)
    return asyncio.run(
        _repurpose_screen_async(
            pocket,
            library,
            top_n=top_n,
            max_parallel=effective_parallel,
            output_dir=output_dir,
        )
    )


async def _repurpose_screen_async(
    pocket: BindingPocket,
    library: list[Ligand],
    *,
    top_n: int,
    max_parallel: int,
    output_dir: Path | None,
) -> list[AffinityPrediction]:
    semaphore = asyncio.Semaphore(max_parallel)
    total = len(library)

    async def run_single(index: int, ligand: Ligand) -> AffinityPrediction | None:
        async with semaphore:
            print(
                f"[molforge-screen] docking {index + 1}/{total}: {ligand.chembl_id or ligand.smiles}",
                file=sys.stderr,
            )
            try:
                poses = await asyncio.to_thread(
                    run_dock,
                    pocket,
                    ligand.smiles,
                    output_dir,
                )
            except Exception as exc:
                print(
                    f"[molforge-screen] skipped {ligand.chembl_id or ligand.smiles}: {exc}",
                    file=sys.stderr,
                )
                return None
            if not poses:
                return None
            top_pose = poses[0]
            return AffinityPrediction(
                ligand_smiles=ligand.smiles,
                target_gene=pocket.structure.gene,
                vina_score=top_pose.vina_score,
                affinity_log_ki=None,
                affinity_confidence=None,
                pose_ref=top_pose,
            )

    predictions = await asyncio.gather(
        *(run_single(index, ligand) for index, ligand in enumerate(library))
    )
    ranked = [prediction for prediction in predictions if prediction is not None]
    ranked.sort(key=lambda prediction: prediction.vina_score)
    return ranked[:top_n]


def run_dock(
    pocket: BindingPocket,
    ligand_smiles: str,
    output_dir: Path | None = None,
):
    vina_module = importlib.import_module("molforge.docking.vina")

    return vina_module.dock(pocket, ligand_smiles, output_dir=output_dir)
