from __future__ import annotations

import asyncio
import hashlib
import importlib
from dataclasses import replace
from pathlib import Path

from contracts.schema import AffinityPrediction, Ligand, TargetCandidate

DEFAULT_AFDB_CACHE_DIR = Path("archive/cache/afdb")
DEFAULT_CHEMBL_CACHE_DIR = Path("archive/cache/chembl")


class DockingRunner:
    def __init__(
        self,
        *,
        afdb_cache_dir: Path = DEFAULT_AFDB_CACHE_DIR,
        chembl_cache_dir: Path = DEFAULT_CHEMBL_CACHE_DIR,
        use_fpocket: bool = True,
        max_parallel: int | None = None,
    ) -> None:
        self.afdb_cache_dir = afdb_cache_dir
        self.chembl_cache_dir = chembl_cache_dir
        self.use_fpocket = use_fpocket
        self.max_parallel = max_parallel

    def run(
        self,
        targets: list[TargetCandidate],
        *,
        store,
        top_n: int = 50,
    ) -> list[AffinityPrediction]:
        if not targets:
            return []

        env_module = importlib.import_module("molforge.docking._env_check")
        env_module.require_vina_binary()
        env_module.require_meeko_available()
        chembl_module = importlib.import_module("molforge.docking.chembl")
        library = chembl_module.load_fda_approved_library(
            cache_dir=self.chembl_cache_dir,
            max_n=max(100, top_n),
        )
        predictions: list[AffinityPrediction] = []

        for target in targets:
            run_id = store.create_run(target, self._compute_config_hash(top_n=top_n))
            target_predictions = self.run_target(
                target,
                store=store,
                run_id=run_id,
                top_n=top_n,
                library=library,
            )
            store.complete_run(run_id)
            predictions.extend(target_predictions)

        predictions.sort(key=lambda prediction: prediction.vina_score)
        return predictions[:top_n]

    def run_target(
        self,
        target: TargetCandidate,
        *,
        store,
        run_id: str,
        top_n: int = 50,
        library: list[Ligand] | None = None,
    ) -> list[AffinityPrediction]:
        env_module = importlib.import_module("molforge.docking._env_check")
        env_module.require_vina_binary()
        env_module.require_meeko_available()
        active_library = library or self._load_library(top_n=top_n)
        return self._run_single_target(
            target,
            run_id=run_id,
            store=store,
            library=active_library,
            top_n=top_n,
        )

    def score_ligands(
        self,
        pocket,
        ligands: list[Ligand],
        *,
        store,
        run_id: str,
        top_n: int = 50,
    ) -> list[AffinityPrediction]:
        if not ligands:
            return []
        env_module = importlib.import_module("molforge.docking._env_check")
        env_module.require_vina_binary()
        env_module.require_meeko_available()
        screen_module = importlib.import_module("molforge.docking.screen")
        pose_output_dir = Path("archive/runs") / run_id / "generated-poses"
        predictions = screen_module.repurpose_screen(
            pocket,
            ligands,
            top_n=top_n,
            max_parallel=self.max_parallel,
            output_dir=pose_output_dir,
        )
        for prediction in predictions:
            if prediction.pose_ref is not None:
                _ = store.save_pose(run_id, prediction.pose_ref)
        return predictions

    def _run_single_target(
        self,
        target: TargetCandidate,
        *,
        run_id: str,
        store,
        library,
        top_n: int,
    ) -> list[AffinityPrediction]:
        structure_module = importlib.import_module("molforge.docking.structure")
        gene_mapping_module = importlib.import_module("molforge.docking.gene_mapping")
        pocket_module = importlib.import_module("molforge.docking.pocket")
        screen_module = importlib.import_module("molforge.docking.screen")

        uniprot = gene_mapping_module.resolve_uniprot(target)
        if not uniprot:
            raise structure_module.MissingStructureError(target.gene, "unknown")

        structure = asyncio.run(
            structure_module.fetch_alphafold_structure(
                uniprot, cache_dir=self.afdb_cache_dir
            )
        )
        structure = replace(structure, gene=target.gene)
        _ = store.save_structure(run_id, structure)

        pocket = pocket_module.detect_pocket(structure, use_fpocket=self.use_fpocket)
        pose_output_dir = Path("archive/runs") / run_id / "poses"
        predictions = screen_module.repurpose_screen(
            pocket,
            library,
            top_n=top_n,
            max_parallel=self.max_parallel,
            output_dir=pose_output_dir,
        )
        for prediction in predictions:
            if prediction.pose_ref is not None:
                _ = store.save_pose(run_id, prediction.pose_ref)
        return predictions

    def _load_library(self, *, top_n: int) -> list[Ligand]:
        chembl_module = importlib.import_module("molforge.docking.chembl")
        return chembl_module.load_fda_approved_library(
            cache_dir=self.chembl_cache_dir,
            max_n=max(100, top_n),
        )

    def _compute_config_hash(self, *, top_n: int) -> str:
        payload = (
            f"top_n={top_n}\n"
            f"use_fpocket={self.use_fpocket}\n"
            f"max_parallel={self.max_parallel}\n"
            f"afdb_cache_dir={self.afdb_cache_dir}\n"
            f"chembl_cache_dir={self.chembl_cache_dir}\n"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
