from __future__ import annotations

import hashlib
import importlib
import json
import logging
import subprocess
from pathlib import Path
from typing import Protocol, Sequence

_logger = logging.getLogger(__name__)

_DOCKING_RECOVERABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    RuntimeError,
    OSError,  # covers FileNotFoundError (vina/meeko/fpocket missing) and I/O
    TimeoutError,
    subprocess.CalledProcessError,
)

from contracts.schema import (
    ADMETProfile,
    AffinityPrediction,
    BindingPocket,
    GeneratedMolecule,
    GenerativeBackend,
    Ligand,
    PipelineRun,
    ProteinStructure,
    RankedCandidate,
    StructureSource,
    TargetCandidate,
)


class DockingModule(Protocol):
    def run_target(
        self,
        target: TargetCandidate,
        *,
        store,
        run_id: str,
        top_n: int = 50,
    ) -> list[AffinityPrediction]: ...

    def score_ligands(
        self,
        pocket: BindingPocket,
        ligands: list[Ligand],
        *,
        store,
        run_id: str,
        top_n: int = 50,
    ) -> list[AffinityPrediction]: ...


class GenerativeModule(Protocol):
    def run(
        self,
        pockets: list[BindingPocket],
        *,
        store,
        run_id: str,
        n_per_pocket: int = 100,
        seed_smiles: str | None = None,
    ) -> list[GeneratedMolecule]: ...


class ADMETModule(Protocol):
    def run(
        self,
        molecules: Sequence[Ligand | GeneratedMolecule],
        *,
        affinity_map: dict[str, AffinityPrediction],
        store,
        run_id: str,
        target: TargetCandidate,
        top_n: int = 10,
        enable_live_chembl: bool = False,
        enable_evebio: bool = False,
        provenance: dict[str, object] | None = None,
    ) -> list[RankedCandidate]: ...


class StoreProtocol(Protocol):
    def create_run(self, input_target: TargetCandidate, config_hash: str) -> str: ...
    def load_run(self, run_id: str) -> PipelineRun: ...
    def save_molecule(
        self, run_id: str, molecule: Ligand | GeneratedMolecule
    ) -> int: ...
    def save_admet_profile(self, run_id: str, admet_profile: ADMETProfile) -> int: ...
    def save_ranking(self, run_id: str, ranking: RankedCandidate) -> int: ...
    def complete_run(self, run_id: str, completed_at: str | None = None) -> None: ...


def run_pipeline(
    targets: list[TargetCandidate],
    *,
    store: StoreProtocol,
    top_n: int = 10,
    enable_live_chembl: bool = False,
    enable_evebio: bool = False,
    docking_runner: DockingModule | None = None,
    generative_module: GenerativeModule | None = None,
    admet_module: ADMETModule | None = None,
) -> PipelineRun:
    if not targets:
        raise ValueError("run_pipeline requires at least one TargetCandidate.")
    if len(targets) != 1:
        raise ValueError(
            "molforge run currently supports exactly one TargetCandidate per pipeline run."
        )

    input_target = targets[0]
    config_hash = compute_config_hash(top_n=top_n)
    run_id = store.create_run(input_target, config_hash)
    active_docking_runner = docking_runner or _build_default_docking_runner()
    active_admet_module = admet_module or _build_default_admet_module()

    docking_status = "completed"
    try:
        docking_predictions = active_docking_runner.run_target(
            input_target,
            store=store,
            run_id=run_id,
            top_n=top_n,
        )
    except _DOCKING_RECOVERABLE_EXCEPTIONS as exc:
        # Swallow only the known Vina/meeko/ChEMBL failure surface so the
        # pipeline can persist diagnostics. Programmer bugs (AttributeError,
        # TypeError, ImportError) still crash loudly.
        _logger.exception("docking stage failed; continuing with empty predictions")
        docking_predictions = []
        docking_status = f"failed:{type(exc).__name__}:{exc}"
    for ligand in _coerce_rankable_ligands(docking_predictions):
        _ = store.save_molecule(run_id, ligand)

    generated_molecules, generative_status = _run_generative_stage(
        target=input_target,
        run_id=run_id,
        store=store,
        docking_predictions=docking_predictions,
        generative_module=generative_module,
        top_n=top_n,
    )

    generated_predictions = _redock_generated_molecules(
        docking_runner=active_docking_runner,
        docking_predictions=docking_predictions,
        generated_molecules=generated_molecules,
        run_id=run_id,
        store=store,
        top_n=top_n,
    )

    ligands = _merge_rankable_ligands(docking_predictions, generated_predictions)
    affinity_map = {
        prediction.ligand_smiles: prediction
        for prediction in [*docking_predictions, *generated_predictions]
    }
    stage_provenance: dict[str, object] = {
        "run_id": run_id,
        "stage": "phase5_pipeline",
        "docking_status": docking_status,
        "docking_candidate_count": len(docking_predictions),
        "generative_status": generative_status,
        "generated_candidate_count": len(generated_molecules),
        "generated_redocked_count": len(generated_predictions),
        "live_chembl_enabled": enable_live_chembl,
        "evebio_enabled": enable_evebio,
        "admet_status": "skipped:empty_affinity_map"
        if not affinity_map
        else "completed",
    }
    if affinity_map:
        _ = active_admet_module.run(
            ligands,
            affinity_map=affinity_map,
            store=store,
            run_id=run_id,
            target=input_target,
            top_n=top_n,
            enable_live_chembl=enable_live_chembl,
            enable_evebio=enable_evebio,
            provenance=stage_provenance,
        )
    else:
        _persist_diagnostic_provenance(run_id, stage_provenance)

    store.complete_run(run_id)
    return store.load_run(run_id)


def _persist_diagnostic_provenance(run_id: str, provenance: dict[str, object]) -> None:
    """Write stage-by-stage diagnostic JSON when the admet stage is skipped.

    Ensures phase5 artifacts surface *why* candidates=0 without needing to
    replay the run. Written to archive/runs/<run_id>/diagnostics.json.
    """
    diag_dir = Path("archive/runs") / run_id
    diag_dir.mkdir(parents=True, exist_ok=True)
    (diag_dir / "diagnostics.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def compute_config_hash(*, top_n: int) -> str:
    pyproject_path = Path("pyproject.toml")
    pyproject_text = (
        pyproject_path.read_text(encoding="utf-8") if pyproject_path.exists() else ""
    )
    payload = f"top_n={top_n}\n{pyproject_text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def structure_fetch_stub(target: TargetCandidate) -> list[object]:
    _ = target
    return []


def docking_stub(target: TargetCandidate) -> list[object]:
    _ = target
    return []


def generative_stub(target: TargetCandidate) -> list[object]:
    _ = target
    return []


def admet_stub(target: TargetCandidate) -> list[object]:
    _ = target
    return []


def ranking_stub(target: TargetCandidate) -> list[RankedCandidate]:
    _ = target
    return []


def _build_default_docking_runner():
    from ..docking.module import DockingRunner

    return DockingRunner()


def _build_default_admet_module():
    from ..admet.module import MolforgeADMETModule

    return MolforgeADMETModule()


def _build_default_generative_module(output_dir: Path):
    generative_module = importlib.import_module("molforge.generative.module")
    return generative_module.MolforgeGenerativeModule(
        generative_module.build_default_backend(output_dir=output_dir)
    )


def _run_generative_stage(
    *,
    target: TargetCandidate,
    run_id: str,
    store: StoreProtocol,
    docking_predictions: list[AffinityPrediction],
    generative_module: GenerativeModule | None,
    top_n: int,
) -> tuple[list[GeneratedMolecule], str]:
    pockets = _collect_unique_pockets(docking_predictions)
    if not pockets:
        return [], "skipped:no_binding_pocket"
    active_generative_module = generative_module
    if active_generative_module is None:
        output_dir = Path("archive/runs") / run_id / "generative"
        active_generative_module = _build_default_generative_module(output_dir)
    try:
        molecules = active_generative_module.run(
            pockets,
            store=store,
            run_id=run_id,
            n_per_pocket=top_n,
        )
    except RuntimeError as exc:
        return [], f"skipped:{exc}"
    return molecules, "completed"


def _collect_unique_pockets(
    predictions: Sequence[AffinityPrediction],
) -> list[BindingPocket]:
    pockets: list[BindingPocket] = []
    seen: set[str] = set()
    for prediction in predictions:
        pocket = prediction.pose_ref.pocket if prediction.pose_ref is not None else None
        if pocket is None:
            continue
        key = json.dumps(
            {
                "gene": pocket.structure.gene,
                "pdb_path": pocket.structure.pdb_path,
                "center_xyz": list(pocket.center_xyz),
                "size_xyz": list(pocket.size_xyz),
            },
            sort_keys=True,
        )
        if key in seen:
            continue
        seen.add(key)
        pockets.append(pocket)
    return pockets


def _redock_generated_molecules(
    *,
    docking_runner: DockingModule,
    docking_predictions: list[AffinityPrediction],
    generated_molecules: Sequence[GeneratedMolecule],
    run_id: str,
    store: StoreProtocol,
    top_n: int,
) -> list[AffinityPrediction]:
    pockets = _collect_unique_pockets(docking_predictions)
    if not pockets or not generated_molecules:
        return []
    ligands: list[Ligand] = []
    seen: set[str] = set()
    for molecule in generated_molecules:
        if molecule.smiles in seen:
            continue
        seen.add(molecule.smiles)
        ligands.append(
            Ligand(smiles=molecule.smiles, source=f"generative:{molecule.backend}")
        )
    return docking_runner.score_ligands(
        pockets[0],
        ligands,
        store=store,
        run_id=run_id,
        top_n=top_n,
    )


def _coerce_rankable_ligands(predictions: Sequence[AffinityPrediction]) -> list[Ligand]:
    return [
        Ligand(smiles=prediction.ligand_smiles, source="chembl_fda")
        for prediction in predictions
    ]


def _merge_rankable_ligands(
    docking_predictions: Sequence[AffinityPrediction],
    generated_predictions: Sequence[AffinityPrediction],
) -> list[Ligand]:
    ligands: list[Ligand] = []
    seen: set[str] = set()
    for ligand in [
        *_coerce_rankable_ligands(docking_predictions),
        *[
            Ligand(smiles=prediction.ligand_smiles, source="generative:reinvent4")
            for prediction in generated_predictions
        ],
    ]:
        if ligand.smiles in seen:
            continue
        seen.add(ligand.smiles)
        ligands.append(ligand)
    return ligands


def _xyz_triplet(values: object, *, field_name: str) -> tuple[float, float, float]:
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError(f"{field_name} must be a three-element list.")
    return (float(values[0]), float(values[1]), float(values[2]))


def load_binding_pocket(path: str | Path) -> BindingPocket:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("BindingPocket JSON must decode to an object.")
    structure_payload = payload.get("structure")
    if not isinstance(structure_payload, dict):
        raise ValueError("BindingPocket JSON must include a structure object.")
    structure = ProteinStructure(
        gene=str(structure_payload["gene"]),
        uniprot=None
        if structure_payload.get("uniprot") is None
        else str(structure_payload["uniprot"]),
        pdb_path=str(structure_payload["pdb_path"]),
        source=StructureSource(str(structure_payload["source"])),
        confidence=None
        if structure_payload.get("confidence") is None
        else float(structure_payload["confidence"]),
    )
    return BindingPocket(
        structure=structure,
        center_xyz=_xyz_triplet(payload.get("center_xyz"), field_name="center_xyz"),
        size_xyz=_xyz_triplet(payload.get("size_xyz"), field_name="size_xyz"),
        druggability_score=None
        if payload.get("druggability_score") is None
        else float(payload["druggability_score"]),
        residues=[str(item) for item in payload.get("residues", []) or []],
    )


def run_generate_stage(
    pocket: BindingPocket,
    *,
    output_dir: str | Path,
    n: int = 100,
    seed_smiles: str | None = None,
    backend: GenerativeBackend | None = None,
):
    generate_molecules = importlib.import_module(
        "molforge.generative.module"
    ).generate_molecules

    return generate_molecules(
        pocket,
        output_dir=output_dir,
        n=n,
        seed_smiles=seed_smiles,
        backend=backend,
    )


def run_admet_pipeline(
    *,
    target: TargetCandidate,
    ligands: Sequence[Ligand | GeneratedMolecule],
    affinities: list[AffinityPrediction],
    store,
    top_n: int = 10,
    scorer=None,
    enable_live_chembl: bool = False,
    enable_evebio: bool = False,
) -> PipelineRun:
    from ..admet.module import MolforgeADMETModule

    config_hash = compute_config_hash(top_n=top_n)
    run_id = store.create_run(target, config_hash)
    module = MolforgeADMETModule(scorer=scorer)
    _ = module.run(
        list(ligands),
        affinity_map={affinity.ligand_smiles: affinity for affinity in affinities},
        store=store,
        run_id=run_id,
        target=target,
        top_n=top_n,
        enable_live_chembl=enable_live_chembl,
        enable_evebio=enable_evebio,
    )
    store.complete_run(run_id)
    return store.load_run(run_id)
