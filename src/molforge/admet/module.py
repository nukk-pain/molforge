from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from contracts.schema import (
    ADMETProfile,
    AffinityPrediction,
    GeneratedMolecule,
    Ligand,
    RankedCandidate,
    TargetCandidate,
)

from .off_target import OffTargetAssessment, assess_off_targets
from .ranker import rank_candidates
from .scorer import ADMETScorer


class ADMETStore(Protocol):
    def save_admet_profile(self, run_id: str, admet_profile: ADMETProfile) -> int: ...
    def save_ranking(self, run_id: str, ranking: RankedCandidate) -> int: ...


@dataclass(frozen=True, slots=True)
class ADMETEvaluationResult:
    profiles: list[ADMETProfile]
    off_target_assessments: dict[str, OffTargetAssessment]
    ranked_candidates: list[RankedCandidate]
    metadata: dict[str, Any]


class MolforgeADMETModule:
    def __init__(self, scorer: ADMETScorer | None = None) -> None:
        self.scorer = scorer or ADMETScorer()

    def run(
        self,
        molecules: Sequence[Ligand | GeneratedMolecule],
        *,
        affinity_map: dict[str, AffinityPrediction],
        store: ADMETStore,
        run_id: str,
        target: TargetCandidate,
        top_n: int = 10,
        enable_live_chembl: bool = False,
        enable_evebio: bool = False,
        provenance: dict[str, Any] | None = None,
    ) -> list[RankedCandidate]:
        ligands = [coerce_ligand(molecule) for molecule in molecules]
        evaluation = run_admet_phase(
            target=target,
            ligands=ligands,
            affinities=[
                affinity_map[ligand.smiles]
                for ligand in ligands
                if ligand.smiles in affinity_map
            ],
            scorer=self.scorer,
            enable_live_chembl=enable_live_chembl,
            enable_evebio=enable_evebio,
            provenance=provenance,
        )
        for profile in evaluation.profiles:
            _ = store.save_admet_profile(run_id, profile)
        top_ranked = evaluation.ranked_candidates[:top_n]
        for candidate in top_ranked:
            _ = store.save_ranking(run_id, candidate)
        return top_ranked


def run_admet_phase(
    *,
    target: TargetCandidate,
    ligands: list[Ligand],
    affinities: list[AffinityPrediction],
    scorer: ADMETScorer | None = None,
    enable_live_chembl: bool = False,
    enable_evebio: bool = False,
    provenance: dict[str, Any] | None = None,
) -> ADMETEvaluationResult:
    active_scorer = scorer or ADMETScorer()
    profiles = active_scorer.score_batch(ligands)
    assessments: dict[str, OffTargetAssessment] = {}
    for ligand, profile in zip(ligands, profiles, strict=True):
        assessment = assess_off_targets(
            ligand,
            enable_live_chembl=enable_live_chembl,
            enable_evebio=enable_evebio,
            target_gene=target.gene,
            target_uniprot_id=target.uniprot_id,
        )
        profile.liability_flags = sorted(
            set(profile.liability_flags) | set(assessment.class_effect_flags)
        )
        assessments[ligand.smiles] = assessment

    ranked_candidates = rank_candidates(
        target=target,
        ligands=ligands,
        affinities=affinities,
        admet_profiles=profiles,
        off_target_hits_by_smiles={
            smiles: assessment.hits for smiles, assessment in assessments.items()
        },
        provenance={
            "stage": "phase4_admet",
            "live_chembl_enabled": enable_live_chembl,
            "evebio_enabled": enable_evebio,
            "evebio_assessment_metadata": {
                smiles: assessment.metadata
                for smiles, assessment in assessments.items()
                if assessment.metadata.get("evebio_lookup_attempted")
            },
            **(provenance or {}),
        },
    )
    return ADMETEvaluationResult(
        profiles=profiles,
        off_target_assessments=assessments,
        ranked_candidates=ranked_candidates,
        metadata={
            "ligand_count": len(ligands),
            "ranked_count": len(ranked_candidates),
            "evebio_enabled": enable_evebio,
        },
    )


def coerce_ligand(molecule: Ligand | GeneratedMolecule) -> Ligand:
    if isinstance(molecule, Ligand):
        return molecule
    return Ligand(smiles=molecule.smiles, source=f"generative:{molecule.backend}")
