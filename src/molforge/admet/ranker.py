from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

from contracts.schema import (
    ADMETProfile,
    AffinityPrediction,
    Ligand,
    OffTargetHit,
    RankedCandidate,
    TargetCandidate,
)

from ._endpoints import canonicalize_endpoint_name, get_ranking_weights
from ._endpoint_direction import ENDPOINT_DIRECTION, normalize_endpoint_score
from .liability import excluded_endpoints_for_flags, liability_penalty
from .target_relevance import TargetRelevanceEvidence, assess_target_relevance
from .vina_normalizer import (
    AdaptiveVinaNormalizer,
    AbsoluteVinaNormalizer,
    VinaNormalizer,
)


@dataclass(frozen=True, slots=True)
class RankingWeights:
    vina: float
    admet: float
    off_target: float


def default_ranking_weights() -> RankingWeights:
    config = get_ranking_weights()
    weights = RankingWeights(
        vina=float(config["vina"]),
        admet=float(config["admet"]),
        off_target=float(config["off_target"]),
    )
    total = weights.vina + weights.admet + weights.off_target
    if abs(total - 1.0) > 1e-9:
        raise ValueError("Phase 4 ranking weights must sum to 1.0.")
    return weights


def rank_candidates(
    target: TargetCandidate,
    ligands: list[Ligand],
    affinities: list[AffinityPrediction],
    admet_profiles: list[ADMETProfile],
    off_target_hits_by_smiles: dict[str, list[OffTargetHit]],
    *,
    provenance: dict[str, Any] | None = None,
    weights: RankingWeights | None = None,
    vina_normalizer: VinaNormalizer | None = None,
    target_relevance_evidence_by_smiles: dict[str, TargetRelevanceEvidence] | None = None,
) -> list[RankedCandidate]:
    chosen_weights = weights or default_ranking_weights()
    chosen_vina_normalizer = vina_normalizer or AdaptiveVinaNormalizer()
    affinity_by_smiles = {affinity.ligand_smiles: affinity for affinity in affinities}
    profile_by_smiles = {profile.ligand_smiles: profile for profile in admet_profiles}
    normalized_vina_by_smiles = _normalize_affinity_scores(
        affinities,
        vina_normalizer=chosen_vina_normalizer,
    )

    ranked: list[RankedCandidate] = []
    for ligand in ligands:
        affinity = affinity_by_smiles.get(ligand.smiles)
        profile = profile_by_smiles.get(ligand.smiles)
        if affinity is None or profile is None:
            raise ValueError(
                f"Ranking requires both affinity and ADMET profile for ligand '{ligand.smiles}'."
            )

        off_targets = off_target_hits_by_smiles.get(ligand.smiles, [])
        composite_score = composite_score_for_candidate(
            normalized_vina_score=normalized_vina_by_smiles[ligand.smiles],
            target=target,
            ligand=ligand,
            profile=profile,
            off_targets=off_targets,
            weights=chosen_weights,
            target_relevance_evidence=(target_relevance_evidence_by_smiles or {}).get(
                ligand.smiles
            ),
        )
        target_relevance = assess_target_relevance(
            target=target,
            ligand=ligand,
            profile=profile,
            evidence=(target_relevance_evidence_by_smiles or {}).get(ligand.smiles),
        )
        candidate_provenance = dict(provenance or {})
        if target_relevance.flags:
            candidate_provenance["target_relevance_flags"] = list(
                target_relevance.flags
            )
            candidate_provenance["target_relevance_penalty"] = round(
                target_relevance.penalty,
                6,
            )
        ranked.append(
            RankedCandidate(
                ligand=ligand,
                target=target,
                affinity=affinity,
                admet=profile,
                off_targets=off_targets,
                composite_score=round(composite_score, 6),
                rank=0,
                provenance=candidate_provenance,
            )
        )

    ranked.sort(key=lambda candidate: candidate.composite_score, reverse=True)
    for index, candidate in enumerate(ranked, start=1):
        candidate.rank = index
    return ranked


def composite_score_for_candidate(
    *,
    normalized_vina_score: float,
    target: TargetCandidate | None = None,
    ligand: Ligand | None = None,
    profile: ADMETProfile,
    off_targets: list[OffTargetHit],
    weights: RankingWeights,
    target_relevance_evidence: TargetRelevanceEvidence | None = None,
) -> float:
    score = (
        weights.vina * normalized_vina_score
        + weights.admet * normalize_admet(profile)
        + weights.off_target * normalize_off_target(off_targets)
    )
    if target is not None and ligand is not None:
        score -= assess_target_relevance(
            target=target,
            ligand=ligand,
            profile=profile,
            evidence=target_relevance_evidence,
        ).penalty
    return max(0.0, min(1.0, score))


def normalize_vina(vina_score: float) -> float:
    warnings.warn(
        "normalize_vina() is deprecated; use AbsoluteVinaNormalizer or AdaptiveVinaNormalizer.",
        DeprecationWarning,
        stacklevel=2,
    )
    return AbsoluteVinaNormalizer().normalize_score(vina_score)


def _normalize_affinity_scores(
    affinities: list[AffinityPrediction],
    *,
    vina_normalizer: VinaNormalizer,
) -> dict[str, float]:
    normalized_scores = vina_normalizer.normalize(
        [affinity.vina_score for affinity in affinities]
    )
    return {
        affinity.ligand_smiles: normalized_score
        for affinity, normalized_score in zip(
            affinities,
            normalized_scores,
            strict=True,
        )
    }


def normalize_admet(profile: ADMETProfile) -> float:
    if not profile.endpoints:
        return 0.0
    excluded = excluded_endpoints_for_flags(profile.liability_flags)
    canonical_endpoints = {
        canonicalize_endpoint_name(endpoint_name): value
        for endpoint_name, value in profile.endpoints.items()
    }
    desirable_values = [
        normalize_endpoint_score(endpoint_name, value)
        for endpoint_name, value in canonical_endpoints.items()
        if endpoint_name in ENDPOINT_DIRECTION and endpoint_name not in excluded
    ]
    if not desirable_values:
        return max(0.0, 1.0 - liability_penalty(profile.liability_flags))
    average = sum(desirable_values) / len(desirable_values)
    return max(0.0, average - liability_penalty(profile.liability_flags))


def normalize_off_target(hits: list[OffTargetHit]) -> float:
    if not hits:
        return 1.0
    severity_scores = {"high": 0.0, "medium": 0.5, "low": 0.8}
    base_score = min(severity_scores[hit.severity] for hit in hits)
    return max(0.0, base_score - (0.1 * len(hits)))
