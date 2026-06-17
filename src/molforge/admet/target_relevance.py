from __future__ import annotations

from dataclasses import dataclass

from contracts.schema import ADMETProfile, Ligand, TargetCandidate


@dataclass(frozen=True, slots=True)
class TargetRelevanceEvidence:
    """Optional, injectable evidence for target-agnostic relevance scoring.

    v1 deliberately keeps this offline/testable: callers may pass cached ChEMBL
    or other public-database evidence without making ranking depend on network I/O.
    """

    on_target_activity_score: float | None = None
    known_primary_targets: tuple[str, ...] = ()
    max_phase: int | None = None
    indication_terms: tuple[str, ...] = ()
    safety_warnings: tuple[str, ...] = ()
    evidence_source: str = "injected"


@dataclass(frozen=True, slots=True)
class TargetRelevanceAdjustment:
    penalty: float
    flags: tuple[str, ...] = ()


def assess_target_relevance(
    *,
    target: TargetCandidate,
    ligand: Ligand,
    profile: ADMETProfile,
    evidence: TargetRelevanceEvidence | None = None,
) -> TargetRelevanceAdjustment:
    flags: list[str] = []
    penalty = 0.0

    if _has_on_target_activity(evidence):
        flags.append("target_match:on_target_activity")
    elif _is_unvalidated_repurposing_hit(ligand):
        flags.append("target_mismatch:unvalidated_repurposing_hit")
        penalty += 0.15

    if _has_known_primary_target_mismatch(target, evidence):
        flags.append("target_mismatch:known_pharmacology")
        penalty += _known_pharmacology_penalty(evidence)

    if _has_indication_mismatch(target, evidence):
        flags.append("target_mismatch:indication")
        penalty += 0.05

    if _has_high_safety_risk(profile.liability_flags, evidence):
        flags.append("target_mismatch:safety_margin")
        penalty += 0.05

    return TargetRelevanceAdjustment(penalty=min(0.45, penalty), flags=tuple(flags))


def _has_on_target_activity(evidence: TargetRelevanceEvidence | None) -> bool:
    if evidence is None or evidence.on_target_activity_score is None:
        return False
    return evidence.on_target_activity_score >= 0.5


def _is_unvalidated_repurposing_hit(ligand: Ligand) -> bool:
    return ligand.source == "chembl_fda"


def _has_known_primary_target_mismatch(
    target: TargetCandidate,
    evidence: TargetRelevanceEvidence | None,
) -> bool:
    if evidence is None or not evidence.known_primary_targets:
        return False
    query_terms = _target_terms(target)
    if not query_terms:
        return False
    known_terms = {_normalize_text(item) for item in evidence.known_primary_targets}
    return query_terms.isdisjoint(known_terms)


def _known_pharmacology_penalty(evidence: TargetRelevanceEvidence | None) -> float:
    max_phase = evidence.max_phase if evidence else None
    if max_phase is None:
        return 0.2
    if max_phase >= 4:
        return 0.3
    if max_phase >= 3:
        return 0.25
    if max_phase >= 1:
        return 0.15
    return 0.1


def _has_indication_mismatch(
    target: TargetCandidate,
    evidence: TargetRelevanceEvidence | None,
) -> bool:
    if evidence is None or not target.disease or not evidence.indication_terms:
        return False
    disease_terms = _split_terms(target.disease)
    indication_terms = set().union(
        *(_split_terms(indication) for indication in evidence.indication_terms)
    )
    return bool(disease_terms) and disease_terms.isdisjoint(indication_terms)


def _has_high_safety_risk(
    liability_flags: list[str],
    evidence: TargetRelevanceEvidence | None,
) -> bool:
    high_risk_flags = {
        "hepatotox",
        "hERG_high",
        "mutagenic",
        "rule:herg_risk_logp_mw",
        "rule:logp_over_5",
    }
    if high_risk_flags.intersection(liability_flags):
        return True
    if evidence is None:
        return False
    warning_text = " ".join(evidence.safety_warnings).lower()
    return any(term in warning_text for term in ("black box", "hepat", "cardio", "qt"))


def _target_terms(target: TargetCandidate) -> set[str]:
    terms = {_normalize_text(target.gene)}
    if target.uniprot_id:
        terms.add(_normalize_text(target.uniprot_id))
    return {term for term in terms if term}


def _split_terms(text: str) -> set[str]:
    return {
        token
        for token in (_normalize_text(part) for part in text.replace("/", " ").split())
        if len(token) >= 3
    }


def _normalize_text(text: str) -> str:
    return "".join(character.lower() for character in text if character.isalnum())
