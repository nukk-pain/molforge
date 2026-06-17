from __future__ import annotations

import warnings

from molforge.admet.ranker import normalize_vina, rank_candidates
from molforge.admet.vina_normalizer import (
    AbsoluteVinaNormalizer,
    AdaptiveVinaNormalizer,
    PercentileVinaNormalizer,
)
from contracts.schema import (
    ADMETProfile,
    AffinityPrediction,
    EvidenceItem,
    Ligand,
    OffTargetHit,
    TargetCandidate,
)


def build_target() -> TargetCandidate:
    return TargetCandidate(
        gene="CXCR4",
        score=0.9,
        disease="pain",
        evidence=[EvidenceItem(source="demo", description="e", confidence=0.9)],
        pathway=["chemokine"],
        extra=None,
    )


def build_profile(smiles: str, *, herg: float = 0.2, hia: float = 0.8) -> ADMETProfile:
    return ADMETProfile(smiles, {"herg": herg, "hia_hou": hia}, [])


def test_percentile_vina_normalizer_returns_monotonic_unit_interval() -> None:
    scores = [-9.5, -9.0, -8.5, -8.0, -7.5, -7.0, -6.5]

    normalized = PercentileVinaNormalizer().normalize(scores)

    assert min(normalized) == 0.0
    assert max(normalized) == 1.0
    assert normalized == sorted(normalized, reverse=True)


def test_adaptive_vina_normalizer_falls_back_to_absolute_for_small_n() -> None:
    scores = [-9.0, -8.0, -7.0]

    adaptive = AdaptiveVinaNormalizer(min_candidates=5).normalize(scores)
    absolute = AbsoluteVinaNormalizer().normalize(scores)

    assert adaptive == absolute


def test_percentile_vina_normalizer_assigns_equal_scores_the_same_percentile() -> None:
    scores = [-9.0, -9.0, -7.0, -7.0]

    normalized = PercentileVinaNormalizer().normalize(scores)

    assert normalized[0] == normalized[1]
    assert normalized[2] == normalized[3]
    assert normalized[0] > normalized[2]


def test_percentile_vina_normalizer_handles_single_candidate() -> None:
    """N=1 cannot compute a rank spread; must not divide by zero."""
    normalized = PercentileVinaNormalizer().normalize([-8.5])

    assert len(normalized) == 1
    assert 0.0 <= normalized[0] <= 1.0


def test_percentile_vina_normalizer_handles_all_equal_scores() -> None:
    """All-equal input degenerates to a single percentile; must not crash."""
    normalized = PercentileVinaNormalizer().normalize([-7.5, -7.5, -7.5, -7.5])

    assert len(normalized) == 4
    assert len(set(normalized)) == 1  # all identical
    assert 0.0 <= normalized[0] <= 1.0


def test_adaptive_vina_normalizer_empty_input_returns_empty_list() -> None:
    normalized = AdaptiveVinaNormalizer().normalize([])

    assert normalized == []


def test_rank_candidates_preserves_existing_small_fixture_order_by_default() -> None:
    target = build_target()
    ligands = [
        Ligand(smiles="CCO", source="user"),
        Ligand(smiles="CCN", source="user"),
    ]
    affinities = [
        AffinityPrediction("CCO", "CXCR4", -9.0, None, None, None),
        AffinityPrediction("CCN", "CXCR4", -5.0, None, None, None),
    ]
    profiles = [
        build_profile("CCO", herg=0.1, hia=0.8),
        build_profile("CCN", herg=0.9, hia=0.3),
    ]
    off_targets = {
        "CCO": [],
        "CCN": [OffTargetHit("CCN", "KCNH2", 0.8, "high")],
    }

    ranked = rank_candidates(target, ligands, affinities, profiles, off_targets)

    assert [candidate.ligand.smiles for candidate in ranked] == ["CCO", "CCN"]
    assert [candidate.rank for candidate in ranked] == [1, 2]


def test_rank_candidates_accepts_percentile_normalizer_for_reordered_batch() -> None:
    target = build_target()
    ligands = [
        Ligand(smiles="L1", source="user"),
        Ligand(smiles="L2", source="user"),
        Ligand(smiles="L3", source="user"),
        Ligand(smiles="L4", source="user"),
        Ligand(smiles="L5", source="user"),
    ]
    affinities = [
        AffinityPrediction("L1", "CXCR4", -9.5, None, None, None),
        AffinityPrediction("L2", "CXCR4", -9.0, None, None, None),
        AffinityPrediction("L3", "CXCR4", -8.5, None, None, None),
        AffinityPrediction("L4", "CXCR4", -8.0, None, None, None),
        AffinityPrediction("L5", "CXCR4", -7.5, None, None, None),
    ]
    profiles = [build_profile(ligand.smiles) for ligand in ligands]
    off_targets = {ligand.smiles: [] for ligand in ligands}

    ranked = rank_candidates(
        target,
        ligands,
        affinities,
        profiles,
        off_targets,
        vina_normalizer=PercentileVinaNormalizer(),
    )

    assert [candidate.ligand.smiles for candidate in ranked] == [
        "L1",
        "L2",
        "L3",
        "L4",
        "L5",
    ]
    assert ranked[0].composite_score > ranked[-1].composite_score


def test_rank_candidates_uses_adaptive_percentile_default_at_five_candidates() -> None:
    target = build_target()
    ligands = [Ligand(smiles=f"L{index}", source="user") for index in range(1, 6)]
    affinities = [
        AffinityPrediction("L1", "CXCR4", -9.50, None, None, None),
        AffinityPrediction("L2", "CXCR4", -9.49, None, None, None),
        AffinityPrediction("L3", "CXCR4", -8.00, None, None, None),
        AffinityPrediction("L4", "CXCR4", -7.90, None, None, None),
        AffinityPrediction("L5", "CXCR4", -7.80, None, None, None),
    ]
    profiles = [
        build_profile("L1", herg=0.3, hia=0.7),
        build_profile("L2", herg=0.2, hia=0.7),
        build_profile("L3", herg=0.4, hia=0.6),
        build_profile("L4", herg=0.4, hia=0.6),
        build_profile("L5", herg=0.4, hia=0.6),
    ]
    off_targets = {ligand.smiles: [] for ligand in ligands}

    adaptive_ranked = rank_candidates(
        target, ligands, affinities, profiles, off_targets
    )
    absolute_ranked = rank_candidates(
        target,
        ligands,
        affinities,
        profiles,
        off_targets,
        vina_normalizer=AbsoluteVinaNormalizer(),
    )

    assert adaptive_ranked[0].ligand.smiles == "L1"
    assert absolute_ranked[0].ligand.smiles == "L2"


def test_normalize_vina_emits_deprecation_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = normalize_vina(-8.0)

    assert value == AbsoluteVinaNormalizer().normalize_score(-8.0)
    assert any(item.category is DeprecationWarning for item in caught)
