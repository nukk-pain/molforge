from __future__ import annotations

import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from contracts.schema import (
    ADMETProfile,
    AffinityPrediction,
    EvidenceItem,
    Ligand,
    OffTargetHit,
    TargetCandidate,
)

ranker_module = importlib.import_module("molforge.admet.ranker")
rank_candidates = ranker_module.rank_candidates
TargetRelevanceEvidence = importlib.import_module(
    "molforge.admet.target_relevance"
).TargetRelevanceEvidence
ENDPOINT_DIRECTION = importlib.import_module(
    "molforge.admet._endpoint_direction"
).ENDPOINT_DIRECTION


def test_ranker_orders_candidates_by_composite_score() -> None:
    target = TargetCandidate(
        gene="CXCR4",
        score=0.9,
        disease="pain",
        evidence=[EvidenceItem(source="demo", description="e", confidence=0.9)],
        pathway=["chemokine"],
        extra=None,
    )
    ligands = [
        Ligand(smiles="CCO", source="user"),
        Ligand(smiles="CCN", source="user"),
    ]
    affinities = [
        AffinityPrediction("CCO", "CXCR4", -9.0, None, None, None),
        AffinityPrediction("CCN", "CXCR4", -5.0, None, None, None),
    ]
    profiles = [
        ADMETProfile("CCO", {"herg": 0.1, "hia_hou": 0.8}, []),
        ADMETProfile("CCN", {"herg": 0.9, "hia_hou": 0.3}, ["hERG_high"]),
    ]
    off_targets = {
        "CCO": [],
        "CCN": [OffTargetHit("CCN", "KCNH2", 0.8, "high")],
    }

    ranked = rank_candidates(target, ligands, affinities, profiles, off_targets)

    assert len(ENDPOINT_DIRECTION) == 41
    assert [candidate.ligand.smiles for candidate in ranked] == ["CCO", "CCN"]
    assert [candidate.rank for candidate in ranked] == [1, 2]
    assert 0.0 <= ranked[0].composite_score <= 1.0
    assert ranked[0].composite_score > ranked[1].composite_score


def test_ranker_penalizes_generic_known_pharmacology_mismatch() -> None:
    target = TargetCandidate(
        gene="DGAT2",
        score=0.9,
        disease="Nonalcoholic steatohepatitis",
        evidence=[EvidenceItem(source="demo", description="e", confidence=0.9)],
        pathway=["triacylglycerol synthesis"],
        extra=None,
    )
    adapalene = (
        "COc1ccc(-c2ccc3cc(C(=O)O)ccc3c2)cc1"
        "C12CC3CC(CC(C3)C1)C2"
    )
    ervogastat = "CCOC1=C(N=CC=C1)OC2=CN=CC(=C2)C3=NC=C(C=N3)C(=O)NC4CCOC4"
    ligands = [
        Ligand(smiles=adapalene, source="chembl_fda", chembl_id="CHEMBL1265"),
        Ligand(smiles=ervogastat, source="chembl_fda"),
    ]
    affinities = [
        AffinityPrediction(adapalene, "DGAT2", -9.5, None, None, None),
        AffinityPrediction(ervogastat, "DGAT2", -8.0, None, None, None),
    ]
    profiles = [
        ADMETProfile(adapalene, {"herg": 0.1, "dili": 0.1, "hia_hou": 0.9}, []),
        ADMETProfile(ervogastat, {"herg": 0.1, "dili": 0.1, "hia_hou": 0.9}, []),
    ]
    off_targets = {ligand.smiles: [] for ligand in ligands}

    ranked = rank_candidates(
        target,
        ligands,
        affinities,
        profiles,
        off_targets,
        target_relevance_evidence_by_smiles={
            adapalene: TargetRelevanceEvidence(
                known_primary_targets=("RARB", "RARG"),
                max_phase=4,
                indication_terms=("acne",),
                evidence_source="test",
            ),
            ervogastat: TargetRelevanceEvidence(
                on_target_activity_score=0.9,
                known_primary_targets=("DGAT2",),
                max_phase=2,
                indication_terms=("Nonalcoholic steatohepatitis",),
                evidence_source="test",
            ),
        },
    )

    assert [candidate.ligand.smiles for candidate in ranked] == [ervogastat, adapalene]
    mismatched_candidate = ranked[1]
    assert mismatched_candidate.provenance["target_relevance_penalty"] == 0.45
    assert mismatched_candidate.provenance["target_relevance_flags"] == [
        "target_mismatch:unvalidated_repurposing_hit",
        "target_mismatch:known_pharmacology",
        "target_mismatch:indication",
    ]


def test_ranker_keeps_known_on_target_fda_ligand_unpenalized_for_any_gene() -> None:
    target = TargetCandidate(
        gene="CXCR4",
        score=0.9,
        disease="pain",
        evidence=[EvidenceItem(source="demo", description="e", confidence=0.9)],
        pathway=["chemokine"],
        extra=None,
    )
    ligand = Ligand(smiles="CCO", source="chembl_fda", chembl_id="CHEMBL-TEST")
    ranked = rank_candidates(
        target,
        [ligand],
        [AffinityPrediction("CCO", "CXCR4", -8.0, None, None, None)],
        [ADMETProfile("CCO", {"herg": 0.1, "hia_hou": 0.9}, [])],
        {"CCO": []},
        target_relevance_evidence_by_smiles={
            "CCO": TargetRelevanceEvidence(
                on_target_activity_score=0.8,
                known_primary_targets=("CXCR4",),
                max_phase=3,
                indication_terms=("pain",),
                evidence_source="test",
            )
        },
    )

    assert ranked[0].provenance["target_relevance_penalty"] == 0.0
    assert ranked[0].provenance["target_relevance_flags"] == [
        "target_match:on_target_activity"
    ]
