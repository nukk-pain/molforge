# pyright: reportMissingImports=false
"""ConsensusADMETScorer tests."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from contracts.schema import ADMETProfile, Ligand  # noqa: E402
from molforge.admet.consensus import (  # noqa: E402
    ConsensusADMETScorer,
    RuleBasedADMETScorer,
)
from molforge.admet.physchem_rules import (  # noqa: E402
    liability_flags_from_rules,
    score_physchem_rules,
)


# ---------------------------------------------------------------------------
# physchem_rules
# ---------------------------------------------------------------------------


def test_physchem_rules_scores_classic_drug_like_smiles():
    """Ibuprofen should pass Lipinski and Veber."""
    smiles = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"  # ibuprofen
    endpoints = score_physchem_rules(smiles)
    assert endpoints["lipinski"] == 1.0
    assert endpoints["veber"] == 1.0
    assert 0 <= endpoints["qed_rule_proxy"] <= 1.0


def test_physchem_rules_flags_large_lipophilic_herg_risk():
    """A synthetic heavy-logP compound should trigger hERG flag."""
    # Large highly lipophilic molecule — over MW 400 and logP > 3.7
    smiles = "CCCCCCCCCCCCCCCCCCCCc1ccc2ccccc2c1"  # C20 chain on naphthalene
    flags = liability_flags_from_rules(smiles)
    assert "rule:herg_risk_logp_mw" in flags


def test_physchem_rules_invalid_smiles_raises():
    import pytest

    with pytest.raises(ValueError):
        score_physchem_rules("not a smiles")


# ---------------------------------------------------------------------------
# RuleBasedADMETScorer — satisfies the same .score() contract as ADMETScorer
# ---------------------------------------------------------------------------


def test_rule_based_scorer_returns_admet_profile():
    scorer = RuleBasedADMETScorer()
    profile = scorer.score("CC(C)Cc1ccc(cc1)C(C)C(=O)O")
    assert isinstance(profile, ADMETProfile)
    assert "lipinski" in profile.endpoints
    assert "bbb_permeability_proxy" in profile.endpoints
    assert "herg_inhibition_proxy" in profile.endpoints


def test_rule_based_scorer_accepts_ligand_object():
    scorer = RuleBasedADMETScorer()
    lg = Ligand(smiles="CCO", source="test")
    profile = scorer.score(lg)
    assert profile.ligand_smiles == "CCO"


# ---------------------------------------------------------------------------
# ConsensusADMETScorer (rule-only, no neural — keeps test CI-fast)
# ---------------------------------------------------------------------------


def test_consensus_single_scorer_mean_equals_raw():
    """Consensus with only one scorer should report the raw value as mean
    and zero disagreement."""
    consensus = ConsensusADMETScorer(
        scorers={"physchem_rules": RuleBasedADMETScorer()},
    )
    result = consensus.score("CC(C)Cc1ccc(cc1)C(C)C(=O)O")
    assert "lipinski" in result.consensus
    c = result.consensus["lipinski"]
    assert c.scorer_values == {"physchem_rules": 1.0}
    assert c.mean_value == 1.0
    assert c.max_disagreement == 0.0
    assert c.disagreement_flagged is False


def test_consensus_two_scorers_flags_disagreement_over_threshold():
    """Fake neural scorer whose predictions contradict the rule-based
    scorer on the aligned endpoint (`herg`) triggers the consensus flag."""
    class _FakeNeuralScorer:
        """Returns always-confident 'safe' scores — forces disagreement
        with the rule-based hERG flag on a heavy lipophilic compound."""

        def score(self, ligand):
            smiles = ligand.smiles if hasattr(ligand, "smiles") else ligand
            return ADMETProfile(
                ligand_smiles=smiles,
                endpoints={"herg": 0.05, "bbb_martins": 0.9, "bioavailability_ma": 0.9},
                liability_flags=[],
            )

    consensus = ConsensusADMETScorer(
        scorers={
            "admet_ai_fake": _FakeNeuralScorer(),
            "physchem_rules": RuleBasedADMETScorer(),
        },
    )
    # Heavy lipophilic compound where rule-based hERG = 0.75, neural = 0.05.
    result = consensus.score("CCCCCCCCCCCCCCCCCCCCc1ccc2ccccc2c1")
    # Disagreement alignment: `herg_inhibition_proxy` (rule) ↔ `herg` (neural).
    herg_key = (
        "herg_inhibition_proxy"
        if "herg_inhibition_proxy" in result.consensus
        else "herg"
    )
    c = result.consensus[herg_key]
    assert c.disagreement_flagged is True
    assert c.max_disagreement > 0.4
    # Flag appears in aggregate
    assert any(
        flag.startswith("consensus:disagree:") for flag in result.aggregate_flags
    )


def test_consensus_aggregate_flags_union_across_scorers():
    class _FakeNeuralFlags:
        def score(self, ligand):
            smiles = ligand.smiles if hasattr(ligand, "smiles") else ligand
            return ADMETProfile(
                ligand_smiles=smiles,
                endpoints={"herg": 0.9},
                liability_flags=["neural:hepatotox"],
            )

    consensus = ConsensusADMETScorer(
        scorers={
            "neural": _FakeNeuralFlags(),
            "physchem_rules": RuleBasedADMETScorer(),
        },
    )
    # Compound with Lipinski violations (heavy lipophilic) → rule-based flags.
    result = consensus.score("CCCCCCCCCCCCCCCCCCCCc1ccc2ccccc2c1")
    assert "neural:hepatotox" in result.aggregate_flags
    assert any(
        flag.startswith("rule:") for flag in result.aggregate_flags
    )


def test_consensus_default_rule_only_mode_skips_admet_ai():
    """`include_neural=False` should build a single-scorer ensemble
    without touching ADMET-AI — important for CI smoke tests that
    don't want the chemprop boot cost."""
    consensus = ConsensusADMETScorer.default(include_neural=False)
    assert list(consensus.scorers.keys()) == ["physchem_rules"]
    result = consensus.score("CCO")
    assert result.profiles["physchem_rules"].endpoints["lipinski"] == 1.0
