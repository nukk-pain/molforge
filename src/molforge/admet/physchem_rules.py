"""Rule-based ADMET predictions from RDKit descriptors.

A complement to the neural `ADMETScorer` (ADMET-AI). This module computes
Lipinski / Veber / Egan / hERG-proxy / BBB-proxy / oral-bioavailability
assessments from fundamental physicochemical descriptors. Each rule is
independent of the neural model, giving the `ConsensusADMETScorer` a
second opinion that disagrees-meaningfully when the learned model
over-confidently accepts a molecule that the rules would reject (and
vice versa).

Endpoints produced here deliberately parallel the ADMET-AI endpoint
namespace where possible (lipinski, bioavailability, bbb_permeability,
herg_inhibition_proxy) so the consensus layer can compare them
directly.

Limitation: rule-based predictors are calibrated but not trained. They
match ~70-75 % of neural predictions on TDC benchmarks (Kim et al.
2021) — worse than ADMET-AI in absolute terms but useful precisely
because the error modes are different.
"""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski


@dataclass(frozen=True, slots=True)
class PhyschemEvaluation:
    """Raw physchem numbers used by the rule-based endpoint predictors."""

    molecular_weight: float
    logp: float           # Crippen logP
    hbd: int              # H-bond donors
    hba: int              # H-bond acceptors
    tpsa: float           # topological polar surface area
    rotatable_bonds: int
    heavy_atoms: int
    aromatic_rings: int
    n_rings: int


def _evaluate_physchem(smiles: str) -> PhyschemEvaluation:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES for physchem evaluation: {smiles!r}")
    return PhyschemEvaluation(
        molecular_weight=Descriptors.MolWt(mol),
        logp=Crippen.MolLogP(mol),
        hbd=Lipinski.NumHDonors(mol),
        hba=Lipinski.NumHAcceptors(mol),
        tpsa=Descriptors.TPSA(mol),
        rotatable_bonds=Descriptors.NumRotatableBonds(mol),
        heavy_atoms=mol.GetNumHeavyAtoms(),
        aromatic_rings=Lipinski.NumAromaticRings(mol),
        n_rings=Lipinski.RingCount(mol),
    )


def _lipinski_pass(p: PhyschemEvaluation) -> float:
    """1.0 if Ro5 pass (≤1 violation), else 0.0 = oral drug-like."""
    violations = (
        int(p.molecular_weight > 500)
        + int(p.logp > 5.0)
        + int(p.hbd > 5)
        + int(p.hba > 10)
    )
    return 1.0 if violations <= 1 else 0.0


def _veber_pass(p: PhyschemEvaluation) -> float:
    """Veber oral-bioavailability rule: rotatable ≤10 AND TPSA ≤140."""
    return 1.0 if p.rotatable_bonds <= 10 and p.tpsa <= 140.0 else 0.0


def _egan_pass(p: PhyschemEvaluation) -> float:
    """Egan oral-absorption rule: TPSA ≤131.6 AND -1 ≤ logP ≤ 5.88."""
    return 1.0 if p.tpsa <= 131.6 and -1.0 <= p.logp <= 5.88 else 0.0


def _herg_risk_proxy(p: PhyschemEvaluation) -> float:
    """Rule-of-thumb hERG liability flag (Aronov 2008-style): logP ≥ 3.7
    AND MW ≥ 400 strongly associate with cardiac risk. Returns
    probability-of-inhibition as 0.0 (safe) or 0.75 (high-risk flag).
    Not a trained predictor — a calibrated rule to disagree with
    ADMET-AI when the learned model under-weights the classic risk
    pattern."""
    if p.logp >= 3.7 and p.molecular_weight >= 400:
        return 0.75
    if p.logp >= 4.5:
        return 0.5
    return 0.1


def _bbb_permeability_proxy(p: PhyschemEvaluation) -> float:
    """Clark's CNS+ rule: TPSA <70 AND MW ≤ 450 favours BBB crossing."""
    if p.tpsa < 70.0 and p.molecular_weight <= 450:
        return 0.85
    if p.tpsa < 90.0:
        return 0.6
    return 0.2


def _bioavailability_proxy(p: PhyschemEvaluation) -> float:
    """A coarse oral bioavailability score combining Lipinski + Veber.
    0, 0.5, or 1.0."""
    return 0.5 * _lipinski_pass(p) + 0.5 * _veber_pass(p)


def _qed_proxy(p: PhyschemEvaluation) -> float:
    """Weighted desirability in [0,1] — hand-rolled substitute for QED
    when we want a rule-only signal. Not a replacement for RDKit's real
    QED which `ADMETScorer` already exposes — this is for the consensus
    layer to see a rule-derived opinion."""
    score = 0.0
    score += 0.25 if 150 <= p.molecular_weight <= 500 else 0.0
    score += 0.25 if -0.5 <= p.logp <= 5.0 else 0.0
    score += 0.25 if p.hbd <= 5 and p.hba <= 10 else 0.0
    score += 0.25 if 20 <= p.tpsa <= 140 else 0.0
    return score


RULE_ENDPOINTS: dict[str, object] = {
    # Endpoint name → callable(PhyschemEvaluation) → float in [0,1]
    "lipinski": _lipinski_pass,
    "veber": _veber_pass,
    "egan": _egan_pass,
    "bioavailability_proxy": _bioavailability_proxy,
    "bbb_permeability_proxy": _bbb_permeability_proxy,
    "herg_inhibition_proxy": _herg_risk_proxy,
    "qed_rule_proxy": _qed_proxy,
}


def score_physchem_rules(smiles: str) -> dict[str, float]:
    """Return a flat endpoint dict for `smiles` using rule-based proxies.

    The returned keys parallel ADMET-AI endpoint naming where conceptually
    similar (e.g. `bbb_permeability_proxy` vs ADMET-AI's `bbb_martins`)
    so the consensus layer can align them via its endpoint mapping.
    """
    p = _evaluate_physchem(smiles)
    return {name: float(func(p)) for name, func in RULE_ENDPOINTS.items()}


def liability_flags_from_rules(smiles: str) -> list[str]:
    """Return a liability_flags list compatible with ADMETProfile.
    Flags fired by the rule-based predictors, for inclusion in the
    wet-lab handoff report."""
    p = _evaluate_physchem(smiles)
    flags: list[str] = []
    if _lipinski_pass(p) == 0.0:
        flags.append("rule:lipinski_violation")
    if _veber_pass(p) == 0.0:
        flags.append("rule:veber_violation")
    if _herg_risk_proxy(p) >= 0.5:
        flags.append("rule:herg_risk_logp_mw")
    if p.molecular_weight > 500:
        flags.append("rule:mw_over_500")
    if p.logp > 5.0:
        flags.append("rule:logp_over_5")
    if p.rotatable_bonds > 10:
        flags.append("rule:rotbonds_over_10")
    if p.tpsa > 140.0:
        flags.append("rule:tpsa_over_140")
    return flags
