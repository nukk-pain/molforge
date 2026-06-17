"""Multi-scorer ADMET consensus with disagreement flagging.

Two independent methodologies (neural + rule-based) predict the same
endpoints; when they agree, confidence is high; when they disagree, the
wet-lab handoff surfaces a `consensus:disagree:<endpoint>` flag so the
medchem triage step can look at the candidate more carefully.

v5 ROADMAP Track H2 deliverable. Design pivot: the original plan
expected MapLight as a second ML model, but MapLight ships as Jupyter
notebooks with no pretrained weights, not as a library. Rule-based
physchem scoring is a genuinely independent methodology — neural model
errors and rule-based errors don't correlate the way two neural models
calibrated on the same dataset would — so it satisfies the ensemble
premise more cleanly than MapLight would have.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from contracts.schema import ADMETProfile, Ligand

from .physchem_rules import liability_flags_from_rules, score_physchem_rules
from .scorer import ADMETScorer, coerce_ligand


class ADMETScorerProtocol(Protocol):
    """Interface any ADMET scorer must satisfy to be composable into the
    consensus. `ADMETScorer` (ADMET-AI) already conforms; the
    rule-based scorer below is a thin shim that also conforms."""

    def score(self, ligand: Ligand | str) -> ADMETProfile: ...


@dataclass(slots=True)
class RuleBasedADMETScorer:
    """ADMETScorer-compatible wrapper around `physchem_rules`.

    Exposes the same `.score(ligand)` signature so the consensus layer
    can treat learned and rule-based predictors symmetrically. The
    resulting `ADMETProfile.endpoints` carries the rule-derived
    endpoint names (lipinski, veber, bbb_permeability_proxy, ...) with
    a `rule:` prefix in liability flags so consumers can tell which
    scorer produced each flag.
    """

    name: str = "physchem_rules"

    def score(self, ligand: Ligand | str) -> ADMETProfile:
        normalized = coerce_ligand(ligand)
        endpoints = score_physchem_rules(normalized.smiles)
        flags = liability_flags_from_rules(normalized.smiles)
        return ADMETProfile(
            ligand_smiles=normalized.smiles,
            endpoints=endpoints,
            liability_flags=flags,
        )


# Mapping from a rule-based endpoint name to the neural endpoint it should be
# compared against. Only populated for pairs where the semantics really
# parallel — not just textually similar. Anything not in this map is
# reported as a single-scorer value without disagreement flagging.
_ENDPOINT_ALIGNMENT: dict[str, str] = {
    # rule endpoint name -> ADMET-AI endpoint name
    "herg_inhibition_proxy": "herg",
    "bbb_permeability_proxy": "bbb_martins",
    "bioavailability_proxy": "bioavailability_ma",
}


# When endpoints are on [0, 1] (all rule outputs here), we flag
# "disagreement" when |neural - rule| > threshold.
_DISAGREEMENT_THRESHOLD = 0.4


@dataclass(frozen=True, slots=True)
class ConsensusEndpoint:
    """Per-endpoint consensus across scorers."""

    endpoint: str
    scorer_values: dict[str, float]  # scorer_name -> value
    mean_value: float
    max_disagreement: float  # max |value_i - value_j|
    disagreement_flagged: bool


@dataclass(frozen=True, slots=True)
class ConsensusADMETProfile:
    """Union of individual per-scorer profiles + cross-scorer consensus.

    `profiles` preserves the raw ADMETProfile from each underlying
    scorer so callers can inspect them. `consensus` rolls each
    endpoint across scorers; `aggregate_flags` includes the union of
    all scorer-specific flags plus any consensus:disagree:* flags.
    """

    ligand_smiles: str
    profiles: dict[str, ADMETProfile]  # scorer_name -> its profile
    consensus: dict[str, ConsensusEndpoint]
    aggregate_flags: list[str]


@dataclass(slots=True)
class ConsensusADMETScorer:
    """Compose N scorers, run them all, surface disagreement.

    Typical use: one `ADMETScorer` (ADMET-AI) + one
    `RuleBasedADMETScorer`. Both score each ligand; the consensus
    layer correlates the aligned endpoints (see `_ENDPOINT_ALIGNMENT`)
    and flags disagreements beyond `_DISAGREEMENT_THRESHOLD`.
    """

    scorers: dict[str, ADMETScorerProtocol]

    @classmethod
    def default(
        cls,
        *,
        include_neural: bool = True,
    ) -> "ConsensusADMETScorer":
        """Assemble the default ensemble: ADMET-AI + rule-based.

        `include_neural=False` skips ADMET-AI so test suites and
        offline smokes that don't want to spin up Chemprop can still
        exercise the rule-based path.
        """
        scorers: dict[str, ADMETScorerProtocol] = {
            "physchem_rules": RuleBasedADMETScorer(),
        }
        if include_neural:
            scorers = {"admet_ai": ADMETScorer(), **scorers}
        return cls(scorers=scorers)

    def score(self, ligand: Ligand | str) -> ConsensusADMETProfile:
        profiles: dict[str, ADMETProfile] = {}
        for name, scorer in self.scorers.items():
            profiles[name] = scorer.score(ligand)

        smiles = next(iter(profiles.values())).ligand_smiles
        consensus_map: dict[str, ConsensusEndpoint] = {}
        disagreement_flags: list[str] = []

        # Strategy: iterate every endpoint that appears in ANY scorer.
        # If the endpoint has a mapped alias in another scorer (via
        # `_ENDPOINT_ALIGNMENT`), compute cross-scorer disagreement.
        endpoint_names: set[str] = set()
        for p in profiles.values():
            endpoint_names.update(p.endpoints.keys())

        for endpoint in endpoint_names:
            values: dict[str, float] = {}
            # Collect direct hits.
            for scorer_name, p in profiles.items():
                if endpoint in p.endpoints:
                    values[scorer_name] = float(p.endpoints[endpoint])
            # Collect via alignment map (rule ↔ neural endpoint names).
            aligned = _ENDPOINT_ALIGNMENT.get(endpoint)
            if aligned:
                for scorer_name, p in profiles.items():
                    if scorer_name not in values and aligned in p.endpoints:
                        values[scorer_name] = float(p.endpoints[aligned])
            # Reverse alignment: if we are looking at a neural endpoint
            # that corresponds to a rule endpoint.
            reverse = [
                rule_name
                for rule_name, neural in _ENDPOINT_ALIGNMENT.items()
                if neural == endpoint
            ]
            for rule_name in reverse:
                for scorer_name, p in profiles.items():
                    if scorer_name not in values and rule_name in p.endpoints:
                        values[scorer_name] = float(p.endpoints[rule_name])

            if not values:
                continue
            nums = list(values.values())
            mean_val = sum(nums) / len(nums)
            if len(nums) >= 2:
                max_disagree = max(nums) - min(nums)
            else:
                max_disagree = 0.0
            flagged = (
                len(nums) >= 2 and max_disagree > _DISAGREEMENT_THRESHOLD
            )
            if flagged:
                disagreement_flags.append(
                    f"consensus:disagree:{endpoint}:{max_disagree:.2f}"
                )
            consensus_map[endpoint] = ConsensusEndpoint(
                endpoint=endpoint,
                scorer_values=values,
                mean_value=mean_val,
                max_disagreement=max_disagree,
                disagreement_flagged=flagged,
            )

        # Flatten all per-scorer flags + consensus disagreement flags.
        aggregate_flags: list[str] = []
        for scorer_name, p in profiles.items():
            for flag in p.liability_flags:
                if flag not in aggregate_flags:
                    aggregate_flags.append(flag)
        for flag in disagreement_flags:
            if flag not in aggregate_flags:
                aggregate_flags.append(flag)

        return ConsensusADMETProfile(
            ligand_smiles=smiles,
            profiles=profiles,
            consensus=consensus_map,
            aggregate_flags=aggregate_flags,
        )

    def score_batch(
        self, ligands: list[Ligand | str]
    ) -> list[ConsensusADMETProfile]:
        return [self.score(lg) for lg in ligands]
