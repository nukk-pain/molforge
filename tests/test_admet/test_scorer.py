from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from contracts.schema import Ligand

endpoint_names = importlib.import_module(
    "molforge.admet._endpoints"
).get_endpoint_names()
scorer_module = importlib.import_module("molforge.admet.scorer")
ADMETScorer = scorer_module.ADMETScorer
build_default_model_kwargs = scorer_module.build_default_model_kwargs
instantiate_model = scorer_module.instantiate_model

RAW_ADMET_AI_ENDPOINT_NAMES = [
    "AMES",
    "BBB_Martins",
    "Bioavailability_Ma",
    "CYP1A2_Veith",
    "CYP2C19_Veith",
    "CYP2C9_Substrate_CarbonMangels",
    "CYP2C9_Veith",
    "CYP2D6_Substrate_CarbonMangels",
    "CYP2D6_Veith",
    "CYP3A4_Substrate_CarbonMangels",
    "CYP3A4_Veith",
    "Caco2_Wang",
    "Carcinogens_Lagunin",
    "Clearance_Hepatocyte_AZ",
    "Clearance_Microsome_AZ",
    "ClinTox",
    "DILI",
    "HIA_Hou",
    "Half_Life_Obach",
    "HydrationFreeEnergy_FreeSolv",
    "LD50_Zhu",
    "Lipophilicity_AstraZeneca",
    "NR-AR",
    "NR-AR-LBD",
    "NR-AhR",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "PAMPA_NCATS",
    "PPBR_AZ",
    "Pgp_Broccatelli",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
    "Skin_Reaction",
    "Solubility_AqSolDB",
    "VDss_Lombardo",
    "hERG",
]


def make_prediction(smiles: str) -> dict[str, float | str]:
    return {"smiles": smiles} | {
        endpoint_name: (index + 1) / 100
        for index, endpoint_name in enumerate(endpoint_names)
    }


def test_admet_scorer_scores_single_profile() -> None:
    class FakeModel:
        def predict(self, smiles: str) -> dict[str, float | str]:
            return make_prediction(smiles)

    scorer = ADMETScorer(model=FakeModel())
    profile = scorer.score(Ligand(smiles="CCO", source="user"))

    assert profile.ligand_smiles == "CCO"
    assert len(profile.endpoints) == 41
    assert set(profile.endpoints) == set(endpoint_names)


def test_admet_scorer_canonicalizes_real_admet_ai_names_and_flags_risks() -> None:
    smiles = "COc1ccc(-c2ccc3cc(C(=O)O)ccc3c2)cc1C12CC3CC(CC(C3)C1)C2"

    class FakeModel:
        def predict(self, requested_smiles: str) -> dict[str, float | str]:
            assert requested_smiles == smiles
            prediction = {name: 0.1 for name in RAW_ADMET_AI_ENDPOINT_NAMES}
            prediction["DILI"] = 0.91
            prediction["hERG"] = 0.72
            return {"smiles": requested_smiles} | prediction

    scorer = ADMETScorer(model=FakeModel())
    profile = scorer.score(Ligand(smiles=smiles, source="chembl_fda"))

    assert len(profile.endpoints) == 41
    assert "dili" in profile.endpoints
    assert "DILI" not in profile.endpoints
    assert "hepatotox" in profile.liability_flags
    assert "hERG_high" in profile.liability_flags
    assert "rule:logp_over_5" in profile.liability_flags


def test_admet_scorer_scores_batch_profiles_and_tolerates_invalid_smiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FrameLike:
        def __init__(self, rows: list[dict[str, float | str]]) -> None:
            self.rows = rows

        def to_dict(self, orient: str = "dict") -> list[dict[str, float | str]]:
            assert orient == "records"
            return self.rows

    class FakeModel:
        def predict(self, smiles_list: list[str]) -> FrameLike:
            return FrameLike([make_prediction(smiles) for smiles in smiles_list])

    monkeypatch.setattr(
        "molforge.admet.scorer.validate_smiles",
        lambda smiles: smiles != "INVALID",
    )
    scorer = ADMETScorer(model=FakeModel())
    profiles = scorer.score_batch(
        [
            Ligand(smiles="CCO", source="user"),
            Ligand(smiles="INVALID", source="user"),
            Ligand(smiles="CCN", source="user"),
        ]
    )

    assert profiles[0].ligand_smiles == "CCO"
    assert len(profiles[0].endpoints) == 41
    assert profiles[1].liability_flags == ["invalid_smiles"]
    assert profiles[1].endpoints == {}
    assert profiles[2].ligand_smiles == "CCN"


def test_instantiate_model_prefers_cpu_safe_defaults_on_macos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, int] = {}

    class FakeModel:
        def __init__(self, **kwargs: int) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(
        "molforge.admet.scorer.load_admet_model_class", lambda: FakeModel
    )
    monkeypatch.setattr("platform.system", lambda: "Darwin")

    instantiate_model()

    assert build_default_model_kwargs() == {"num_workers": 0}
    assert captured == {"num_workers": 0}


def test_admet_scorer_single_invalid_smiles_raises_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scorer = ADMETScorer(model=object())
    monkeypatch.setattr("molforge.admet.scorer.validate_smiles", lambda smiles: False)

    with pytest.raises(ValueError, match="Invalid SMILES value"):
        scorer.score(Ligand(smiles="not-a-smiles", source="user"))


def test_instantiate_model_falls_back_to_external_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "molforge.admet.scorer.load_admet_model_class",
        lambda: (_ for _ in ()).throw(RuntimeError("missing admet_ai")),
    )
    monkeypatch.setattr(
        "molforge.admet.scorer.resolve_external_admet_python",
        lambda: Path("/fake/python"),
    )

    model = instantiate_model()

    assert type(model).__name__ == "ExternalADMETModel"
    assert model.python_executable == Path("/fake/python")


def test_admet_scorer_external_runtime_supports_batch_predictions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "molforge.admet.scorer.load_admet_model_class",
        lambda: (_ for _ in ()).throw(RuntimeError("missing admet_ai")),
    )
    monkeypatch.setattr(
        "molforge.admet.scorer.resolve_external_admet_python",
        lambda: Path("/fake/python"),
    )
    monkeypatch.setattr(
        "molforge.admet.scorer.run_external_prediction",
        lambda smiles, python_executable: [make_prediction(item) for item in smiles],
    )

    scorer = ADMETScorer(model=instantiate_model())
    profiles = scorer.score_batch(
        [
            Ligand(smiles="CCO", source="user"),
            Ligand(smiles="CCN", source="user"),
        ]
    )

    assert [profile.ligand_smiles for profile in profiles] == ["CCO", "CCN"]
    assert all(len(profile.endpoints) == 41 for profile in profiles)
