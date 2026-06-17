from __future__ import annotations

import json
import importlib
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from contracts.schema import ADMETProfile, Ligand

from ._endpoints import (
    canonicalize_endpoint_name,
    get_endpoint_names,
    get_expected_endpoint_count,
)
from .liability import derive_liability_flags
from .phase0_admet import (
    NON_ENDPOINT_KEYS,
    collect_endpoint_keys,
    load_admet_model_class,
    normalize_prediction_map,
    resolve_external_admet_python,
)


@dataclass(slots=True)
class ADMETScorer:
    model: Any | None = None

    def score(self, ligand: Ligand | str) -> ADMETProfile:
        normalized_ligand = coerce_ligand(ligand)
        if not validate_smiles(normalized_ligand.smiles):
            raise ValueError(
                f"Invalid SMILES value for ADMET scoring: {normalized_ligand.smiles!r}"
            )
        prediction_output = self._get_model().predict(normalized_ligand.smiles)
        normalized_prediction = normalize_prediction_map(
            normalized_ligand.smiles,
            prediction_output,
        )
        return build_profile(normalized_ligand, normalized_prediction)

    def score_batch(self, ligands: Sequence[Ligand | str]) -> list[ADMETProfile]:
        normalized_ligands = [coerce_ligand(ligand) for ligand in ligands]
        if not normalized_ligands:
            return []

        valid_ligands: list[Ligand] = []
        invalid_profiles: dict[int, ADMETProfile] = {}
        for index, ligand in enumerate(normalized_ligands):
            if validate_smiles(ligand.smiles):
                valid_ligands.append(ligand)
            else:
                invalid_profiles[index] = ADMETProfile(
                    ligand_smiles=ligand.smiles,
                    endpoints={},
                    liability_flags=["invalid_smiles"],
                )

        results: list[ADMETProfile] = [
            ADMETProfile(ligand_smiles=ligand.smiles, endpoints={}, liability_flags=[])
            for ligand in normalized_ligands
        ]
        if valid_ligands:
            prediction_output = self._get_model().predict(
                [ligand.smiles for ligand in valid_ligands]
            )
            records = normalize_batch_output(prediction_output)
            if len(records) != len(valid_ligands):
                raise ValueError(
                    "ADMET-AI batch output count did not match the number of requested ligands."
                )
            valid_index = 0
            for index, ligand in enumerate(normalized_ligands):
                if index in invalid_profiles:
                    continue
                normalized_prediction = normalize_prediction_map(
                    ligand.smiles,
                    records[valid_index],
                )
                results[index] = build_profile(ligand, normalized_prediction)
                valid_index += 1

        for index, profile in invalid_profiles.items():
            results[index] = profile
        return results

    def predict_profile(self, ligand: Ligand | str) -> ADMETProfile:
        return self.score(ligand)

    def predict_profiles(self, ligands: Sequence[Ligand | str]) -> list[ADMETProfile]:
        return self.score_batch(ligands)

    def _get_model(self) -> Any:
        if self.model is None:
            self.model = instantiate_model()
        return self.model


def instantiate_model() -> Any:
    try:
        model_class = load_admet_model_class()
    except RuntimeError:
        external_python = resolve_external_admet_python()
        if external_python is None:
            raise
        return ExternalADMETModel(external_python)
    default_kwargs = build_default_model_kwargs()
    if default_kwargs:
        try:
            return model_class(**default_kwargs)
        except TypeError:
            return model_class()
    return model_class()


@dataclass(slots=True)
class ExternalADMETModel:
    python_executable: Path

    def predict(self, smiles: str | list[str]) -> Any:
        return run_external_prediction(smiles, self.python_executable)


def run_external_prediction(
    smiles: str | list[str], python_executable: Path
) -> dict[str, Any] | list[dict[str, Any]]:
    payload = json.dumps(smiles)
    script = "\n".join(
        [
            "import contextlib, io, json, sys",
            "from admet_ai import ADMETModel",
            "def coerce(value):",
            "    if value is None or isinstance(value, (str, int, float, bool)): return value",
            "    if isinstance(value, dict): return {str(k): coerce(v) for k, v in value.items()}",
            "    if isinstance(value, (list, tuple)): return [coerce(v) for v in value]",
            "    if hasattr(value, 'item'): return coerce(value.item())",
            "    return str(value)",
            "smiles = json.loads(sys.argv[1])",
            "capture = io.StringIO()",
            "with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):",
            "    model = ADMETModel()",
            "    preds = model.predict(smiles=smiles)",
            "if hasattr(preds, 'to_dict'):",
            "    try:",
            "        preds = preds.to_dict(orient='records')",
            "    except TypeError:",
            "        preds = preds.to_dict()",
            "print(json.dumps(coerce(preds)))",
        ]
    )
    completed = subprocess.run(
        [str(python_executable), "-c", script, payload],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "ADMET-AI v2 external runtime prediction failed: "
            + (completed.stderr.strip() or completed.stdout.strip() or "unknown error")
        )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "ADMET-AI v2 external runtime returned invalid JSON."
        ) from exc


def build_default_model_kwargs() -> dict[str, int]:
    if platform.system() == "Darwin":
        return {"num_workers": 0}
    return {}


def build_profile(ligand: Ligand, prediction_map: dict[str, Any]) -> ADMETProfile:
    raw_endpoint_names = collect_endpoint_keys(prediction_map)
    canonical_endpoint_names = set(get_endpoint_names())
    expected_endpoint_count = get_expected_endpoint_count()
    endpoints: dict[str, float] = {}
    unknown_endpoint_names: list[str] = []
    for raw_name in raw_endpoint_names:
        canonical_name = canonicalize_endpoint_name(raw_name)
        if canonical_name not in canonical_endpoint_names:
            unknown_endpoint_names.append(raw_name)
            continue
        if canonical_name in endpoints:
            raise ValueError(
                "ADMET-AI returned duplicate endpoint names after canonicalization: "
                f"{canonical_name}."
            )
        endpoints[canonical_name] = float(prediction_map[raw_name])

    missing_endpoint_names = sorted(canonical_endpoint_names - set(endpoints))
    if (
        len(endpoints) != expected_endpoint_count
        or unknown_endpoint_names
        or missing_endpoint_names
    ):
        raise ValueError(
            "ADMET-AI returned an unexpected endpoint set for Phase 4: "
            f"expected {expected_endpoint_count}, observed {len(endpoints)}, "
            f"unknown={unknown_endpoint_names}, missing={missing_endpoint_names}."
        )

    raw_endpoint_name_set = set(raw_endpoint_names)
    physchem = {
        key: prediction_map[key]
        for key in prediction_map
        if key not in raw_endpoint_name_set
        and key not in NON_ENDPOINT_KEYS
        and not key.startswith("_")
    }
    liability_flags = sorted(
        set(derive_liability_flags(endpoints, physchem))
        | set(derive_rule_liability_flags(ligand.smiles))
    )
    return ADMETProfile(
        ligand_smiles=ligand.smiles,
        endpoints=endpoints,
        liability_flags=liability_flags,
    )


def derive_rule_liability_flags(smiles: str) -> list[str]:
    try:
        from .physchem_rules import liability_flags_from_rules
    except ModuleNotFoundError:
        return []
    return liability_flags_from_rules(smiles)


def normalize_batch_output(prediction_output: Any) -> list[dict[str, Any]]:
    candidate = prediction_output
    if hasattr(candidate, "to_dict"):
        try:
            candidate = candidate.to_dict(orient="records")
        except TypeError:
            candidate = candidate.to_dict()

    if isinstance(candidate, list):
        if not all(isinstance(item, dict) for item in candidate):
            raise ValueError("ADMET-AI batch output must be a list of prediction maps.")
        return list(candidate)

    if isinstance(candidate, dict):
        return [candidate]

    raise ValueError("ADMET-AI batch output must be dict-like or DataFrame-like.")


def coerce_ligand(ligand: Ligand | str) -> Ligand:
    if isinstance(ligand, Ligand):
        normalized_smiles = ligand.smiles.strip()
        if not normalized_smiles:
            raise ValueError("Ligand SMILES is required for ADMET scoring.")
        return Ligand(
            smiles=normalized_smiles,
            source=ligand.source,
            chembl_id=ligand.chembl_id,
        )

    normalized_smiles = ligand.strip()
    if not normalized_smiles:
        raise ValueError("Ligand SMILES is required for ADMET scoring.")
    return Ligand(smiles=normalized_smiles, source="user")


def validate_smiles(smiles: str) -> bool:
    normalized_smiles = smiles.strip()
    if not normalized_smiles:
        return False
    try:
        Chem = importlib.import_module("rdkit.Chem")
    except ModuleNotFoundError:
        return True
    return Chem.MolFromSmiles(normalized_smiles) is not None
