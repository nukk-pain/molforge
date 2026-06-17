from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib import parse, request

CONFIG_PATH = Path(__file__).resolve().parent / "config" / "off_targets.json"
CHEMBL_ACTIVITY_URL = "https://www.ebi.ac.uk/chembl/api/data/activity.json?target_chembl_id={target_chembl_id}&standard_type=IC50&limit={limit}"
CHEMBL_MOLECULE_URL = (
    "https://www.ebi.ac.uk/chembl/api/data/molecule/{molecule_chembl_id}.json"
)


@dataclass(frozen=True, slots=True)
class OffTargetDefinition:
    gene: str
    target_chembl_id: str
    fallback_reference_smiles: list[str]


def load_off_target_definitions() -> list[OffTargetDefinition]:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return [OffTargetDefinition(**entry) for entry in payload["targets"]]


def fetch_target_ligands(
    target_chembl_id: str,
    *,
    max_n: int = 50,
    cache_dir: str | Path,
    timeout_seconds: float = 15.0,
    limit: int = 200,
    sleep_seconds: float = 1.0,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> list[str]:
    cache_path = Path(cache_dir) / f"chembl_off_target_{target_chembl_id}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        return [str(item) for item in cached.get("smiles", [])]

    activity_url = CHEMBL_ACTIVITY_URL.format(
        target_chembl_id=parse.quote(target_chembl_id, safe=""),
        limit=limit,
    )
    activity_payload = fetch_json(activity_url, timeout_seconds=timeout_seconds)
    activities = activity_payload.get("activities", [])
    if not isinstance(activities, list):
        raise ValueError("Unexpected ChEMBL activity payload: missing activities list.")

    molecule_ids = dedupe_molecule_ids(activities)[:max_n]
    smiles_values: list[str] = []
    for molecule_id in molecule_ids:
        sleep_fn(sleep_seconds)
        molecule_payload = fetch_json(
            CHEMBL_MOLECULE_URL.format(
                molecule_chembl_id=parse.quote(molecule_id, safe="")
            ),
            timeout_seconds=timeout_seconds,
        )
        smiles = extract_canonical_smiles(molecule_payload)
        if smiles:
            smiles_values.append(smiles)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {"target_chembl_id": target_chembl_id, "smiles": smiles_values},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return smiles_values


def dedupe_molecule_ids(activities: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    molecule_ids: list[str] = []
    for activity in activities:
        molecule_id = activity.get("molecule_chembl_id")
        if not isinstance(molecule_id, str) or not molecule_id.strip():
            continue
        if molecule_id in seen:
            continue
        seen.add(molecule_id)
        molecule_ids.append(molecule_id)
    return molecule_ids


def extract_canonical_smiles(payload: dict[str, Any]) -> str | None:
    structures = payload.get("molecule_structures")
    if not isinstance(structures, dict):
        return None
    canonical_smiles = structures.get("canonical_smiles")
    if not isinstance(canonical_smiles, str) or not canonical_smiles.strip():
        return None
    return canonical_smiles.strip()


def fetch_json(url: str, *, timeout_seconds: float) -> dict[str, Any]:
    http_request = request.Request(url, headers={"Accept": "application/json"})
    with request.urlopen(http_request, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Expected ChEMBL JSON object payload.")
    return payload
