from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
from rdkit import Chem
from rdkit.Chem import rdDistGeom

from contracts.schema import Ligand

CHEMBL_API_ROOT = "https://www.ebi.ac.uk/chembl/api/data"
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_CHEMBL_CACHE_DIR = Path("archive/cache/chembl")


def load_fda_approved_library(
    *,
    cache_dir: Path = DEFAULT_CHEMBL_CACHE_DIR,
    max_n: int = 500,
) -> list[Ligand]:
    cache_path = _cache_path(cache_dir)
    if cache_path.exists():
        return _load_cached_library(cache_path, max_n=max_n)

    chembl_ids: list[str] = []
    seen_ids: set[str] = set()
    next_url = f"{CHEMBL_API_ROOT}/drug_indication.json?max_phase=4&limit=200&offset=0"

    while next_url and len(chembl_ids) < max_n:
        payload = _get_json(next_url)
        drug_indications = payload.get("drug_indications")
        if not isinstance(drug_indications, list):
            drug_indications = []
        for item in drug_indications:
            if not isinstance(item, dict):
                continue
            chembl_id = item.get("molecule_chembl_id")
            if not isinstance(chembl_id, str) or not chembl_id.strip():
                continue
            if chembl_id in seen_ids:
                continue
            seen_ids.add(chembl_id)
            chembl_ids.append(chembl_id)
        page_meta = payload.get("page_meta")
        next_token = page_meta.get("next") if isinstance(page_meta, dict) else None
        next_url = _normalize_next_url(next_token)

    library: list[Ligand] = []
    for chembl_id in chembl_ids:
        smiles = _fetch_canonical_smiles(chembl_id)
        if smiles is None or not is_docking_compatible_smiles(smiles):
            continue
        library.append(Ligand(smiles=smiles, source="chembl_fda", chembl_id=chembl_id))
        if len(library) >= max_n:
            break

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            [
                {
                    "smiles": ligand.smiles,
                    "source": ligand.source,
                    "chembl_id": ligand.chembl_id,
                }
                for ligand in library
            ],
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return library


def _cache_path(cache_dir: Path) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d")
    return cache_dir / f"chembl_fda_v{stamp}.json"


def _load_cached_library(path: Path, *, max_n: int) -> list[Ligand]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Cached ChEMBL library at {path} must contain a JSON array.")
    library: list[Ligand] = []
    for item in payload[:max_n]:
        if not isinstance(item, dict):
            continue
        smiles = item.get("smiles")
        source = item.get("source")
        chembl_id = item.get("chembl_id")
        if not isinstance(smiles, str) or not isinstance(source, str):
            continue
        if not is_docking_compatible_smiles(smiles):
            continue
        library.append(
            Ligand(
                smiles=smiles,
                source=source,
                chembl_id=chembl_id if isinstance(chembl_id, str) else None,
            )
        )
    return library


def _fetch_canonical_smiles(chembl_id: str) -> str | None:
    payload = _get_json(f"{CHEMBL_API_ROOT}/molecule/{chembl_id}.json")
    molecule_structures = payload.get("molecule_structures")
    if not isinstance(molecule_structures, dict):
        return None
    smiles = molecule_structures.get("canonical_smiles")
    if not isinstance(smiles, str):
        return None
    normalized = smiles.strip()
    return normalized or None


DOCKING_ALLOWED_ATOMIC_NUMBERS = {
    5,   # B
    6,   # C
    7,   # N
    8,   # O
    9,   # F
    15,  # P
    16,  # S
    17,  # Cl
    35,  # Br
    53,  # I
}


def is_docking_compatible_smiles(smiles: str) -> bool:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return False
    molecule = _select_largest_fragment(molecule)
    if any(
        atom.GetAtomicNum() not in DOCKING_ALLOWED_ATOMIC_NUMBERS
        for atom in molecule.GetAtoms()
    ):
        return False
    molecule = Chem.AddHs(molecule)
    params = rdDistGeom.ETKDGv3()
    params.randomSeed = 0xF00D
    return rdDistGeom.EmbedMolecule(molecule, params) == 0


def _select_largest_fragment(molecule: Chem.Mol) -> Chem.Mol:
    fragments = Chem.GetMolFrags(molecule, asMols=True, sanitizeFrags=True)
    if not fragments:
        return molecule
    return max(
        fragments,
        key=lambda fragment: (fragment.GetNumHeavyAtoms(), fragment.GetNumAtoms()),
    )


def _get_json(url: str) -> dict[str, object]:
    response = httpx.get(url, timeout=DEFAULT_TIMEOUT_SECONDS)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(f"ChEMBL response for {url} must decode to a JSON object.")
    return payload


def _normalize_next_url(next_url: object) -> str | None:
    if not isinstance(next_url, str) or not next_url.strip():
        return None
    if next_url.startswith("http://") or next_url.startswith("https://"):
        return next_url
    return f"https://www.ebi.ac.uk{next_url}"
