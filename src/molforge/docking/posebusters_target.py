"""Fetch PoseBusters target context (sequence + ligand SMILES) from RCSB PDB.

PoseBench target names follow the `<PDB_ID>_<LIGAND_3LETTER_CODE>` pattern
(confirmed via phase2 artifact inspection). This module resolves each name
to the minimum inputs `reselect_via_boltz2` needs:
  - protein one-letter sequence
  - ligand canonical SMILES (RCSB's authoritative SMILES for the 3-letter code)

Both are fetched from the public RCSB REST API; no auth, low rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import httpx

RCSB_POLYMER_URL = "https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/1"
RCSB_CHEMCOMP_URL = "https://data.rcsb.org/rest/v1/core/chemcomp/{ligand_code}"

FetchJson = Callable[[str], dict]


@dataclass(frozen=True, slots=True)
class PoseTarget:
    name: str
    pdb_id: str
    ligand_code: str
    sequence: str
    ligand_smiles: str
    reference_pose_pdbqt: Path | None = None


def parse_target_name(name: str) -> tuple[str, str]:
    """Parse a PoseBench target like `6X8D_ARA` into (pdb_id, ligand_code)."""
    parts = name.split("_")
    if len(parts) != 2 or not all(parts):
        raise ValueError(
            f"Target name must match <PDB_ID>_<LIGAND_CODE>, got {name!r}"
        )
    pdb_id, ligand_code = parts
    return pdb_id.upper(), ligand_code.upper()


def _default_fetch_json(url: str) -> dict:
    with httpx.Client(timeout=30.0) as client:
        response = client.get(url, headers={"Accept": "application/json"})
        response.raise_for_status()
        payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object from {url}, got {type(payload).__name__}")
    return payload


def fetch_protein_sequence(pdb_id: str, *, fetch_json: FetchJson = _default_fetch_json) -> str:
    """Return canonical one-letter protein sequence for the first polymer entity."""
    payload = fetch_json(RCSB_POLYMER_URL.format(pdb_id=pdb_id.upper()))
    entity_poly = payload.get("entity_poly")
    if not isinstance(entity_poly, dict):
        raise ValueError(f"RCSB polymer_entity payload missing entity_poly: {pdb_id}")
    sequence = entity_poly.get("pdbx_seq_one_letter_code_can")
    if not isinstance(sequence, str) or not sequence.strip():
        raise ValueError(f"RCSB polymer_entity returned empty sequence for {pdb_id}")
    # Strip whitespace/newlines that RCSB sometimes embeds.
    return "".join(sequence.split())


def fetch_ligand_smiles(ligand_code: str, *, fetch_json: FetchJson = _default_fetch_json) -> str:
    """Return a canonical SMILES for the given ligand 3-letter code.

    RCSB emits an array of descriptors; we prefer OpenEye canonical if
    present, then any SMILES descriptor.
    """
    payload = fetch_json(RCSB_CHEMCOMP_URL.format(ligand_code=ligand_code.upper()))
    descriptors = payload.get("rcsb_chem_comp_descriptor")
    if not isinstance(descriptors, dict):
        # Some entries use a nested list form; try `pdbx_chem_comp_descriptor`.
        raw_list = payload.get("pdbx_chem_comp_descriptor")
        if isinstance(raw_list, list):
            smiles = _select_smiles_from_list(raw_list)
            if smiles:
                return smiles
        raise ValueError(
            f"RCSB chemcomp {ligand_code} payload missing smiles descriptors"
        )

    # The singular field form. RCSB uses capitalised keys
    # (SMILES / SMILES_stereo); older clients may emit lowercase. Check both.
    for key in (
        "SMILES_stereo", "smiles_stereo",
        "SMILES", "smiles",
    ):
        value = descriptors.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    # Some entries only populate the nested `pdbx_chem_comp_descriptor` list.
    raw_list = payload.get("pdbx_chem_comp_descriptor")
    if isinstance(raw_list, list):
        smiles = _select_smiles_from_list(raw_list)
        if smiles:
            return smiles
    raise ValueError(f"RCSB chemcomp {ligand_code} had no usable SMILES field")


def _select_smiles_from_list(entries: list) -> str | None:
    preferred = None
    fallback = None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        program = str(entry.get("program") or "").lower()
        descriptor_type = str(entry.get("type") or "").lower()
        value = entry.get("descriptor")
        if not isinstance(value, str) or not value.strip():
            continue
        if "smiles" not in descriptor_type:
            continue
        if "openeye" in program:
            preferred = value.strip()
        elif fallback is None:
            fallback = value.strip()
    return preferred or fallback


def build_pose_target(
    name: str,
    *,
    reference_pose_pdbqt: Path | None = None,
    fetch_json: FetchJson = _default_fetch_json,
) -> PoseTarget:
    """One-shot assembly: name → full PoseTarget via RCSB."""
    pdb_id, ligand_code = parse_target_name(name)
    sequence = fetch_protein_sequence(pdb_id, fetch_json=fetch_json)
    ligand_smiles = fetch_ligand_smiles(ligand_code, fetch_json=fetch_json)
    return PoseTarget(
        name=name,
        pdb_id=pdb_id,
        ligand_code=ligand_code,
        sequence=sequence,
        ligand_smiles=ligand_smiles,
        reference_pose_pdbqt=reference_pose_pdbqt,
    )
