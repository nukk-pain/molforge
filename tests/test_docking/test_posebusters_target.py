# pyright: reportMissingImports=false
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from molforge.docking.posebusters_target import (  # noqa: E402
    build_pose_target,
    fetch_ligand_smiles,
    fetch_protein_sequence,
    parse_target_name,
)


def test_parse_target_name_splits_pdb_and_ligand() -> None:
    assert parse_target_name("6X8D_ARA") == ("6X8D", "ARA")
    assert parse_target_name("7a1p_qw2") == ("7A1P", "QW2")


def test_parse_target_name_rejects_malformed() -> None:
    with pytest.raises(ValueError):
        parse_target_name("no-separator")
    with pytest.raises(ValueError):
        parse_target_name("6X8D_")


def test_fetch_protein_sequence_reads_entity_poly() -> None:
    payload = {
        "entity_poly": {
            "pdbx_seq_one_letter_code_can": "MKTAYIAKQR\nALAIN"
        }
    }
    seq = fetch_protein_sequence("6X8D", fetch_json=lambda url: payload)
    assert seq == "MKTAYIAKQRALAIN"


def test_fetch_protein_sequence_raises_when_payload_incomplete() -> None:
    with pytest.raises(ValueError):
        fetch_protein_sequence("6X8D", fetch_json=lambda url: {})


def test_fetch_ligand_smiles_prefers_singular_smiles_field() -> None:
    payload = {
        "rcsb_chem_comp_descriptor": {
            "SMILES": "c1ccccc1",
            "SMILES_stereo": "c1cccnc1",
        }
    }
    smiles = fetch_ligand_smiles("ARA", fetch_json=lambda url: payload)
    # SMILES_stereo is preferred over plain SMILES (RCSB real schema)
    assert smiles == "c1cccnc1"


def test_fetch_ligand_smiles_falls_back_to_descriptor_list() -> None:
    payload = {
        "pdbx_chem_comp_descriptor": [
            {"program": "CACTVS", "type": "SMILES", "descriptor": "O=C=O"},
            {"program": "OpenEye OEToolkits", "type": "SMILES_CANONICAL", "descriptor": "O=C=O"},
        ]
    }
    smiles = fetch_ligand_smiles("CO2", fetch_json=lambda url: payload)
    assert smiles == "O=C=O"


def test_fetch_ligand_smiles_raises_when_no_smiles_anywhere() -> None:
    with pytest.raises(ValueError):
        fetch_ligand_smiles("XYZ", fetch_json=lambda url: {})


def test_build_pose_target_assembles_full_record() -> None:
    polymer = {
        "entity_poly": {"pdbx_seq_one_letter_code_can": "MKTAY"}
    }
    ligand = {"rcsb_chem_comp_descriptor": {"SMILES": "CCO"}}

    def fake_fetch(url: str) -> dict:
        return polymer if "polymer_entity" in url else ligand

    target = build_pose_target("6X8D_ARA", fetch_json=fake_fetch)
    assert target.pdb_id == "6X8D"
    assert target.ligand_code == "ARA"
    assert target.sequence == "MKTAY"
    assert target.ligand_smiles == "CCO"


@pytest.mark.skipif(
    os.environ.get("RCSB_LIVE") != "1",
    reason="RCSB_LIVE=1 not set — live RCSB smoke is opt-in.",
)
def test_build_pose_target_live_6x8d() -> None:
    target = build_pose_target("6X8D_ARA")
    assert len(target.sequence) >= 20
    assert target.ligand_smiles and len(target.ligand_smiles) > 3
    # ARA = arabinose → should contain an oxygen + sugar pattern
    assert "O" in target.ligand_smiles.upper()
