# pyright: reportMissingImports=false
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from molforge.docking import chembl  # noqa: E402


def test_load_fda_approved_library_paginates_and_filters_invalid_smiles(
    tmp_path: Path, monkeypatch
) -> None:
    total_ids = 101
    all_ids = [f"CHEMBL{i:04d}" for i in range(total_ids)]

    def fake_get_json(url: str) -> dict[str, object]:
        if "drug_indication.json" in url and "offset=0" in url:
            return {
                "drug_indications": [
                    {"molecule_chembl_id": chembl_id} for chembl_id in all_ids[:60]
                ],
                "page_meta": {
                    "next": "/chembl/api/data/drug_indication.json?max_phase=4&limit=200&offset=60"
                },
            }
        if "drug_indication.json" in url and "offset=60" in url:
            return {
                "drug_indications": [
                    {"molecule_chembl_id": chembl_id} for chembl_id in all_ids[60:]
                ],
                "page_meta": {"next": None},
            }
        chembl_id = url.rsplit("/", 1)[-1].replace(".json", "")
        smiles = "invalid@@" if chembl_id == all_ids[-1] else "CCO"
        return {"molecule_structures": {"canonical_smiles": smiles}}

    monkeypatch.setattr(chembl, "_get_json", fake_get_json)

    library = chembl.load_fda_approved_library(cache_dir=tmp_path, max_n=100)

    assert len(library) == 100
    assert all(ligand.source == "chembl_fda" for ligand in library)
    assert all(ligand.chembl_id is not None for ligand in library)
    cache_files = list(tmp_path.glob("chembl_fda_v*.json"))
    assert len(cache_files) == 1


def test_is_docking_compatible_smiles_rejects_meeko_unfriendly_elements() -> None:
    assert chembl.is_docking_compatible_smiles("CCO") is True
    assert chembl.is_docking_compatible_smiles("O=[As][As+](=O)[O-]") is False


def test_cached_library_filters_non_dockable_smiles(tmp_path: Path) -> None:
    cache_path = tmp_path / "chembl_fda_v20990101.json"
    cache_path.write_text(
        """
        [
          {"smiles": "CCO", "source": "chembl_fda", "chembl_id": "CHEMBL_OK"},
          {"smiles": "O=[As][As+](=O)[O-]", "source": "chembl_fda", "chembl_id": "CHEMBL_BAD"}
        ]
        """,
        encoding="utf-8",
    )

    library = chembl._load_cached_library(cache_path, max_n=10)

    assert [ligand.chembl_id for ligand in library] == ["CHEMBL_OK"]
