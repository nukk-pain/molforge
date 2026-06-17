from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

off_target_data = importlib.import_module("molforge.admet.off_target_data")


def test_fetch_target_ligands_uses_activity_then_molecule_lookup(
    tmp_path: Path,
    monkeypatch,
) -> None:
    requested_urls: list[str] = []

    def fake_fetch_json(url: str, *, timeout_seconds: float):
        requested_urls.append(url)
        if "activity.json" in url:
            return {
                "activities": [
                    {"molecule_chembl_id": "CHEMBL1"},
                    {"molecule_chembl_id": "CHEMBL1"},
                    {"molecule_chembl_id": "CHEMBL2"},
                ]
            }
        if url.endswith("/CHEMBL1.json"):
            return {"molecule_structures": {"canonical_smiles": "CCO"}}
        if url.endswith("/CHEMBL2.json"):
            return {"molecule_structures": {"canonical_smiles": "CCN"}}
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(off_target_data, "fetch_json", fake_fetch_json)

    ligands = off_target_data.fetch_target_ligands(
        "CHEMBL240",
        cache_dir=tmp_path,
        sleep_seconds=0.0,
        sleep_fn=lambda seconds: None,
    )

    assert ligands == ["CCO", "CCN"]
    assert any("activity.json" in url for url in requested_urls)
    assert any(url.endswith("/CHEMBL1.json") for url in requested_urls)
    assert json.loads((tmp_path / "chembl_off_target_CHEMBL240.json").read_text())[
        "smiles"
    ] == [
        "CCO",
        "CCN",
    ]
