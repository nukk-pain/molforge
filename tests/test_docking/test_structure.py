# pyright: reportMissingImports=false
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from contracts.schema import StructureSource  # noqa: E402
from molforge.docking import structure  # noqa: E402
from molforge.docking.structure import MissingStructureError  # noqa: E402


def test_fetch_alphafold_structure_downloads_and_caches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = [
        {
            "entryId": "AF-P61073-2-F1",
            "uniprotAccession": "P61073-2",
            "pdbUrl": "https://alphafold.ebi.ac.uk/files/AF-P61073-2-F1-model_v6.pdb",
            "cifUrl": "https://alphafold.ebi.ac.uk/files/AF-P61073-2-F1-model_v6.cif",
            "bcifUrl": "https://alphafold.ebi.ac.uk/files/AF-P61073-2-F1-model_v6.bcif",
            "latestVersion": 6,
            "gene": "CXCR4",
        },
        {
            "entryId": "AF-P61073-F1",
            "uniprotAccession": "P61073",
            "pdbUrl": "https://alphafold.ebi.ac.uk/files/AF-P61073-F1-model_v6.pdb",
            "cifUrl": "https://alphafold.ebi.ac.uk/files/AF-P61073-F1-model_v6.cif",
            "bcifUrl": "https://alphafold.ebi.ac.uk/files/AF-P61073-F1-model_v6.bcif",
            "latestVersion": 6,
            "gene": "CXCR4",
        },
    ]
    download_calls: list[str] = []
    pdb_text = "\n".join(
        [
            "ATOM      1  CA  ALA A   1      11.104   8.408   7.200  1.00 50.00           C",
            "ATOM      2  CB  ALA A   1      12.104   8.408   7.200  1.00 10.00           C",
            "ATOM      3  CA  GLY A   2      13.104   8.408   7.200  1.00 70.00           C",
        ]
    )

    monkeypatch.setattr(
        structure.afdb, "fetch_prediction_payload", lambda *args, **kwargs: payload
    )

    async def fake_download(url: str) -> str:
        download_calls.append(url)
        return pdb_text

    monkeypatch.setattr(structure, "_download_pdb_text", fake_download)

    result = asyncio.run(
        structure.fetch_alphafold_structure("P61073", cache_dir=tmp_path)
    )
    cached_path = tmp_path / "AF-P61073-F1-model_v4.pdb"

    assert result.gene == "CXCR4"
    assert result.uniprot == "P61073"
    assert result.source is StructureSource.ALPHAFOLD_DB
    assert result.pdb_path == str(cached_path)
    assert result.confidence == pytest.approx(60.0)
    assert download_calls == [
        "https://alphafold.ebi.ac.uk/files/AF-P61073-F1-model_v6.pdb"
    ]
    assert cached_path.read_text(encoding="utf-8") == pdb_text

    second = asyncio.run(
        structure.fetch_alphafold_structure("P61073", cache_dir=tmp_path)
    )
    assert second.pdb_path == str(cached_path)
    assert len(download_calls) == 1


def test_fetch_alphafold_structure_raises_missing_structure_on_404(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def raise_404(*args, **kwargs):
        raise RuntimeError(
            "AlphaFold DB returned HTTP 404 for UniProt accession 'Q99999'."
        )

    monkeypatch.setattr(structure.afdb, "fetch_prediction_payload", raise_404)

    with pytest.raises(MissingStructureError, match="Q99999"):
        asyncio.run(structure.fetch_alphafold_structure("Q99999", cache_dir=tmp_path))


def test_fetch_alphafold_structure_uses_cache_without_network(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_path = tmp_path / "AF-P61073-F1-model_v4.pdb"
    cache_path.write_text(
        "ATOM      1  CA  ALA A   1      11.104   8.408   7.200  1.00 50.00           C\n",
        encoding="utf-8",
    )

    def fail_fetch(*args, **kwargs):
        raise AssertionError("network fetch should not be called when cache exists")

    monkeypatch.setattr(structure, "_fetch_prediction", fail_fetch)

    result = asyncio.run(
        structure.fetch_alphafold_structure("P61073", cache_dir=tmp_path)
    )

    assert result.uniprot == "P61073"
    assert result.pdb_path == str(cache_path)
    assert result.confidence == pytest.approx(50.0)


def test_calculate_mean_plddt_returns_none_without_ca_atoms() -> None:
    pdb_text = (
        "ATOM      1  CB  ALA A   1      11.104   8.408   7.200  1.00 50.00           C"
    )
    assert structure.calculate_mean_plddt(pdb_text) is None
