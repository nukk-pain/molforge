# pyright: reportMissingImports=false
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from contracts.schema import TargetCandidate  # noqa: E402
from molforge.docking import gene_mapping  # noqa: E402


def build_candidate(
    *, gene: str = "TGFB1", uniprot_id: str | None = None
) -> TargetCandidate:
    return TargetCandidate(
        gene=gene,
        score=0.87,
        disease="ALS",
        ncbi_id=None,
        uniprot_id=uniprot_id,
        evidence=[],
        pathway=[],
        extra=None,
    )


def test_resolve_uniprot_returns_existing_candidate_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def fail_query(*args, **kwargs):
        nonlocal called
        called = True
        return []

    monkeypatch.setattr(gene_mapping, "_query_mygene_hits", fail_query)

    assert (
        gene_mapping.resolve_uniprot(build_candidate(uniprot_id="P01137")) == "P01137"
    )
    assert called is False


def test_resolve_uniprot_reads_swissprot_from_mygene(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gene_mapping._resolve_uniprot_cached.cache_clear()

    def fake_query(gene: str, *, species: str):
        assert species == "human"
        if gene == "TGFB1":
            return [
                {
                    "query": f"symbol:{gene}",
                    "symbol": "TGFB1",
                    "uniprot": {"Swiss-Prot": "P01137"},
                }
            ]
        return [
            {
                "query": f"symbol:{gene}",
                "symbol": "CXCR4",
                "uniprot": {"Swiss-Prot": ["P61073"]},
            }
        ]

    monkeypatch.setattr(gene_mapping, "_query_mygene_hits", fake_query)

    assert gene_mapping.resolve_uniprot(build_candidate(gene="TGFB1")) == "P01137"
    assert gene_mapping.resolve_uniprot(build_candidate(gene="CXCR4")) == "P61073"


def test_resolve_uniprot_returns_none_when_mapping_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gene_mapping._resolve_uniprot_cached.cache_clear()
    monkeypatch.setattr(gene_mapping, "_query_mygene_hits", lambda *args, **kwargs: [])

    assert gene_mapping.resolve_uniprot(build_candidate(gene="UNKNOWN1")) is None


def test_resolve_uniprot_ignores_mismatched_symbol_hits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gene_mapping._resolve_uniprot_cached.cache_clear()
    monkeypatch.setattr(
        gene_mapping,
        "_query_mygene_hits",
        lambda *args, **kwargs: [
            {
                "query": "symbol:MAPT",
                "symbol": "MTP",
                "uniprot": {"Swiss-Prot": "P01137"},
            },
            {
                "query": "symbol:MAPT",
                "symbol": "MAPT",
                "uniprot": {"Swiss-Prot": "P10636"},
            },
        ],
    )

    assert gene_mapping.resolve_uniprot(build_candidate(gene="MAPT")) == "P10636"
