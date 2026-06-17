from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from molforge.admet.evebio import (  # noqa: E402
    filter_reference_rows,
    load_evebio_rows,
    lookup_evebio_off_targets,
    parse_evebio_activities,
    read_evebio_parquet,
)


def evebio_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "target__gene": "KCNH2",
        "target__uniprot_id": "Q12809",
        "compound__smiles": "CCO",
        "outcome_is_active": True,
        "outcome_potency_pxc50": 7.4,
        "viability_flag": "",
        "frequency_flag": "frequent",
    }
    row.update(overrides)
    return row


def test_parse_evebio_activities_validates_required_columns_and_skips_empty_smiles() -> None:
    with pytest.raises(ValueError, match="target__gene"):
        parse_evebio_activities([{"compound__smiles": "CCO"}])

    activities = parse_evebio_activities(
        [
            evebio_row(compound__smiles=""),
            evebio_row(compound__smiles="not a smiles"),
            evebio_row(target__gene="VEGFR2"),
        ]
    )

    assert len(activities) == 1
    assert activities[0].target_gene == "VEGFR2"


def test_filter_reference_rows_matches_gene_or_uniprot_case_insensitively() -> None:
    rows = [
        evebio_row(target__gene="KCNH2", target__uniprot_id="Q12809"),
        evebio_row(target__gene="VEGFR2", target__uniprot_id="P35968"),
    ]

    assert [row.target_gene for row in filter_reference_rows(rows, target_gene="vegfr2")] == [
        "VEGFR2"
    ]
    assert [
        row.target_gene
        for row in filter_reference_rows(rows, target_uniprot_id="q12809")
    ] == ["KCNH2"]


def test_lookup_evebio_off_targets_maps_activity_to_contract_hit_and_metadata() -> None:
    hits, metadata = lookup_evebio_off_targets(
        "CCO",
        target_gene="CXCR4",
        rows=[
            evebio_row(outcome_potency_pxc50=7.5),
            evebio_row(target__gene="KCNH2", compound__smiles="CCN"),
            evebio_row(target__gene="KCNH2", outcome_is_active=False),
        ],
    )

    assert len(hits) == 1
    assert hits[0].off_target_gene == "KCNH2"
    assert hits[0].severity == "high"
    assert hits[0].similarity == 1.0
    assert metadata["evebio_reference_row_count"] == 3
    assert metadata["evebio_active_match_count"] == 1
    assert metadata["evebio_primary_target_match_count"] == 0
    assert metadata["evebio_flag_annotations"][0]["outcome_potency_pxc50"] == "7.5"
    assert metadata["evebio_flag_annotations"][0]["frequency_flag"] == "frequent"


def test_lookup_evebio_excludes_primary_target_activity_from_off_target_hits() -> None:
    hits, metadata = lookup_evebio_off_targets(
        "CCO",
        target_gene="KCNH2",
        target_uniprot_id="Q12809",
        rows=[evebio_row()],
    )

    assert hits == []
    assert metadata["evebio_active_match_count"] == 0
    assert metadata["evebio_primary_target_match_count"] == 1


def test_lookup_evebio_activity_flags_raise_medium_severity_without_quantified_potency() -> None:
    hits, _metadata = lookup_evebio_off_targets(
        "CCO",
        rows=[evebio_row(outcome_potency_pxc50="", viability_flag="cytotoxic")],
    )

    assert hits[0].severity == "medium"
    assert hits[0].similarity == 1.0


def test_load_evebio_rows_prefers_local_json_fixture_without_network(tmp_path: Path) -> None:
    cache_path = tmp_path / "evebio.json"
    cache_path.write_text(json.dumps({"rows": [evebio_row()]}), encoding="utf-8")

    rows = load_evebio_rows(cache_path=cache_path)

    assert rows[0]["target__gene"] == "KCNH2"


def test_load_evebio_rows_supports_reader_injection(tmp_path: Path) -> None:
    cache_path = tmp_path / "evebio.parquet"
    cache_path.write_bytes(b"placeholder")

    rows = load_evebio_rows(
        cache_path=cache_path,
        reader=lambda path: [evebio_row(target__gene=path.stem.upper())],
    )

    assert rows[0]["target__gene"] == "EVEBIO"


def test_read_evebio_parquet_reports_missing_optional_dependency(monkeypatch) -> None:
    def fake_import_module(name: str):
        if name == "pandas":
            raise ModuleNotFoundError("No module named 'pandas'")
        raise AssertionError(name)

    monkeypatch.setattr("molforge.admet.evebio.importlib.import_module", fake_import_module)

    with pytest.raises(RuntimeError, match="optional 'evebio' extra"):
        read_evebio_parquet(Path("cache.parquet"))
