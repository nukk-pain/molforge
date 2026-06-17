# pyright: reportMissingImports=false
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from contracts.schema import BIOCOMPUTE_SCHEMA_VERSION
from molforge.core.input import load_target_candidates


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def sample_candidate() -> dict[str, object]:
    return {
        "gene": {"symbol": "TGFB1", "ncbi_id": 7040, "uniprot_id": None},
        "score": 0.87,
        "evidence": [
            {
                "source": "literature",
                "description": "demo evidence",
                "confidence": 0.9,
            }
        ],
        "pathway": ["SMAD3"],
    }


def test_load_target_candidates_valid_array(tmp_path: Path) -> None:
    fixture_path = tmp_path / "targets.json"
    write_json(fixture_path, [sample_candidate()])

    with pytest.warns(DeprecationWarning, match="legacy bare-array"):
        candidates = load_target_candidates(fixture_path, disease="ALS")

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.gene == "TGFB1"
    assert candidate.score == 0.87
    assert candidate.disease == "ALS"
    assert candidate.ncbi_id == 7040
    assert candidate.uniprot_id is None
    assert len(candidate.evidence) == 1
    assert candidate.evidence[0].source == "literature"
    assert candidate.pathway == ["SMAD3"]


def test_load_target_candidates_rejects_empty_array(tmp_path: Path) -> None:
    fixture_path = tmp_path / "targets.json"
    write_json(fixture_path, [])

    with pytest.raises(ValueError, match="No candidates found"):
        _ = load_target_candidates(fixture_path)


def test_load_target_candidates_rejects_missing_gene_symbol(tmp_path: Path) -> None:
    fixture_path = tmp_path / "targets.json"
    bad_candidate = sample_candidate()
    bad_candidate["gene"] = {"ncbi_id": 7040, "uniprot_id": None}
    write_json(fixture_path, [bad_candidate])

    with pytest.raises(ValueError, match="missing gene.symbol"):
        _ = load_target_candidates(fixture_path)


def test_load_target_candidates_accepts_versioned_envelope(tmp_path: Path) -> None:
    fixture_path = tmp_path / "targets.json"
    write_json(
        fixture_path,
        {
            "schema_version": BIOCOMPUTE_SCHEMA_VERSION,
            "candidates": [sample_candidate()],
        },
    )

    candidates = load_target_candidates(fixture_path)

    assert len(candidates) == 1
    assert candidates[0].gene == "TGFB1"


def test_load_target_candidates_rejects_schema_version_mismatch(
    tmp_path: Path,
) -> None:
    fixture_path = tmp_path / "targets.json"
    write_json(
        fixture_path,
        {"schema_version": "1900-01-01", "candidates": [sample_candidate()]},
    )

    with pytest.raises(ValueError, match="schema_version mismatch"):
        _ = load_target_candidates(fixture_path)


def test_load_target_candidates_rejects_unversioned_wrapped_payload(
    tmp_path: Path,
) -> None:
    fixture_path = tmp_path / "targets.json"
    write_json(fixture_path, {"version": "2026-04-17", "candidates": [sample_candidate()]})

    with pytest.raises(ValueError, match="schema_version mismatch"):
        _ = load_target_candidates(fixture_path)


def test_load_target_candidates_rejects_malformed_json(tmp_path: Path) -> None:
    fixture_path = tmp_path / "targets.json"
    fixture_path.write_text('[{"gene":', encoding="utf-8")

    with pytest.raises(ValueError, match="Failed to parse TargetCandidate JSON"):
        _ = load_target_candidates(fixture_path)
