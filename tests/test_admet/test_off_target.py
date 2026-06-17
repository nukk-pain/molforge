from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from contracts.schema import Ligand

off_target_module = importlib.import_module("molforge.admet.off_target")
assess_off_targets = off_target_module.assess_off_targets
scan_off_targets = off_target_module.scan_off_targets


def test_scan_off_targets_returns_contract_safe_hits() -> None:
    hits = scan_off_targets(
        "query",
        {"VEGFR2": ["known-ligand"]},
        threshold=0.5,
        similarity_fn=lambda left, right: 1.0 if right == "known-ligand" else 0.0,
    )

    assert len(hits) == 1
    assert hits[0].off_target_gene == "VEGFR2"
    assert hits[0].severity == "high"


def test_assess_off_targets_adds_class_effect_flags_from_hit_genes() -> None:
    assessment = assess_off_targets(
        Ligand(smiles="CCO", source="user"),
        off_target_map={"VEGFR2": ["known-ligand"]},
        similarity_fn=lambda left, right: 0.9,
    )

    assert assessment.hits[0].off_target_gene == "VEGFR2"
    assert "class_effect:VEGFR2_HTN_risk" in assessment.class_effect_flags


def test_assess_off_targets_reports_live_lookup_failure_honestly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "molforge.admet.off_target.fetch_target_ligands",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("network unavailable")
        ),
    )

    assessment = assess_off_targets(
        Ligand(smiles="CCO", source="user"),
        enable_live_chembl=True,
        similarity_fn=lambda left, right: 0.0,
    )

    assert assessment.metadata["live_lookup_attempted"] is True
    assert assessment.metadata["live_lookup_succeeded"] is False
    assert "network unavailable" in str(assessment.metadata["live_lookup_errors"])


def test_assess_off_targets_merges_evebio_hits_only_when_enabled() -> None:
    row = {
        "target__gene": "KCNH2",
        "target__uniprot_id": "Q12809",
        "compound__smiles": "CCO",
        "outcome_is_active": True,
        "outcome_potency_pxc50": 7.2,
        "viability_flag": "",
        "frequency_flag": "frequent",
    }

    disabled = assess_off_targets(
        Ligand(smiles="CCO", source="user"),
        off_target_map={},
        evebio_rows=[row],
    )
    enabled = assess_off_targets(
        Ligand(smiles="CCO", source="user"),
        off_target_map={},
        enable_evebio=True,
        target_gene="CXCR4",
        evebio_rows=[row],
    )

    assert disabled.hits == []
    assert disabled.metadata["evebio_lookup_attempted"] is False
    assert enabled.metadata["evebio_lookup_attempted"] is True
    assert enabled.metadata["evebio_lookup_succeeded"] is True
    assert enabled.hits[0].off_target_gene == "KCNH2"
    assert enabled.hits[0].severity == "high"


def test_assess_off_targets_excludes_evebio_primary_target_activity() -> None:
    row = {
        "target__gene": "KCNH2",
        "target__uniprot_id": "Q12809",
        "compound__smiles": "CCO",
        "outcome_is_active": True,
        "outcome_potency_pxc50": 7.2,
        "viability_flag": "",
        "frequency_flag": "frequent",
    }

    assessment = assess_off_targets(
        Ligand(smiles="CCO", source="user"),
        off_target_map={},
        enable_evebio=True,
        target_gene="KCNH2",
        target_uniprot_id="Q12809",
        evebio_rows=[row],
    )

    assert assessment.hits == []
    assert assessment.metadata["evebio_primary_target_match_count"] == 1


def test_assess_off_targets_records_evebio_failure_without_failing(monkeypatch) -> None:
    monkeypatch.setattr(
        "molforge.admet.off_target.lookup_evebio_off_targets",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("cache unreadable")),
    )

    assessment = assess_off_targets(
        Ligand(smiles="CCO", source="user"),
        off_target_map={},
        enable_evebio=True,
    )

    assert assessment.hits == []
    assert assessment.metadata["evebio_lookup_succeeded"] is False
    assert "cache unreadable" in assessment.metadata["evebio_lookup_error"]
