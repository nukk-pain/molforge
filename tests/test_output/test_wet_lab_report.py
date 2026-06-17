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

from contracts.schema import (  # noqa: E402
    ADMETProfile,
    AffinityPrediction,
    BindingPocket,
    DockingPose,
    Ligand,
    OffTargetHit,
    PipelineRun,
    ProteinStructure,
    RankedCandidate,
    StructureSource,
    TargetCandidate,
)
from molforge.core.multi_target_batch import BatchResult, BatchTargetResult  # noqa: E402
from molforge.output.wet_lab_report import (  # noqa: E402
    SCHEMA_VERSION,
    build_candidate,
    build_report,
    write_report,
)


def _aspirin_candidate(rank: int = 1) -> RankedCandidate:
    target = TargetCandidate(gene="CXCL12", score=0.75, disease="test pain")
    structure = ProteinStructure(
        gene="CXCL12", uniprot="P48061", pdb_path="/tmp/cxcl12.pdb",
        source=StructureSource.ALPHAFOLD_DB, confidence=88.0,
    )
    pocket = BindingPocket(
        structure=structure, center_xyz=(0.0, 0.0, 0.0), size_xyz=(10.0, 10.0, 10.0),
    )
    smiles = "CC(=O)Oc1ccccc1C(=O)O"  # aspirin
    pose = DockingPose(
        ligand_smiles=smiles, pocket=pocket, pose_pdb_path="/tmp/p.pdb",
        vina_score=-6.3, rank=1,
    )
    return RankedCandidate(
        ligand=Ligand(smiles=smiles, source="chembl_fda", chembl_id="CHEMBL25"),
        target=target,
        affinity=AffinityPrediction(
            ligand_smiles=smiles, target_gene="CXCL12", vina_score=-6.3,
            pose_ref=pose,
        ),
        admet=ADMETProfile(
            ligand_smiles=smiles, endpoints={f"ep_{i}": 0.1 for i in range(41)},
            liability_flags=["PAINS_alert", "BBB_penetrant"],
        ),
        off_targets=[
            OffTargetHit(
                ligand_smiles=smiles, off_target_gene="KCNH2",
                similarity=0.18, severity="low",
            )
        ],
        composite_score=0.642,
        rank=rank,
        provenance={"sa_score": 2.2},
    )


def test_build_candidate_computes_rdkit_descriptors() -> None:
    candidate = build_candidate(
        _aspirin_candidate(),
        batch_id="batch-001",
        molforge_run_id="run-xxx",
        pubchem_lookup=lambda _: 2244,  # aspirin PubChem CID
    )
    # aspirin: MW ~180.16
    assert candidate.physchem["molecular_weight"] == pytest.approx(180.16, abs=0.2)
    assert candidate.physchem["ro5_violations"] == 0
    assert 0.5 <= candidate.physchem["qed"] <= 1.0
    assert "sa_score" in candidate.physchem
    assert candidate.admet_summary["composite_score"] == 0.642
    assert candidate.admet_summary["top_liabilities"] == ["PAINS_alert", "BBB_penetrant"]
    assert candidate.admet_summary["endpoint_count"] == 41
    assert candidate.off_target_warnings[0]["off_target_gene"] == "KCNH2"
    assert candidate.scoring["vina_score"] == -6.3
    assert candidate.scoring["rescored_by"] == "vina-top1"
    assert candidate.identifiers["pubchem_cid"] == 2244
    assert candidate.identifiers["chembl_id"] == "CHEMBL25"
    assert candidate.order_hint["synthesis_risk"] == "low"
    assert candidate.provenance["batch_id"] == "batch-001"


def test_build_candidate_tags_off_target_warnings_with_source() -> None:
    """Codex review: off_target_warnings must disclose whether they came from
    live ChEMBL or the packaged fallback cache."""
    candidate = build_candidate(
        _aspirin_candidate(),
        batch_id="b",
        molforge_run_id="r",
        pubchem_lookup=lambda _: None,
        off_target_source="live_chembl",
    )
    assert candidate.off_target_warnings[0]["source"] == "live_chembl"

    fallback_candidate = build_candidate(
        _aspirin_candidate(),
        batch_id="b",
        molforge_run_id="r",
        pubchem_lookup=lambda _: None,
        # default off_target_source="fallback_cache"
    )
    assert fallback_candidate.off_target_warnings[0]["source"] == "fallback_cache"


def test_build_report_records_live_chembl_flag_in_upstream() -> None:
    good_run = PipelineRun(
        run_id="r", input_target=_aspirin_candidate().target,
        started_at="s", completed_at="c",
        candidates=[_aspirin_candidate()], config_hash="h",
    )
    batch = BatchResult(
        per_target=[
            BatchTargetResult(
                target_gene="CXCL12", status="completed", run=good_run,
                run_id="r", candidate_count=1,
            )
        ],
        total_candidates=1, unique_smiles_count=1,
    )
    report_fallback = build_report(
        batch, batch_id="b", disease_context="demo",
        upstream={}, pubchem_lookup=lambda _: None,
    )
    assert report_fallback.upstream["live_chembl_enabled"] is False
    assert report_fallback.upstream["off_target_warnings_source"] == "fallback_cache"
    assert (
        report_fallback.targets[0].candidates[0].off_target_warnings[0]["source"]
        == "fallback_cache"
    )

    report_live = build_report(
        batch, batch_id="b", disease_context="demo",
        upstream={}, pubchem_lookup=lambda _: None,
        live_chembl_enabled=True,
    )
    assert report_live.upstream["live_chembl_enabled"] is True
    assert report_live.upstream["off_target_warnings_source"] == "live_chembl"


def test_build_candidate_handles_pubchem_lookup_failure_gracefully() -> None:
    candidate = build_candidate(
        _aspirin_candidate(),
        batch_id="b",
        molforge_run_id="r",
        pubchem_lookup=lambda _: None,
    )
    assert candidate.identifiers["pubchem_cid"] is None
    assert candidate.canonical_smiles


def test_build_candidate_marks_boltz_rescored() -> None:
    candidate = _aspirin_candidate()
    # simulate Boltz affinity population
    boosted = RankedCandidate(
        ligand=candidate.ligand, target=candidate.target,
        affinity=AffinityPrediction(
            ligand_smiles=candidate.ligand.smiles, target_gene="CXCL12",
            vina_score=-6.3, affinity_log_ki=-7.2, affinity_confidence=0.81,
        ),
        admet=candidate.admet, off_targets=candidate.off_targets,
        composite_score=candidate.composite_score, rank=candidate.rank,
        provenance=candidate.provenance,
    )
    out = build_candidate(boosted, batch_id="b", molforge_run_id="r",
                         pubchem_lookup=lambda _: None)
    assert out.scoring["rescored_by"] == "boltz-2-reselection"
    assert out.scoring["affinity_log_ki"] == -7.2


def test_build_report_collects_targets_and_failures(tmp_path: Path) -> None:
    good_run = PipelineRun(
        run_id="run-ok", input_target=_aspirin_candidate().target,
        started_at="2026-04-20T00:00:00Z", completed_at="2026-04-20T00:01:00Z",
        candidates=[_aspirin_candidate(rank=1)], config_hash="deadbeef",
    )
    batch = BatchResult(
        per_target=[
            BatchTargetResult(
                target_gene="CXCL12", status="completed", run=good_run,
                run_id="run-ok", candidate_count=1,
            ),
            BatchTargetResult(
                target_gene="CTGF", status="failed:RuntimeError",
                error_message="vina missing",
            ),
        ],
        total_candidates=1,
        unique_smiles_count=1,
    )

    report = build_report(
        batch, batch_id="batch-v3c3", disease_context="myofascial pain syndrome",
        upstream={"biocompute_run_ref": "archive/runs/xyz"},
        pubchem_lookup=lambda _: None,
    )
    assert report.schema_version == SCHEMA_VERSION
    assert len(report.targets) == 1
    assert report.targets[0].target_gene == "CXCL12"
    assert report.targets[0].candidates[0].physchem["molecular_weight"] > 100
    assert len(report.per_target_failures) == 1
    assert report.per_target_failures[0]["target_gene"] == "CTGF"


def test_write_report_round_trip(tmp_path: Path) -> None:
    batch = BatchResult(
        per_target=[
            BatchTargetResult(
                target_gene="CXCL12", status="completed",
                run=PipelineRun(
                    run_id="r", input_target=_aspirin_candidate().target,
                    started_at="s", completed_at="c",
                    candidates=[_aspirin_candidate()], config_hash="h",
                ),
                run_id="r", candidate_count=1,
            ),
        ],
        total_candidates=1, unique_smiles_count=1,
    )
    report = build_report(
        batch, batch_id="b", disease_context="demo",
        upstream={}, pubchem_lookup=lambda _: None,
    )
    output_path = tmp_path / "report.json"
    write_report(report, output_path)
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["targets"][0]["target_gene"] == "CXCL12"
