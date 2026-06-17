# pyright: reportMissingImports=false
from __future__ import annotations

import json
import sys
from io import BytesIO
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from contracts.schema import (  # noqa: E402
    ADMETProfile,
    AffinityPrediction,
    Ligand,
    OffTargetHit,
    PipelineRun,
    RankedCandidate,
    TargetCandidate,
)
from molforge.api import build_app  # noqa: E402
from molforge.core.store import MolforgeStore  # noqa: E402


def invoke_json(app, method: str, path: str, payload: object | None = None):
    body = b""
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
    environ = {
        "REQUEST_METHOD": method,
        "PATH_INFO": path,
        "QUERY_STRING": "",
        "CONTENT_LENGTH": str(len(body)),
        "wsgi.input": BytesIO(body),
    }
    captured: dict[str, object] = {}

    def start_response(status, headers):
        captured["status"] = status
        captured["headers"] = headers

    chunks = app(environ, start_response)
    response_body = b"".join(chunks).decode("utf-8")
    return str(captured["status"]), json.loads(response_body)


def test_post_runs_executes_pipeline_and_returns_serialized_run(monkeypatch) -> None:
    target = TargetCandidate(
        gene="TGFB1",
        score=0.87,
        disease="ALS",
        ncbi_id=7040,
        uniprot_id=None,
        evidence=[],
        pathway=["SMAD3"],
        extra=None,
    )
    ranked = RankedCandidate(
        ligand=Ligand(smiles="CCO", source="chembl_fda"),
        target=target,
        affinity=AffinityPrediction(
            ligand_smiles="CCO",
            target_gene="TGFB1",
            vina_score=-8.1,
            affinity_log_ki=None,
            affinity_confidence=None,
            pose_ref=None,
        ),
        admet=ADMETProfile(
            ligand_smiles="CCO", endpoints={"hERG": 0.2}, liability_flags=[]
        ),
        off_targets=[
            OffTargetHit(
                ligand_smiles="CCO",
                off_target_gene="KCNH2",
                similarity=0.2,
                severity="low",
            )
        ],
        composite_score=0.8,
        rank=1,
        provenance={"run_id": "run-123", "stage": "phase5_pipeline"},
    )

    observed: dict[str, object] = {}

    def fake_run_pipeline(targets, *, store, top_n=10, enable_evebio=False):
        observed.update({"top_n": top_n, "enable_evebio": enable_evebio})
        _ = targets, store
        return PipelineRun(
            run_id="run-123",
            input_target=target,
            started_at="2026-04-18T00:00:00+00:00",
            completed_at="2026-04-18T00:01:00+00:00",
            candidates=[ranked],
            config_hash="a" * 64,
        )

    monkeypatch.setattr("molforge.api.run_pipeline", fake_run_pipeline)
    app = build_app(db_path="sqlite:///:memory:")

    status, payload = invoke_json(
        app,
        "POST",
        "/runs",
        payload={
            "disease": "ALS",
            "top": 5,
            "targets": [
                {
                    "gene": {"symbol": "TGFB1", "ncbi_id": 7040, "uniprot_id": None},
                    "score": 0.87,
                    "evidence": [],
                    "pathway": ["SMAD3"],
                }
            ],
        },
    )

    assert status == "201 Created"
    assert payload["run_id"] == "run-123"
    assert payload["candidate_count"] == 1
    assert payload["candidates"][0]["ligand"]["smiles"] == "CCO"
    assert observed == {"top_n": 5, "enable_evebio": False}


def test_post_runs_passes_enable_evebio_to_pipeline(monkeypatch) -> None:
    observed: dict[str, object] = {}

    def fake_run_pipeline(targets, *, store, top_n=10, enable_evebio=False):
        observed.update({"target_count": len(targets), "enable_evebio": enable_evebio})
        target = targets[0]
        return PipelineRun(
            run_id="run-evebio",
            input_target=target,
            started_at="2026-04-18T00:00:00+00:00",
            completed_at="2026-04-18T00:01:00+00:00",
            candidates=[],
            config_hash="b" * 64,
        )

    monkeypatch.setattr("molforge.api.run_pipeline", fake_run_pipeline)
    app = build_app(db_path="sqlite:///:memory:")

    status, payload = invoke_json(
        app,
        "POST",
        "/runs",
        payload={
            "enable_evebio": True,
            "targets": [
                {
                    "gene": {"symbol": "KCNH2", "ncbi_id": 3757, "uniprot_id": "Q12809"},
                    "score": 0.77,
                    "evidence": [],
                    "pathway": [],
                }
            ],
        },
    )

    assert status == "201 Created"
    assert payload["run_id"] == "run-evebio"
    assert observed == {"target_count": 1, "enable_evebio": True}


def test_get_status_returns_current_status_markdown() -> None:
    app = build_app(db_path="sqlite:///:memory:")

    status, payload = invoke_json(app, "GET", "/status")

    assert status == "200 OK"
    assert "molforge is ready" in payload["current_status"]


def test_get_run_returns_persisted_run_payload(tmp_path: Path) -> None:
    db_path = tmp_path / "api.db"
    target = TargetCandidate(
        gene="TGFB1",
        score=0.87,
        disease="ALS",
        ncbi_id=7040,
        uniprot_id=None,
        evidence=[],
        pathway=["SMAD3"],
        extra=None,
    )
    ranked = RankedCandidate(
        ligand=Ligand(smiles="CCO", source="chembl_fda"),
        target=target,
        affinity=AffinityPrediction(
            ligand_smiles="CCO",
            target_gene="TGFB1",
            vina_score=-8.1,
            affinity_log_ki=None,
            affinity_confidence=None,
            pose_ref=None,
        ),
        admet=ADMETProfile(
            ligand_smiles="CCO", endpoints={"hERG": 0.2}, liability_flags=[]
        ),
        off_targets=[
            OffTargetHit(
                ligand_smiles="CCO",
                off_target_gene="KCNH2",
                similarity=0.2,
                severity="low",
            )
        ],
        composite_score=0.8,
        rank=1,
        provenance={"stage": "phase5_pipeline"},
    )
    with MolforgeStore(db_path) as store:
        run_id = store.create_run(target, "a" * 64)
        ranked.provenance["run_id"] = run_id
        _ = store.save_admet_profile(run_id, ranked.admet)
        _ = store.save_ranking(run_id, ranked)
        store.complete_run(run_id, "2026-04-18T00:01:00+00:00")

    app = build_app(db_path=db_path)
    status, payload = invoke_json(app, "GET", f"/runs/{run_id}")

    assert status == "200 OK"
    assert payload["run_id"] == run_id
    assert payload["candidate_count"] == 1
    assert payload["completed_at"] == "2026-04-18T00:01:00+00:00"


def test_post_runs_rejects_multiple_targets(monkeypatch) -> None:
    app = build_app(db_path="sqlite:///:memory:")

    status, payload = invoke_json(
        app,
        "POST",
        "/runs",
        payload={
            "targets": [
                {
                    "gene": {"symbol": "TGFB1", "ncbi_id": 7040, "uniprot_id": None},
                    "score": 0.87,
                    "evidence": [],
                    "pathway": ["SMAD3"],
                },
                {
                    "gene": {
                        "symbol": "CXCR4",
                        "ncbi_id": 7852,
                        "uniprot_id": "P61073",
                    },
                    "score": 0.92,
                    "evidence": [],
                    "pathway": ["chemokine"],
                },
            ]
        },
    )

    assert status == "400 Bad Request"
    assert "exactly one TargetCandidate" in payload["error"]


def test_get_missing_run_returns_404_json(tmp_path: Path) -> None:
    app = build_app(db_path=tmp_path / "api.db")

    status, payload = invoke_json(app, "GET", "/runs/missing-run")

    assert status == "404 Not Found"
    assert "missing-run" in payload["error"]


def test_post_run_rescore_creates_new_run(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "api.db"
    target = TargetCandidate(
        gene="TGFB1",
        score=0.87,
        disease="ALS",
        ncbi_id=7040,
        uniprot_id=None,
        evidence=[],
        pathway=["SMAD3"],
        extra=None,
    )
    ranked = RankedCandidate(
        ligand=Ligand(smiles="CCO", source="chembl_fda"),
        target=target,
        affinity=AffinityPrediction(
            ligand_smiles="CCO",
            target_gene="TGFB1",
            vina_score=-8.1,
            affinity_log_ki=None,
            affinity_confidence=None,
            pose_ref=None,
        ),
        admet=ADMETProfile(
            ligand_smiles="CCO", endpoints={"hERG": 0.2}, liability_flags=[]
        ),
        off_targets=[],
        composite_score=0.8,
        rank=1,
        provenance={"run_id": "placeholder", "stage": "phase5_pipeline"},
    )
    with MolforgeStore(db_path) as store:
        run_id = store.create_run(target, "a" * 64)
        ranked.provenance["run_id"] = run_id
        _ = store.save_admet_profile(run_id, ranked.admet)
        _ = store.save_molecule(run_id, ranked.ligand)
        _ = store.save_ranking(run_id, ranked)
        store.complete_run(run_id)

    def fake_rescore_predictions(predictions, *, backend, batch_size=10):
        _ = backend, batch_size
        return [
            AffinityPrediction(
                ligand_smiles=predictions[0].ligand_smiles,
                target_gene=predictions[0].target_gene,
                vina_score=predictions[0].vina_score,
                affinity_log_ki=-9.3,
                affinity_confidence=0.91,
                pose_ref=predictions[0].pose_ref,
            )
        ], 0.6

    monkeypatch.setattr("molforge.api.rescore_predictions", fake_rescore_predictions)
    monkeypatch.setattr("molforge.api.build_remote_backend", lambda: object())

    app = build_app(db_path=db_path)
    status, payload = invoke_json(app, "POST", f"/runs/{run_id}/rescore")

    assert status == "201 Created"
    assert payload["run_id"] != run_id
    assert payload["provenance"]["rescored_from"] == run_id
    assert payload["candidates"][0]["affinity"]["affinity_log_ki"] == -9.3
