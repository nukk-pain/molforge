from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from contracts.schema import PipelineRun, RankedCandidate

from .store import (
    MolforgeStore,
    expect_dict,
    parse_ranked_candidate,
    parse_target_candidate,
)


def load_pipeline_run_artifact(path: Path) -> PipelineRun:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Pipeline run artifact must decode to an object.")
    candidates_payload = payload.get("candidates")
    if not isinstance(candidates_payload, list):
        raise ValueError("Pipeline run artifact must include a candidates list.")
    return PipelineRun(
        run_id=str(payload["run_id"]),
        input_target=parse_target_candidate(
            expect_dict(payload["input_target"], "input_target")
        ),
        started_at=str(payload["started_at"]),
        completed_at=None
        if payload.get("completed_at") is None
        else str(payload["completed_at"]),
        candidates=[
            parse_ranked_candidate(expect_dict(candidate, "candidate"))
            for candidate in candidates_payload
        ],
        config_hash=str(payload["config_hash"]),
        schema_version=str(payload.get("schema_version") or ""),
    )


def infer_run_provenance(run: PipelineRun) -> dict[str, object]:
    provenance: dict[str, object] = {
        "run_id": run.run_id,
        "schema_version": run.schema_version,
        "candidate_count": len(run.candidates),
    }
    rescored_from_values = {
        candidate.provenance.get("rescored_from")
        for candidate in run.candidates
        if isinstance(candidate.provenance.get("rescored_from"), str)
    }
    if len(rescored_from_values) == 1:
        provenance["rescored_from"] = next(iter(rescored_from_values))
    total_rescore_cost = sum(
        _coerce_optional_cost(candidate.provenance.get("rescore_cost_estimate_usd"))
        for candidate in run.candidates
    )
    if total_rescore_cost > 0.0:
        provenance["rescore_cost_estimate_usd"] = round(total_rescore_cost, 6)
    return provenance


def persist_pipeline_run(store: MolforgeStore, run: PipelineRun) -> PipelineRun:
    new_run_id = store.create_run(run.input_target, run.config_hash)
    persisted_candidates: list[RankedCandidate] = []
    for candidate in run.candidates:
        updated_provenance = dict(candidate.provenance)
        updated_provenance["run_id"] = new_run_id
        persisted_candidate = replace(candidate, provenance=updated_provenance)
        if (
            persisted_candidate.affinity is not None
            and persisted_candidate.affinity.pose_ref is not None
        ):
            _ = store.save_structure(
                new_run_id, persisted_candidate.affinity.pose_ref.pocket.structure
            )
            _ = store.save_pose(new_run_id, persisted_candidate.affinity.pose_ref)
        _ = store.save_molecule(new_run_id, persisted_candidate.ligand)
        _ = store.save_admet_profile(new_run_id, persisted_candidate.admet)
        _ = store.save_ranking(new_run_id, persisted_candidate)
        persisted_candidates.append(persisted_candidate)
    store.complete_run(new_run_id)
    return store.load_run(new_run_id)


def _coerce_optional_cost(value: object) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(
        "RankedCandidate.provenance.rescore_cost_estimate_usd must be numeric when present."
    )
