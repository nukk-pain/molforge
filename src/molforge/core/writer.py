from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

from contracts.schema import PipelineRun

from .run_io import infer_run_provenance


def write_ranked_candidates(run: PipelineRun, path: Path) -> None:
    payload = {
        "run_id": run.run_id,
        "started_at": run.started_at,
        "completed_at": run.completed_at,
        "config_hash": run.config_hash,
        "schema_version": run.schema_version,
        "input_target": serialize(run.input_target),
        "candidates": [serialize(candidate) for candidate in run.candidates],
        "provenance": infer_run_provenance(run),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def serialize(value: object) -> object:
    if hasattr(value, "__dataclass_fields__") and not isinstance(value, type):
        return {str(key): serialize(item) for key, item in vars(value).items()}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, list):
        return [serialize(item) for item in value]
    if isinstance(value, tuple):
        return [serialize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): serialize(item) for key, item in value.items()}
    return value
