from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, cast
from urllib.parse import parse_qs
from wsgiref.simple_server import make_server

from contracts.schema import PipelineRun

from .core.input import load_target_candidates_payload
from .core.pipeline import run_pipeline
from .core.run_io import infer_run_provenance, persist_pipeline_run
from .core.store import MolforgeStore
from .core.writer import serialize
from .docking.rescore import rescore_predictions
from molforge.remote import build_remote_backend

StartResponse = Callable[[str, list[tuple[str, str]]], object]
WSGIApp = Callable[[dict[str, Any], StartResponse], list[bytes]]


def build_app(*, db_path: str | Path = "molforge.db") -> WSGIApp:
    normalized_db_path = str(db_path)

    def app(environ: dict[str, Any], start_response: StartResponse):
        method = str(environ.get("REQUEST_METHOD") or "GET").upper()
        path = str(environ.get("PATH_INFO") or "/")
        query = parse_qs(str(environ.get("QUERY_STRING") or ""), keep_blank_values=True)
        try:
            if method == "GET" and path == "/status":
                return _json_response(
                    start_response,
                    200,
                    {"current_status": _read_current_status()},
                )
            if method == "POST" and path == "/runs":
                payload = _read_json_request(environ)
                return _create_run(
                    payload=payload,
                    query=query,
                    db_path=normalized_db_path,
                    start_response=start_response,
                )
            if (
                method == "POST"
                and path.startswith("/runs/")
                and path.endswith("/rescore")
            ):
                run_id = path.removeprefix("/runs/").removesuffix("/rescore").strip("/")
                if not run_id:
                    return _json_response(
                        start_response,
                        400,
                        {"error": "run_id path segment is required."},
                    )
                return _rescore_run(
                    run_id=run_id,
                    db_path=normalized_db_path,
                    start_response=start_response,
                )
            if method == "GET" and path.startswith("/runs/"):
                run_id = path.removeprefix("/runs/").strip()
                if not run_id:
                    return _json_response(
                        start_response,
                        400,
                        {"error": "run_id path segment is required."},
                    )
                with MolforgeStore(normalized_db_path) as store:
                    try:
                        run = store.load_run(run_id)
                    except ValueError as exc:
                        return _json_response(start_response, 404, {"error": str(exc)})
                return _json_response(start_response, 200, serialize_pipeline_run(run))
            return _json_response(
                start_response,
                404,
                {"error": f"No route for {method} {path}."},
            )
        except ValueError as exc:
            return _json_response(start_response, 400, {"error": str(exc)})
        except FileNotFoundError as exc:
            return _json_response(start_response, 404, {"error": str(exc)})
        except Exception as exc:
            return _json_response(
                start_response, 500, {"error": f"Internal server error: {exc}"}
            )

    return app


def serve(
    *, host: str = "127.0.0.1", port: int = 8000, db_path: str | Path = "molforge.db"
) -> None:
    app = build_app(db_path=db_path)
    with make_server(host, port, app) as server:
        print(f"molforge API serving on http://{host}:{port}")
        server.serve_forever()


def serialize_pipeline_run(run: PipelineRun) -> dict[str, object]:
    return {
        "run_id": run.run_id,
        "started_at": run.started_at,
        "completed_at": run.completed_at,
        "config_hash": run.config_hash,
        "schema_version": run.schema_version,
        "input_target": serialize(run.input_target),
        "candidates": [serialize(candidate) for candidate in run.candidates],
        "candidate_count": len(run.candidates),
        "provenance": infer_run_provenance(run),
    }


def _create_run(
    *,
    payload: object,
    query: dict[str, list[str]],
    db_path: str,
    start_response: StartResponse,
):
    request = _expect_dict(payload, label="run request")
    disease = _optional_str(request.get("disease"))
    raw_top = request.get("top")
    top_n = 10 if raw_top is None else _coerce_int(raw_top, field_name="top")
    enable_evebio = _coerce_bool_option(
        request.get("enable_evebio"),
        query.get("enable_evebio", []),
        field_name="enable_evebio",
    )
    targets = load_target_candidates_payload(
        request.get("targets"),
        disease=disease,
        source="HTTP request body",
    )
    with MolforgeStore(db_path) as store:
        run = run_pipeline(
            targets,
            store=store,
            top_n=top_n,
            enable_evebio=enable_evebio,
        )
    return _json_response(start_response, 201, serialize_pipeline_run(run))


def _rescore_run(*, run_id: str, db_path: str, start_response: StartResponse):
    with MolforgeStore(db_path) as store:
        try:
            run = store.load_run(run_id)
        except ValueError as exc:
            return _json_response(start_response, 404, {"error": str(exc)})
        affinities = [
            candidate.affinity
            for candidate in run.candidates
            if candidate.affinity is not None
        ]
        rescored_affinities, total_cost = rescore_predictions(
            affinities,
            backend=build_remote_backend(),
        )
        rescored_by_smiles = {
            affinity.ligand_smiles: affinity for affinity in rescored_affinities
        }
        rescored_candidates = []
        for candidate in run.candidates:
            affinity = candidate.affinity
            updated_affinity = (
                None if affinity is None else rescored_by_smiles[affinity.ligand_smiles]
            )
            updated_provenance = dict(candidate.provenance)
            updated_provenance["rescored_from"] = run.run_id
            updated_provenance["rescore_cost_estimate_usd"] = round(
                total_cost / max(len(run.candidates), 1), 6
            )
            rescored_candidates.append(
                candidate.__class__(
                    ligand=candidate.ligand,
                    target=candidate.target,
                    affinity=updated_affinity,
                    admet=candidate.admet,
                    off_targets=candidate.off_targets,
                    composite_score=candidate.composite_score,
                    rank=candidate.rank,
                    provenance=updated_provenance,
                )
            )
        persisted_run = persist_pipeline_run(
            store,
            PipelineRun(
                run_id=run.run_id,
                input_target=run.input_target,
                started_at=run.started_at,
                completed_at=run.completed_at,
                candidates=rescored_candidates,
                config_hash=run.config_hash,
                schema_version=run.schema_version,
            ),
        )
    return _json_response(start_response, 201, serialize_pipeline_run(persisted_run))


def _read_json_request(environ: dict[str, Any]) -> object:
    raw_length = environ.get("CONTENT_LENGTH")
    if raw_length in (None, ""):
        content_length = 0
    else:
        content_length = _coerce_int(raw_length, field_name="CONTENT_LENGTH")
    stream = environ.get("wsgi.input")
    if isinstance(stream, BytesIO):
        data = stream.read(content_length)
    elif stream is not None and hasattr(stream, "read"):
        data = cast(Any, stream).read(content_length)
    else:
        data = b""
    if not data:
        raise ValueError("Request body must be non-empty JSON.")
    try:
        return json.loads(data.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON request body: {exc}") from exc


def _json_response(
    start_response: StartResponse, status_code: int, payload: dict[str, object]
):
    body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    status_text = {
        200: "200 OK",
        201: "201 Created",
        400: "400 Bad Request",
        404: "404 Not Found",
        500: "500 Internal Server Error",
    }.get(status_code, f"{status_code} OK")
    start_response(
        status_text,
        [
            ("Content-Type", "application/json; charset=utf-8"),
            ("Content-Length", str(len(body))),
        ],
    )
    return [body]


def _read_current_status() -> str:
    return (
        "molforge is ready. Use `molforge run <input.json>` for the pipeline, "
        "`molforge api` for the HTTP API, and `molforge --help` for commands."
    )


def _expect_dict(payload: object, *, label: str) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError(f"Expected {label} to be a JSON object.")
    return {str(key): value for key, value in payload.items()}


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Expected optional string value, got {type(value).__name__}.")
    normalized = value.strip()
    return normalized or None


def _coerce_int(value: object, *, field_name: str) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value)
    raise ValueError(f"Expected integer-compatible value for {field_name}.")


def _coerce_bool_option(
    body_value: object,
    query_values: list[str],
    *,
    field_name: str,
) -> bool:
    value = body_value
    if value is None and query_values:
        value = query_values[-1]
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    raise ValueError(f"Expected boolean-compatible value for {field_name}.")
