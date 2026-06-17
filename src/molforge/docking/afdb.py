from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib import error, request

AFDB_BASE_URL = "https://alphafold.ebi.ac.uk"
AFDB_PREDICTION_PATH = "/api/prediction/{uniprot_accession}"
DEFAULT_TIMEOUT_SECONDS = 30.0
TASK_NAME = "P0-5 AlphaFold DB API contract and PoseBench source validation"
REQUEST_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "molforge-phase0-afdb-smoke/0.1",
}
POSEBENCH_SAMPLE_SOURCE_URL = (
    "https://zenodo.org/records/19138652/files/posebusters_benchmark_set.tar.gz"
)
UNIPROT_ACCESSION_PATTERN = re.compile(
    r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})(?:-\d+)?$"
)
REQUIRED_PREDICTION_FIELDS = (
    "entryId",
    "uniprotAccession",
    "pdbUrl",
    "cifUrl",
    "bcifUrl",
    "latestVersion",
)


@dataclass(frozen=True, slots=True)
class AFDBNormalizedPrediction:
    source: str
    api_url: str
    requested_uniprot_accession: str
    uniprot_accession: str
    entry_id: str
    latest_version: int
    gene: str | None
    pdb_url: str
    cif_url: str
    bcif_url: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_uniprot_accession(uniprot_accession: str) -> str:
    accession = uniprot_accession.strip().upper()
    if not accession:
        raise ValueError("UniProt accession is required for AlphaFold DB lookups.")
    if not UNIPROT_ACCESSION_PATTERN.fullmatch(accession):
        raise ValueError(
            f"Invalid UniProt accession for AlphaFold DB lookup: '{uniprot_accession}'."
        )
    return accession


def build_prediction_url(uniprot_accession: str) -> str:
    accession = normalize_uniprot_accession(uniprot_accession)
    return AFDB_BASE_URL + AFDB_PREDICTION_PATH.format(uniprot_accession=accession)


def fetch_prediction_payload(
    uniprot_accession: str,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    accession = normalize_uniprot_accession(uniprot_accession)
    prediction_url = build_prediction_url(accession)
    http_request = request.Request(prediction_url, headers=REQUEST_HEADERS)

    try:
        with request.urlopen(http_request, timeout=timeout_seconds) as response:
            status_code = response.getcode()
            response_text = response.read().decode("utf-8")
    except error.HTTPError as exc:
        raise RuntimeError(
            f"AlphaFold DB returned HTTP {exc.code} for UniProt accession '{accession}'."
        ) from exc
    except error.URLError as exc:
        raise RuntimeError(
            f"AlphaFold DB request failed for UniProt accession '{accession}': {exc.reason}"
        ) from exc

    if status_code != 200:
        raise RuntimeError(
            f"AlphaFold DB returned unexpected HTTP {status_code} for UniProt accession '{accession}'."
        )

    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"AlphaFold DB returned invalid JSON for UniProt accession '{accession}'."
        ) from exc

    if not isinstance(payload, list):
        raise ValueError(
            "AlphaFold DB prediction payload must be a JSON array of prediction objects."
        )
    if not payload:
        raise ValueError(
            f"AlphaFold DB returned an empty prediction list for UniProt accession '{accession}'."
        )
    if any(not isinstance(item, dict) for item in payload):
        raise ValueError(
            "AlphaFold DB prediction payload must contain only JSON objects."
        )

    return payload


def normalize_prediction_payload(
    uniprot_accession: str,
    payload: list[dict[str, Any]],
) -> dict[str, Any]:
    accession = normalize_uniprot_accession(uniprot_accession)
    prediction = select_prediction(payload, accession)
    return normalize_prediction(prediction, accession).to_dict()


def select_prediction(
    payload: list[dict[str, Any]],
    requested_uniprot_accession: str,
) -> dict[str, Any]:
    accession = normalize_uniprot_accession(requested_uniprot_accession)
    exact_matches = [
        item for item in payload if item.get("uniprotAccession") == accession
    ]
    if not exact_matches:
        raise ValueError(
            f"AlphaFold DB returned predictions, but none matched the requested UniProt accession '{accession}'."
        )

    prediction = max(
        exact_matches, key=lambda item: coerce_latest_version(item.get("latestVersion"))
    )

    for field_name in REQUIRED_PREDICTION_FIELDS:
        if field_name not in prediction:
            raise ValueError(
                f"AlphaFold DB prediction for '{accession}' is missing required field '{field_name}'."
            )

    for url_field in ("entryId", "uniprotAccession", "pdbUrl", "cifUrl", "bcifUrl"):
        field_value = prediction.get(url_field)
        if not isinstance(field_value, str) or not field_value.strip():
            raise ValueError(
                f"AlphaFold DB prediction for '{accession}' has invalid field '{url_field}'."
            )

    coerce_latest_version(prediction.get("latestVersion"))
    return prediction


def normalize_prediction(
    prediction: dict[str, Any],
    requested_uniprot_accession: str,
) -> AFDBNormalizedPrediction:
    accession = normalize_uniprot_accession(requested_uniprot_accession)
    entry_accession = prediction["uniprotAccession"]
    if entry_accession != accession:
        raise ValueError(
            f"AlphaFold DB normalized prediction mismatch: requested '{accession}', got '{entry_accession}'."
        )

    return AFDBNormalizedPrediction(
        source="alphafold_db",
        api_url=build_prediction_url(accession),
        requested_uniprot_accession=accession,
        uniprot_accession=entry_accession,
        entry_id=prediction["entryId"],
        latest_version=coerce_latest_version(prediction["latestVersion"]),
        gene=coerce_optional_string(prediction.get("gene")),
        pdb_url=prediction["pdbUrl"],
        cif_url=prediction["cifUrl"],
        bcif_url=prediction["bcifUrl"],
    )


def run_smoke_slice(
    uniprot_accession: str,
    output_dir: str | Path,
    posebench_source: str = POSEBENCH_SAMPLE_SOURCE_URL,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    error_path = output_path / "task-error.log"

    try:
        payload = fetch_prediction_payload(
            uniprot_accession=uniprot_accession,
            timeout_seconds=timeout_seconds,
        )
        normalized_prediction = normalize_prediction_payload(uniprot_accession, payload)
    except Exception as exc:
        write_task_error(output_path, uniprot_accession, exc)
        raise

    if error_path.exists():
        error_path.unlink()

    raw_prediction_path = output_path / "raw_prediction.json"
    normalized_prediction_path = output_path / "normalized_prediction.json"
    posebench_source_path = output_path / "posebench-source.txt"
    command_log_path = output_path / "command.log"
    task_path = output_path / "task.json"

    raw_prediction_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    normalized_prediction_path.write_text(
        json.dumps(normalized_prediction, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    posebench_source_path.write_text(posebench_source.rstrip() + "\n", encoding="utf-8")
    write_command_log(
        path=command_log_path,
        requested_uniprot_accession=normalize_uniprot_accession(uniprot_accession),
        timeout_seconds=timeout_seconds,
        posebench_source=posebench_source,
    )

    task_payload = {
        "status": "pass",
        "source": "alphafold_db",
        "requested_uniprot_accession": normalize_uniprot_accession(uniprot_accession),
        "api_url": build_prediction_url(uniprot_accession),
        "command_log_path": str(command_log_path),
        "normalized_prediction_path": str(normalized_prediction_path),
        "raw_prediction_path": str(raw_prediction_path),
        "posebench_source_path": str(posebench_source_path),
        "error_log_path": str(error_path),
    }
    task_path.write_text(
        json.dumps(task_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return task_payload


def write_task_error(
    output_dir: str | Path,
    uniprot_accession: str,
    exc: Exception,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    error_path = output_path / "task-error.log"
    error_path.write_text(
        "\n".join(
            [
                f"requested_uniprot_accession={uniprot_accession}",
                f"error_type={type(exc).__name__}",
                f"error_message={exc}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return error_path


def write_command_log(
    path: Path,
    requested_uniprot_accession: str,
    timeout_seconds: float,
    posebench_source: str,
) -> None:
    api_url = build_prediction_url(requested_uniprot_accession)
    lines = [
        f"task={TASK_NAME}",
        "source=alphafold_db",
        "execution_mode=urllib.request.urlopen",
        f"http_method=GET",
        f"request_url={api_url}",
        f"request_timeout_seconds={timeout_seconds}",
        f"request_accept={REQUEST_HEADERS['Accept']}",
        f"request_user_agent={REQUEST_HEADERS['User-Agent']}",
        f"requested_uniprot_accession={requested_uniprot_accession}",
        f"posebench_source={posebench_source}",
        "step=fetch-afdb-prediction",
        f"command=GET {api_url}",
        "step=normalize-afdb-prediction",
        f"command=normalize_prediction_payload('{requested_uniprot_accession}', raw_prediction_payload)",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def coerce_latest_version(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("AlphaFold DB latestVersion must be an integer.")
    if isinstance(value, int):
        latest_version = value
    elif isinstance(value, str) and value.isdigit():
        latest_version = int(value)
    else:
        raise ValueError("AlphaFold DB latestVersion must be an integer.")

    if latest_version <= 0:
        raise ValueError("AlphaFold DB latestVersion must be positive.")
    return latest_version


def coerce_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped_value = value.strip()
        return stripped_value or None
    raise ValueError(
        "AlphaFold DB optional string fields must be strings when present."
    )
