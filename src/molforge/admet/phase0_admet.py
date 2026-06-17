from __future__ import annotations

import importlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

TASK_NAME = "P0-4 ADMET-AI v2 41-endpoint smoke verification"
SOURCE_NAME = "admet_ai_v2"
SMOKE_SMILES = "CCO"
EXPECTED_ENDPOINT_COUNT = 41
AUTHORITATIVE_INSTALL_COMMAND = "pip install admet_ai"
REPO_ROOT = Path(__file__).resolve().parents[3]
EXTERNAL_ADMET_PYTHONS = (REPO_ROOT / ".uv" / "phase0-admet-311" / "bin" / "python",)
NON_ENDPOINT_KEYS = frozenset(
    {
        "smiles",
        "SMILES",
        "canonical_smiles",
        "molecule",
        "mol",
        "compound_id",
        "id",
        "inchi",
        "inchikey",
        "molecular_weight",
        "logP",
        "hydrogen_bond_acceptors",
        "hydrogen_bond_donors",
        "Lipinski",
        "QED",
        "stereo_centers",
        "tpsa",
        "PAINS_alert",
        "BRENK_alert",
        "NIH_alert",
    }
)
SUCCESS_ARTIFACT_NAMES = ("predictions.json", "endpoint-keys.txt")
COMMAND_LOG_FILENAME = "command.log"
LAST_PREDICTION_RUNTIME: dict[str, str] = {}


@dataclass(frozen=True, slots=True)
class SmokeSummary:
    task: str
    source: str
    status: str
    smiles: str
    expected_endpoint_count: int
    observed_endpoint_count: int
    endpoint_count_matches_expectation: bool
    predictions_available: bool
    blocker: bool
    failure_kind: str | None
    error_type: str | None
    error_message: str | None
    command_log_path: str
    predictions_path: str
    endpoint_keys_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CommandStepRecord:
    step: str
    command: str
    duration_seconds: float
    outcome: str
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_smoke_slice(smiles: str, output_dir: str | Path) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    normalized_smiles = normalize_smiles(smiles)
    summary_path = output_path / "summary.json"
    task_path = output_path / "task.json"
    error_path = output_path / "task-error.log"
    predictions_path = output_path / "predictions.json"
    endpoint_keys_path = output_path / "endpoint-keys.txt"
    command_log_path = output_path / COMMAND_LOG_FILENAME
    command_steps: list[CommandStepRecord] = []

    try:
        prediction_output = record_command_step(
            command_steps,
            step="predict_smiles",
            command=f"molforge.admet.phase0_admet.predict_smiles({normalized_smiles!r})",
            action=lambda: predict_smiles(normalized_smiles),
        )
        prediction_map = record_command_step(
            command_steps,
            step="normalize_prediction_map",
            command=(
                "molforge.admet.phase0_admet.normalize_prediction_map("
                f"{normalized_smiles!r}, prediction_output)"
            ),
            action=lambda: normalize_prediction_map(
                normalized_smiles, prediction_output
            ),
        )
        endpoint_keys = record_command_step(
            command_steps,
            step="collect_endpoint_keys",
            command="molforge.admet.phase0_admet.collect_endpoint_keys(prediction_map)",
            action=lambda: collect_endpoint_keys(prediction_map),
        )
    except Exception as exc:
        remove_artifacts(output_path, SUCCESS_ARTIFACT_NAMES)
        failure_kind = classify_failure(exc)
        write_blocked_evidence_artifacts(
            predictions_path=predictions_path,
            endpoint_keys_path=endpoint_keys_path,
            smiles=normalized_smiles,
            failure_kind=failure_kind,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        write_task_error(error_path, normalized_smiles, exc)

        summary = SmokeSummary(
            task=TASK_NAME,
            source=SOURCE_NAME,
            status="blocked",
            smiles=normalized_smiles,
            expected_endpoint_count=EXPECTED_ENDPOINT_COUNT,
            observed_endpoint_count=0,
            endpoint_count_matches_expectation=False,
            predictions_available=False,
            blocker=True,
            failure_kind=failure_kind,
            error_type=type(exc).__name__,
            error_message=str(exc),
            command_log_path=str(command_log_path),
            predictions_path=str(predictions_path),
            endpoint_keys_path=str(endpoint_keys_path),
        )
        write_command_log(
            path=command_log_path,
            smiles=normalized_smiles,
            command_steps=command_steps,
            status=summary.status,
            failure_kind=summary.failure_kind,
            error_message=summary.error_message,
        )
        summary_path.write_text(
            json.dumps(summary.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        task_payload = {
            "task": TASK_NAME,
            "source": SOURCE_NAME,
            "status": "blocked",
            "smiles": normalized_smiles,
            "expected_endpoint_count": EXPECTED_ENDPOINT_COUNT,
            "observed_endpoint_count": 0,
            "summary_path": str(summary_path),
            "task_path": str(task_path),
            "error_log_path": str(error_path),
            "predictions_path": str(predictions_path),
            "endpoint_keys_path": str(endpoint_keys_path),
            "command_log_path": str(command_log_path),
        }
        task_path.write_text(
            json.dumps(task_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return task_payload

    if error_path.exists():
        error_path.unlink()

    predictions_path.write_text(
        json.dumps(prediction_map, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    endpoint_keys_path.write_text(
        "\n".join(endpoint_keys) + "\n",
        encoding="utf-8",
    )

    observed_endpoint_count = len(endpoint_keys)
    endpoint_count_matches_expectation = (
        observed_endpoint_count == EXPECTED_ENDPOINT_COUNT
    )
    if endpoint_count_matches_expectation:
        status = "pass"
        blocker = False
        failure_kind = None
        error_type = None
        error_message = None
    else:
        status = "blocked"
        blocker = True
        failure_kind = "endpoint_count_mismatch"
        error_type = None
        error_message = (
            "ADMET-AI v2 smoke run completed, but the observed endpoint count "
            f"({observed_endpoint_count}) did not match the expected {EXPECTED_ENDPOINT_COUNT}."
        )
        write_endpoint_count_mismatch_error(
            error_path=error_path,
            smiles=normalized_smiles,
            observed_endpoint_count=observed_endpoint_count,
        )

    summary = SmokeSummary(
        task=TASK_NAME,
        source=SOURCE_NAME,
        status=status,
        smiles=normalized_smiles,
        expected_endpoint_count=EXPECTED_ENDPOINT_COUNT,
        observed_endpoint_count=observed_endpoint_count,
        endpoint_count_matches_expectation=endpoint_count_matches_expectation,
        predictions_available=True,
        blocker=blocker,
        failure_kind=failure_kind,
        error_type=error_type,
        error_message=error_message,
        command_log_path=str(command_log_path),
        predictions_path=str(predictions_path),
        endpoint_keys_path=str(endpoint_keys_path),
    )
    write_command_log(
        path=command_log_path,
        smiles=normalized_smiles,
        command_steps=command_steps,
        status=summary.status,
        failure_kind=summary.failure_kind,
        error_message=summary.error_message,
    )
    summary_path.write_text(
        json.dumps(summary.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    task_payload = {
        "task": TASK_NAME,
        "source": SOURCE_NAME,
        "status": status,
        "smiles": normalized_smiles,
        "expected_endpoint_count": EXPECTED_ENDPOINT_COUNT,
        "observed_endpoint_count": observed_endpoint_count,
        "summary_path": str(summary_path),
        "task_path": str(task_path),
        "error_log_path": str(error_path),
        "predictions_path": str(predictions_path),
        "endpoint_keys_path": str(endpoint_keys_path),
        "command_log_path": str(command_log_path),
    }
    task_path.write_text(
        json.dumps(task_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return task_payload


def normalize_smiles(smiles: str) -> str:
    normalized_smiles = smiles.strip()
    if not normalized_smiles:
        raise ValueError("SMILES is required for the ADMET smoke slice.")
    return normalized_smiles


def predict_smiles(smiles: str) -> Any:
    try:
        model_class = load_admet_model_class()
    except RuntimeError as exc:
        external_python = resolve_external_admet_python()
        if external_python is None:
            raise exc
        set_last_prediction_runtime(
            mode="external_python",
            runtime_python=str(external_python),
        )
        payload = predict_smiles_with_external_python(smiles, external_python)
        return payload

    model = model_class()
    payload = model.predict(smiles)
    set_last_prediction_runtime(mode="in_process", runtime_python="current_runtime")
    return payload


def load_admet_model_class() -> type[Any]:
    try:
        admet_module = importlib.import_module("admet_ai")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "ADMET-AI v2 is unavailable in the current runtime: missing Python package 'admet_ai'."
        ) from exc

    ADMETModel = getattr(admet_module, "ADMETModel", None)
    if ADMETModel is None:
        raise RuntimeError(
            "ADMET-AI v2 import succeeded but 'ADMETModel' was not exposed by the package."
        )

    return ADMETModel


def normalize_prediction_map(smiles: str, prediction_output: Any) -> dict[str, Any]:
    candidate = prediction_output

    if hasattr(candidate, "to_dict"):
        try:
            candidate = candidate.to_dict(orient="records")
        except TypeError:
            candidate = candidate.to_dict()

    if isinstance(candidate, list):
        if len(candidate) != 1:
            raise ValueError(
                "ADMET-AI v2 smoke output must contain exactly one prediction record."
            )
        candidate = candidate[0]

    if not isinstance(candidate, dict) or not candidate:
        raise ValueError(
            "ADMET-AI v2 returned an empty or unsupported prediction payload."
        )

    normalized_prediction = {
        str(key): coerce_json_value(value) for key, value in candidate.items()
    }

    smiles_in_payload = normalized_prediction.get("smiles")
    if isinstance(smiles_in_payload, str) and smiles_in_payload.strip() != smiles:
        raise ValueError(
            "ADMET-AI v2 prediction payload SMILES did not match the requested smoke SMILES."
        )

    if not collect_endpoint_keys(normalized_prediction):
        raise ValueError("ADMET-AI v2 returned no endpoint keys for the smoke SMILES.")

    return normalized_prediction


def collect_endpoint_keys(prediction_map: dict[str, Any]) -> list[str]:
    return sorted(
        key
        for key in prediction_map
        if key not in NON_ENDPOINT_KEYS
        and not key.startswith("_")
        and not key.endswith("_drugbank_approved_percentile")
    )


def classify_failure(exc: Exception) -> str:
    message = str(exc)
    if "admet_ai" in message or (
        isinstance(exc, ModuleNotFoundError) and exc.name == "admet_ai"
    ):
        return "missing_dependency"
    return "runtime_error"


def record_command_step(
    command_steps: list[CommandStepRecord],
    step: str,
    command: str,
    action: Any,
) -> Any:
    started_at = perf_counter()
    try:
        result = action()
    except Exception as exc:
        command_steps.append(
            CommandStepRecord(
                step=step,
                command=command,
                duration_seconds=perf_counter() - started_at,
                outcome="error",
                detail=f"{type(exc).__name__}: {exc}",
            )
        )
        raise

    command_steps.append(
        CommandStepRecord(
            step=step,
            command=command,
            duration_seconds=perf_counter() - started_at,
            outcome="success",
        )
    )
    return result


def write_command_log(
    path: Path,
    smiles: str,
    command_steps: list[CommandStepRecord],
    status: str,
    failure_kind: str | None,
    error_message: str | None,
) -> None:
    runtime_info = get_last_prediction_runtime()
    lines = [
        f"authoritative_install_command={AUTHORITATIVE_INSTALL_COMMAND}",
        "install_command_executed=unknown",
        f"probe_execution_mode={runtime_info.get('mode', 'in_process')}",
        f"runtime_python={runtime_info.get('runtime_python', 'current_runtime')}",
        (
            "planned_probe_path="
            f"import admet_ai -> ADMETModel() -> ADMETModel.predict({smiles!r})"
        ),
        f"final_status={status}",
        f"failure_kind={failure_kind or 'none'}",
        f"error_message={error_message or 'none'}",
        "",
    ]
    for record in command_steps:
        lines.extend(
            [
                f"step={record.step}",
                f"command={record.command}",
                f"duration_seconds={record.duration_seconds:.6f}",
                f"outcome={record.outcome}",
                f"detail={record.detail or 'none'}",
                "",
            ]
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_task_error(error_path: Path, smiles: str, exc: Exception) -> None:
    error_path.write_text(
        "\n".join(
            [
                f"task={TASK_NAME}",
                f"smiles={smiles}",
                f"failure_kind={classify_failure(exc)}",
                f"error_type={type(exc).__name__}",
                f"error_message={exc}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_blocked_evidence_artifacts(
    predictions_path: Path,
    endpoint_keys_path: Path,
    *,
    smiles: str,
    failure_kind: str,
    error_type: str,
    error_message: str,
) -> None:
    blocked_prediction_payload = {
        "task": TASK_NAME,
        "source": SOURCE_NAME,
        "status": "blocked",
        "failure_kind": failure_kind,
        "error_type": error_type,
        "error_message": error_message,
        "smiles": smiles,
        "predictions_available": False,
    }
    predictions_path.write_text(
        json.dumps(blocked_prediction_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    endpoint_keys_path.write_text(
        "\n".join(
            [
                "# blocked evidence file: no endpoint keys were produced.",
                f"# task={TASK_NAME}",
                f"# source={SOURCE_NAME}",
                "# status=blocked",
                f"# failure_kind={failure_kind}",
                f"# smiles={smiles}",
                f"# error_type={error_type}",
                f"# error_message={error_message}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_endpoint_count_mismatch_error(
    error_path: Path,
    smiles: str,
    observed_endpoint_count: int,
) -> None:
    error_path.write_text(
        "\n".join(
            [
                f"task={TASK_NAME}",
                f"smiles={smiles}",
                "failure_kind=endpoint_count_mismatch",
                f"expected_endpoint_count={EXPECTED_ENDPOINT_COUNT}",
                f"observed_endpoint_count={observed_endpoint_count}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def remove_artifacts(output_dir: Path, artifact_names: tuple[str, ...]) -> None:
    for artifact_name in artifact_names:
        artifact_path = output_dir / artifact_name
        if artifact_path.exists():
            artifact_path.unlink()


def coerce_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): coerce_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [coerce_json_value(item) for item in value]
    if hasattr(value, "item"):
        coerced_value = value.item()
        return coerce_json_value(coerced_value)
    return str(value)


def resolve_external_admet_python() -> Path | None:
    for candidate in EXTERNAL_ADMET_PYTHONS:
        if candidate.exists():
            return candidate
    return None


def predict_smiles_with_external_python(smiles: str, python_executable: Path) -> Any:
    script = "\n".join(
        [
            "import contextlib, io, json, sys",
            "from admet_ai import ADMETModel",
            "smiles = sys.argv[1]",
            "capture = io.StringIO()",
            "with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):",
            "    model = ADMETModel()",
            "    preds = model.predict(smiles=smiles)",
            "if hasattr(preds, 'to_dict'):",
            "    try:",
            "        preds = preds.to_dict(orient='records')",
            "    except TypeError:",
            "        preds = preds.to_dict()",
            "print(json.dumps(preds))",
        ]
    )
    completed = subprocess.run(
        [str(python_executable), "-c", script, smiles],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "ADMET-AI v2 external runtime probe failed: "
            + (completed.stderr.strip() or completed.stdout.strip() or "unknown error")
        )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "ADMET-AI v2 external runtime returned invalid JSON."
        ) from exc


def set_last_prediction_runtime(*, mode: str, runtime_python: str) -> None:
    LAST_PREDICTION_RUNTIME.clear()
    LAST_PREDICTION_RUNTIME.update(
        {
            "mode": mode,
            "runtime_python": runtime_python,
        }
    )


def get_last_prediction_runtime() -> dict[str, str]:
    if not LAST_PREDICTION_RUNTIME:
        return {"mode": "in_process", "runtime_python": "current_runtime"}
    return dict(LAST_PREDICTION_RUNTIME)
