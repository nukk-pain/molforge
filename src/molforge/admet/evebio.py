from __future__ import annotations

"""Optional EvE Bio drug-target activity reference lookup.

EvE Bio data is used here as a research-only, non-commercial off-target
reference. The upstream dataset is CC BY-NC-SA 4.0; molforge does not bundle or
redistribute the source data and only reads user-provided cache files unless a
caller explicitly opts into live Hugging Face download.
"""

from dataclasses import dataclass
import importlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from contracts.schema import OffTargetHit

EVE_BIO_REPO_ID = "eve-bio/drug-target-activity"
EVE_BIO_FILENAME = "drug-target-activity.parquet"
DEFAULT_EVEBIO_CACHE_DIR = Path("archive/cache/evebio")
REQUIRED_COLUMNS = frozenset(
    {
        "target__gene",
        "target__uniprot_id",
        "compound__smiles",
        "outcome_is_active",
        "outcome_potency_pxc50",
        "viability_flag",
        "frequency_flag",
    }
)

EveBioReader = Callable[[Path], Sequence[Mapping[str, Any]]]


@dataclass(frozen=True, slots=True)
class EveBioActivity:
    target_gene: str
    target_uniprot_id: str | None
    compound_smiles: str
    outcome_is_active: bool
    outcome_potency_pxc50: float | None
    viability_flag: str | None
    frequency_flag: str | None


def load_evebio_rows(
    *,
    cache_path: str | Path | None = None,
    enable_live_download: bool = False,
    reader: EveBioReader | None = None,
) -> list[dict[str, Any]]:
    """Load EvE Bio rows from a local cache or explicit live download.

    The default path is deliberately non-networked. Tests can pass a reader or a
    JSON fixture path; parquet/Hugging Face dependencies are imported lazily only
    when those paths are exercised.
    """

    local_path = Path(cache_path) if cache_path is not None else None
    if local_path is None and enable_live_download:
        local_path = download_evebio_dataset(DEFAULT_EVEBIO_CACHE_DIR)
    if local_path is None:
        default_path = DEFAULT_EVEBIO_CACHE_DIR / EVE_BIO_FILENAME
        if not default_path.exists():
            return []
        local_path = default_path
    if not local_path.exists():
        return []
    active_reader = reader or read_evebio_cache_file
    return [dict(row) for row in active_reader(local_path)]


def read_evebio_cache_file(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            rows = payload.get("rows", [])
        else:
            rows = payload
        if not isinstance(rows, list):
            raise ValueError("EvE Bio JSON cache must contain a list of row objects.")
        return [_expect_mapping(row) for row in rows]
    if suffix in {".parquet", ".pq"}:
        return read_evebio_parquet(path)
    raise ValueError(f"Unsupported EvE Bio cache format: '{path.suffix}'.")


def read_evebio_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        pandas = importlib.import_module("pandas")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Reading EvE Bio parquet cache requires the optional 'evebio' extra "
            "(pandas + pyarrow). Install with `uv sync --extra evebio`."
        ) from exc
    frame = pandas.read_parquet(path)
    records = frame.to_dict(orient="records")
    if not isinstance(records, list):
        raise ValueError("EvE Bio parquet reader returned an unexpected payload.")
    return [_expect_mapping(row) for row in records]


def download_evebio_dataset(cache_dir: str | Path) -> Path:
    try:
        hub = importlib.import_module("huggingface_hub")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Live EvE Bio download requires the optional 'evebio' extra "
            "(huggingface-hub). Install with `uv sync --extra evebio`."
        ) from exc
    downloaded = hub.hf_hub_download(
        repo_id=EVE_BIO_REPO_ID,
        filename=EVE_BIO_FILENAME,
        repo_type="dataset",
        local_dir=str(cache_dir),
    )
    return Path(str(downloaded))


def parse_evebio_activities(rows: Iterable[Mapping[str, Any]]) -> list[EveBioActivity]:
    activities: list[EveBioActivity] = []
    for row in rows:
        validate_evebio_row(row)
        smiles = _optional_str(row["compound__smiles"])
        target_gene = _optional_str(row["target__gene"])
        if smiles is None or target_gene is None or not is_valid_smiles(smiles):
            continue
        activities.append(
            EveBioActivity(
                target_gene=target_gene,
                target_uniprot_id=_optional_str(row["target__uniprot_id"]),
                compound_smiles=smiles,
                outcome_is_active=_coerce_bool(row["outcome_is_active"]),
                outcome_potency_pxc50=_optional_float(row["outcome_potency_pxc50"]),
                viability_flag=_optional_str(row["viability_flag"]),
                frequency_flag=_optional_str(row["frequency_flag"]),
            )
        )
    return activities


def validate_evebio_row(row: Mapping[str, Any]) -> None:
    missing = sorted(REQUIRED_COLUMNS.difference(row.keys()))
    if missing:
        raise ValueError(f"EvE Bio row is missing required columns: {', '.join(missing)}")


def filter_reference_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    target_gene: str | None = None,
    target_uniprot_id: str | None = None,
) -> list[EveBioActivity]:
    activities = parse_evebio_activities(rows)
    normalized_gene = _normalize_key(target_gene)
    normalized_uniprot = _normalize_key(target_uniprot_id)
    if normalized_gene is None and normalized_uniprot is None:
        return activities
    return [
        activity
        for activity in activities
        if (
            normalized_gene is not None
            and _normalize_key(activity.target_gene) == normalized_gene
        )
        or (
            normalized_uniprot is not None
            and _normalize_key(activity.target_uniprot_id) == normalized_uniprot
        )
    ]


def lookup_evebio_off_targets(
    ligand_smiles: str,
    *,
    target_gene: str | None = None,
    target_uniprot_id: str | None = None,
    rows: Iterable[Mapping[str, Any]] | None = None,
    cache_path: str | Path | None = None,
    enable_live_download: bool = False,
    reader: EveBioReader | None = None,
) -> tuple[list[OffTargetHit], dict[str, Any]]:
    normalized_smiles = ligand_smiles.strip()
    if not normalized_smiles:
        raise ValueError("Ligand SMILES is required for EvE Bio lookup.")
    raw_rows = list(rows) if rows is not None else load_evebio_rows(
        cache_path=cache_path,
        enable_live_download=enable_live_download,
        reader=reader,
    )
    references = parse_evebio_activities(raw_rows)
    active_matches = [
        activity
        for activity in references
        if activity.compound_smiles == normalized_smiles and activity.outcome_is_active
    ]
    excluded_primary_targets = [
        activity
        for activity in active_matches
        if is_primary_target_activity(
            activity,
            target_gene=target_gene,
            target_uniprot_id=target_uniprot_id,
        )
    ]
    matched = [
        activity
        for activity in active_matches
        if not is_primary_target_activity(
            activity,
            target_gene=target_gene,
            target_uniprot_id=target_uniprot_id,
        )
    ]
    hits = [activity_to_off_target_hit(activity) for activity in matched]
    return hits, {
        "evebio_reference_row_count": len(references),
        "evebio_active_match_count": len(matched),
        "evebio_primary_target_match_count": len(excluded_primary_targets),
        "evebio_flag_annotations": [
            activity_annotation(activity)
            for activity in matched
            if activity.viability_flag or activity.frequency_flag
        ],
    }


def activity_to_off_target_hit(activity: EveBioActivity) -> OffTargetHit:
    return OffTargetHit(
        ligand_smiles=activity.compound_smiles,
        off_target_gene=activity.target_gene,
        similarity=1.0,
        severity=severity_for_activity(activity),
    )


def severity_for_activity(activity: EveBioActivity) -> str:
    potency = activity.outcome_potency_pxc50
    if potency is not None:
        if potency >= 7.0:
            return "high"
        if potency >= 6.0:
            return "medium"
    if activity.viability_flag or activity.frequency_flag:
        return "medium"
    return "low"


def activity_annotation(activity: EveBioActivity) -> dict[str, str]:
    annotation: dict[str, str] = {"target_gene": activity.target_gene}
    if activity.target_uniprot_id:
        annotation["target_uniprot_id"] = activity.target_uniprot_id
    if activity.outcome_potency_pxc50 is not None:
        annotation["outcome_potency_pxc50"] = str(activity.outcome_potency_pxc50)
    if activity.viability_flag:
        annotation["viability_flag"] = activity.viability_flag
    if activity.frequency_flag:
        annotation["frequency_flag"] = activity.frequency_flag
    return annotation


def is_primary_target_activity(
    activity: EveBioActivity,
    *,
    target_gene: str | None,
    target_uniprot_id: str | None,
) -> bool:
    normalized_gene = _normalize_key(target_gene)
    normalized_uniprot = _normalize_key(target_uniprot_id)
    return (
        normalized_gene is not None
        and _normalize_key(activity.target_gene) == normalized_gene
    ) or (
        normalized_uniprot is not None
        and _normalize_key(activity.target_uniprot_id) == normalized_uniprot
    )


def is_valid_smiles(smiles: str) -> bool:
    try:
        chem = importlib.import_module("rdkit.Chem")
    except ModuleNotFoundError:
        return True
    return chem.MolFromSmiles(smiles) is not None


def _expect_mapping(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("EvE Bio rows must be JSON/parquet objects.")
    return {str(key): item for key, item in value.items()}


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized or normalized.lower() in {"nan", "none", "null"}:
        return None
    return normalized


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    if not isinstance(value, (str, int, float)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "active"}
    return False


def _normalize_key(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().upper()
    return normalized or None
