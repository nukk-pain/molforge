from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

from contracts.schema import (
    BIOCOMPUTE_CANDIDATES_FIELD,
    BIOCOMPUTE_SCHEMA_VERSION,
    BIOCOMPUTE_SCHEMA_VERSION_FIELD,
    EvidenceItem,
    TargetCandidate,
)


def load_target_candidates(
    path: Path, *, disease: str | None = None
) -> list[TargetCandidate]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Failed to parse TargetCandidate JSON at {path}: {exc}"
        ) from exc

    return load_target_candidates_payload(payload, disease=disease, source=str(path))


def load_target_candidates_payload(
    payload: object,
    *,
    disease: str | None = None,
    source: str = "request payload",
) -> list[TargetCandidate]:
    payload = normalize_target_candidates_payload(payload, source=source)

    if not payload:
        raise ValueError(f"No candidates found in {source}.")

    candidates: list[TargetCandidate] = []
    for index, item in enumerate(payload):
        candidates.append(parse_target_candidate(item, index=index, disease=disease))
    return candidates


def normalize_target_candidates_payload(payload: object, *, source: str) -> list[object]:
    if isinstance(payload, list):
        warnings.warn(
            "TargetCandidate input uses the legacy bare-array shape; prefer "
            f"{{'{BIOCOMPUTE_SCHEMA_VERSION_FIELD}': '{BIOCOMPUTE_SCHEMA_VERSION}', "
            f"'{BIOCOMPUTE_CANDIDATES_FIELD}': [...]}} to detect contract drift.",
            DeprecationWarning,
            stacklevel=3,
        )
        return payload

    if isinstance(payload, dict):
        if BIOCOMPUTE_CANDIDATES_FIELD not in payload:
            raise ValueError(
                "TargetCandidate envelope must include a candidates list."
            )
        schema_version = payload.get(BIOCOMPUTE_SCHEMA_VERSION_FIELD)
        if schema_version != BIOCOMPUTE_SCHEMA_VERSION:
            raise ValueError(
                f"TargetCandidate schema_version mismatch in {source}: expected "
                f"{BIOCOMPUTE_SCHEMA_VERSION}, got {schema_version!r}."
            )
        candidates = payload[BIOCOMPUTE_CANDIDATES_FIELD]
        if not isinstance(candidates, list):
            raise ValueError("TargetCandidate envelope candidates must be a list.")
        return candidates

    raise ValueError(
        "TargetCandidate input must be a legacy bare JSON array or a versioned "
        "envelope with schema_version and candidates."
    )


def parse_target_candidate(
    item: Any, *, index: int, disease: str | None
) -> TargetCandidate:
    if not isinstance(item, dict):
        raise ValueError(f"Candidate at index {index} must be an object.")

    require_keys(
        item, index=index, required_keys=("gene", "score", "evidence", "pathway")
    )

    gene_payload = item["gene"]
    if not isinstance(gene_payload, dict):
        raise ValueError(
            f"Candidate at index {index} has invalid gene payload: expected object."
        )

    gene_symbol = gene_payload.get("symbol")
    if not isinstance(gene_symbol, str) or not gene_symbol.strip():
        raise ValueError(f"Candidate at index {index} is missing gene.symbol.")

    score = item["score"]
    if not isinstance(score, (int, float)):
        raise ValueError(f"Candidate at index {index} has non-numeric score.")

    evidence_payload = item["evidence"]
    if not isinstance(evidence_payload, list):
        raise ValueError(
            f"Candidate at index {index} has invalid evidence: expected list."
        )

    pathway_payload = item["pathway"]
    if not isinstance(pathway_payload, list):
        raise ValueError(
            f"Candidate at index {index} has invalid pathway: expected list."
        )

    evidence = [parse_evidence_item(entry, index=index) for entry in evidence_payload]
    pathway = [str(entry) for entry in pathway_payload]
    extra = {
        key: value
        for key, value in item.items()
        if key not in {"gene", "score", "evidence", "pathway"}
    }

    return TargetCandidate(
        gene=gene_symbol.strip(),
        score=float(score),
        disease=disease,
        ncbi_id=parse_optional_int(gene_payload.get("ncbi_id")),
        uniprot_id=parse_optional_str(gene_payload.get("uniprot_id")),
        evidence=evidence,
        pathway=pathway,
        extra=extra or None,
    )


def parse_evidence_item(entry: Any, *, index: int) -> EvidenceItem:
    if not isinstance(entry, dict):
        raise ValueError(f"Candidate at index {index} has non-object evidence entry.")
    source = entry.get("source")
    description = entry.get("description")
    confidence = entry.get("confidence")
    if not isinstance(source, str) or not source.strip():
        raise ValueError(
            f"Candidate at index {index} has evidence entry without source."
        )
    if not isinstance(description, str) or not description.strip():
        raise ValueError(
            f"Candidate at index {index} has evidence entry without description."
        )
    if not isinstance(confidence, (int, float)):
        raise ValueError(
            f"Candidate at index {index} has evidence entry with non-numeric confidence."
        )
    return EvidenceItem(
        source=source.strip(),
        description=description.strip(),
        confidence=float(confidence),
    )


def require_keys(
    item: dict[str, Any], *, index: int, required_keys: tuple[str, ...]
) -> None:
    missing = [key for key in required_keys if key not in item]
    if missing:
        raise ValueError(
            f"Candidate at index {index} is missing required keys: {', '.join(missing)}."
        )


def parse_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    raise ValueError(
        f"Expected integer or null for ncbi_id, got {type(value).__name__}."
    )


def parse_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    raise ValueError(
        f"Expected string or null for uniprot_id, got {type(value).__name__}."
    )
