from __future__ import annotations

from functools import lru_cache
from typing import Any

import httpx

from contracts.schema import TargetCandidate

MYGENE_QUERY_URL = "https://mygene.info/v3/query"
DEFAULT_TIMEOUT_SECONDS = 30.0


def resolve_uniprot(
    candidate: TargetCandidate, *, species: str = "human"
) -> str | None:
    if candidate.uniprot_id:
        return candidate.uniprot_id
    return _resolve_uniprot_cached(candidate.gene, species)


@lru_cache(maxsize=256)
def _resolve_uniprot_cached(gene: str, species: str) -> str | None:
    hits = _query_mygene_hits(gene, species=species)
    return _extract_swissprot_accession(hits)


def _query_mygene_hits(gene: str, *, species: str) -> list[dict[str, Any]]:
    response = httpx.get(
        MYGENE_QUERY_URL,
        params={
            "q": f"symbol:{gene}",
            "species": species,
            "fields": "symbol,uniprot",
        },
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    hits = payload.get("hits", [])
    if not isinstance(hits, list):
        return []
    return [item for item in hits if isinstance(item, dict)]


def _extract_swissprot_accession(hits: list[dict[str, Any]]) -> str | None:
    for hit in hits:
        query = hit.get("query")
        symbol = hit.get("symbol")
        if isinstance(query, str) and isinstance(symbol, str):
            if symbol.strip().upper() != query.removeprefix("symbol:").strip().upper():
                continue
        uniprot_payload = hit.get("uniprot")
        accession = _extract_from_uniprot_payload(uniprot_payload)
        if accession:
            return accession
    return None


def _extract_from_uniprot_payload(uniprot_payload: Any) -> str | None:
    if not isinstance(uniprot_payload, dict):
        return None
    swiss_prot = uniprot_payload.get("Swiss-Prot")
    if isinstance(swiss_prot, str):
        normalized = swiss_prot.strip()
        return normalized or None
    if isinstance(swiss_prot, list):
        for entry in swiss_prot:
            if isinstance(entry, str) and entry.strip():
                return entry.strip()
    return None
