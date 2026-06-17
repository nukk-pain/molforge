from __future__ import annotations

import asyncio
from pathlib import Path

import httpx

from contracts.schema import ProteinStructure, StructureSource

from . import afdb

DEFAULT_AFDB_CACHE_DIR = Path("archive/cache/afdb")
DEFAULT_TIMEOUT_SECONDS = 30.0
MAX_ATTEMPTS = 3


class MissingStructureError(ValueError):
    def __init__(self, gene: str | None, uniprot: str) -> None:
        detail = (
            f"gene '{gene}' / UniProt '{uniprot}'" if gene else f"UniProt '{uniprot}'"
        )
        super().__init__(f"AlphaFold DB structure not available for {detail}.")
        self.gene = gene
        self.uniprot = uniprot


async def fetch_alphafold_structure(
    uniprot: str,
    *,
    cache_dir: Path = DEFAULT_AFDB_CACHE_DIR,
) -> ProteinStructure:
    normalized_uniprot = afdb.normalize_uniprot_accession(uniprot)
    cache_path = cache_dir / f"AF-{normalized_uniprot}-F1-model_v4.pdb"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        pdb_text = cache_path.read_text(encoding="utf-8")
        return ProteinStructure(
            gene=normalized_uniprot,
            uniprot=normalized_uniprot,
            pdb_path=str(cache_path),
            source=StructureSource.ALPHAFOLD_DB,
            confidence=calculate_mean_plddt(pdb_text),
        )

    prediction = await _fetch_prediction(normalized_uniprot)
    gene = prediction.get("gene")
    pdb_url = _require_prediction_url(prediction)
    pdb_text = await _download_pdb_text(pdb_url)
    cache_path.write_text(pdb_text, encoding="utf-8")

    return ProteinStructure(
        gene=gene.strip()
        if isinstance(gene, str) and gene.strip()
        else normalized_uniprot,
        uniprot=normalized_uniprot,
        pdb_path=str(cache_path),
        source=StructureSource.ALPHAFOLD_DB,
        confidence=calculate_mean_plddt(pdb_text),
    )


async def _fetch_prediction(uniprot: str) -> dict[str, object]:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            payload = await asyncio.to_thread(
                afdb.fetch_prediction_payload,
                uniprot,
                DEFAULT_TIMEOUT_SECONDS,
            )
            return afdb.select_prediction(payload, uniprot)
        except ValueError:
            raise
        except RuntimeError as exc:
            if _is_missing_structure_error(exc):
                raise MissingStructureError(None, uniprot) from exc
            if attempt >= MAX_ATTEMPTS:
                raise
            await asyncio.sleep(2 ** (attempt - 1))
    raise RuntimeError("unreachable")


async def _download_pdb_text(url: str) -> str:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_SECONDS) as client:
                response = await client.get(url)
                response.raise_for_status()
            return response.text
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise MissingStructureError(
                    None, _extract_uniprot_from_url(url)
                ) from exc
            if attempt >= MAX_ATTEMPTS:
                raise RuntimeError(
                    f"Failed to download AlphaFold DB PDB from {url}."
                ) from exc
        except httpx.HTTPError as exc:
            if attempt >= MAX_ATTEMPTS:
                raise RuntimeError(
                    f"Failed to download AlphaFold DB PDB from {url}."
                ) from exc
        await asyncio.sleep(2 ** (attempt - 1))
    raise RuntimeError("unreachable")


def calculate_mean_plddt(pdb_text: str) -> float | None:
    b_factors: list[float] = []
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        if line[12:16].strip() != "CA":
            continue
        raw_b_factor = line[60:66].strip()
        if not raw_b_factor:
            continue
        try:
            b_factors.append(float(raw_b_factor))
        except ValueError:
            continue
    if not b_factors:
        return None
    return sum(b_factors) / len(b_factors)


def _is_missing_structure_error(exc: RuntimeError) -> bool:
    return "HTTP 404" in str(exc) or "empty prediction list" in str(exc)


def _extract_uniprot_from_url(url: str) -> str:
    name = url.rsplit("/", 1)[-1]
    if name.startswith("AF-"):
        return name.split("-")[1]
    return name


def _require_prediction_url(prediction: dict[str, object]) -> str:
    value = prediction.get("pdbUrl")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("AlphaFold DB prediction is missing a valid pdbUrl.")
    return value
