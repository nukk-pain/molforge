from __future__ import annotations

from dataclasses import dataclass, field
import importlib
from pathlib import Path
from typing import Any, Callable

from contracts.schema import Ligand, OffTargetHit

from .class_effects import class_effect_flags_for_hits
from .evebio import EveBioReader, lookup_evebio_off_targets
from .off_target_data import fetch_target_ligands, load_off_target_definitions

SimilarityFunction = Callable[[str, str], float]


@dataclass(frozen=True, slots=True)
class OffTargetAssessment:
    hits: list[OffTargetHit]
    metadata: dict[str, Any] = field(default_factory=dict)
    class_effect_flags: list[str] = field(default_factory=list)


def assess_off_targets(
    ligand: Ligand,
    *,
    enable_live_chembl: bool = False,
    enable_evebio: bool = False,
    target_gene: str | None = None,
    target_uniprot_id: str | None = None,
    off_target_map: dict[str, list[str]] | None = None,
    similarity_threshold: float = 0.5,
    similarity_fn: SimilarityFunction | None = None,
    cache_dir: str | Path = "archive/cache",
    evebio_cache_path: str | Path | None = None,
    evebio_rows: list[dict[str, Any]] | None = None,
    evebio_reader: EveBioReader | None = None,
    enable_evebio_live_download: bool = False,
) -> OffTargetAssessment:
    normalized_smiles = ligand.smiles.strip()
    if not normalized_smiles:
        raise ValueError("Ligand SMILES is required for off-target assessment.")

    metadata: dict[str, Any] = {
        "live_lookup_attempted": enable_live_chembl,
        "live_lookup_succeeded": False,
        "evebio_lookup_attempted": enable_evebio,
        "evebio_lookup_succeeded": False,
    }
    target_map = off_target_map or {
        entry.gene: list(entry.fallback_reference_smiles)
        for entry in load_off_target_definitions()
    }

    if enable_live_chembl:
        live_successes = 0
        for entry in load_off_target_definitions():
            try:
                target_map[entry.gene] = fetch_target_ligands(
                    entry.target_chembl_id,
                    cache_dir=cache_dir,
                )
            except Exception as exc:
                metadata.setdefault("live_lookup_errors", {})[entry.gene] = (
                    f"{type(exc).__name__}: {exc}"
                )
            else:
                live_successes += 1
        metadata["live_lookup_succeeded"] = live_successes > 0

    try:
        hits = scan_off_targets(
            normalized_smiles,
            target_map,
            threshold=similarity_threshold,
            similarity_fn=similarity_fn,
        )
    except RuntimeError as exc:
        metadata["similarity_backend_error"] = f"{type(exc).__name__}: {exc}"
        hits = []

    if enable_evebio:
        try:
            evebio_hits, evebio_metadata = lookup_evebio_off_targets(
                normalized_smiles,
                target_gene=target_gene,
                target_uniprot_id=target_uniprot_id,
                rows=evebio_rows,
                cache_path=evebio_cache_path,
                enable_live_download=enable_evebio_live_download,
                reader=evebio_reader,
            )
        except Exception as exc:
            metadata["evebio_lookup_error"] = f"{type(exc).__name__}: {exc}"
        else:
            hits = merge_off_target_hits([*hits, *evebio_hits])
            metadata.update(evebio_metadata)
            metadata["evebio_lookup_succeeded"] = True

    return OffTargetAssessment(
        hits=hits,
        metadata=metadata,
        class_effect_flags=class_effect_flags_for_hits(hits),
    )


def merge_off_target_hits(hits: list[OffTargetHit]) -> list[OffTargetHit]:
    severity_rank = {"low": 0, "medium": 1, "high": 2}
    merged: dict[tuple[str, str], OffTargetHit] = {}
    for hit in hits:
        key = (hit.ligand_smiles, hit.off_target_gene)
        existing = merged.get(key)
        if existing is None:
            merged[key] = hit
            continue
        if severity_rank[hit.severity] > severity_rank[existing.severity]:
            merged[key] = hit
            continue
        if hit.severity == existing.severity and hit.similarity > existing.similarity:
            merged[key] = hit
    return sorted(merged.values(), key=lambda item: (-item.similarity, item.off_target_gene))


def scan_off_targets(
    smiles: str,
    off_target_map: dict[str, list[str]],
    *,
    threshold: float = 0.5,
    similarity_fn: SimilarityFunction | None = None,
) -> list[OffTargetHit]:
    calculator = similarity_fn or morgan_tanimoto_similarity
    hits: list[OffTargetHit] = []
    for off_target_gene, reference_smiles_list in off_target_map.items():
        similarities = [
            calculator(smiles, reference_smiles)
            for reference_smiles in reference_smiles_list
            if reference_smiles.strip()
        ]
        if not similarities:
            continue
        max_similarity = max(similarities)
        if max_similarity < threshold:
            continue
        hits.append(
            OffTargetHit(
                ligand_smiles=smiles,
                off_target_gene=off_target_gene,
                similarity=round(max_similarity, 4),
                severity=severity_for_similarity(max_similarity),
            )
        )
    return sorted(hits, key=lambda item: (-item.similarity, item.off_target_gene))


def severity_for_similarity(similarity: float) -> str:
    if similarity >= 0.8:
        return "high"
    if similarity >= 0.65:
        return "medium"
    return "low"


def morgan_tanimoto_similarity(left_smiles: str, right_smiles: str) -> float:
    try:
        Chem = importlib.import_module("rdkit.Chem")
        DataStructs = importlib.import_module("rdkit.DataStructs")
        AllChem = importlib.import_module("rdkit.Chem.AllChem")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "RDKit is unavailable in the current runtime, so off-target similarity cannot be computed."
        ) from exc

    left_mol = Chem.MolFromSmiles(left_smiles)
    right_mol = Chem.MolFromSmiles(right_smiles)
    if left_mol is None or right_mol is None:
        return 0.0
    left_fp = AllChem.GetMorganFingerprintAsBitVect(left_mol, 2, nBits=2048)
    right_fp = AllChem.GetMorganFingerprintAsBitVect(right_mol, 2, nBits=2048)
    return float(DataStructs.TanimotoSimilarity(left_fp, right_fp))
