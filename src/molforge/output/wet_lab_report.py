"""RankedCandidate + BatchResult → wet-lab handoff JSON (v0 format)."""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from contracts.schema import RankedCandidate
from molforge.core.multi_target_batch import BatchResult

SCHEMA_VERSION = "wet-lab/v0"


@dataclass(frozen=True, slots=True)
class WetLabCandidate:
    rank: int
    smiles: str
    canonical_smiles: str
    inchikey: str
    physchem: dict[str, Any]
    admet_summary: dict[str, Any]
    off_target_warnings: list[dict[str, Any]]
    scoring: dict[str, Any]
    identifiers: dict[str, Any]
    order_hint: dict[str, Any]
    provenance: dict[str, Any]


@dataclass(frozen=True, slots=True)
class WetLabTarget:
    target_gene: str
    target_score: float | None
    run_id: str | None
    candidate_count: int
    candidates: list[WetLabCandidate]


@dataclass(frozen=True, slots=True)
class WetLabBatchReport:
    schema_version: str
    generated_at_utc: str
    batch_id: str
    disease_context: str
    upstream: dict[str, Any]
    targets: list[WetLabTarget]
    per_target_failures: list[dict[str, Any]] = field(default_factory=list)


def _rdkit_physchem(smiles: str) -> dict[str, Any]:
    """Compute RDKit-derived physchem descriptors, tolerating invalid SMILES."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, QED
        from rdkit.Chem.inchi import MolToInchiKey
    except ImportError:  # pragma: no cover — rdkit is a base dep
        return {"error": "rdkit_unavailable"}

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": "smiles_invalid", "raw_smiles": smiles}

    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
    inchikey = MolToInchiKey(mol) or ""

    mw = round(Descriptors.MolWt(mol), 3)
    logp = round(Crippen.MolLogP(mol), 3)
    tpsa = round(Descriptors.TPSA(mol), 3)
    hba = int(Lipinski.NumHAcceptors(mol))
    hbd = int(Lipinski.NumHDonors(mol))
    rot = int(Lipinski.NumRotatableBonds(mol))
    heavy = int(mol.GetNumHeavyAtoms())
    ro5 = int(
        (mw > 500) + (logp > 5) + (hba > 10) + (hbd > 5)
    )
    qed_value = round(QED.qed(mol), 4)
    _ = AllChem  # keep import alive for potential downstream reuse
    return {
        "canonical_smiles": canonical_smiles,
        "inchikey": inchikey,
        "molecular_weight": mw,
        "logp": logp,
        "tpsa": tpsa,
        "hba": hba,
        "hbd": hbd,
        "rotatable_bonds": rot,
        "heavy_atoms": heavy,
        "ro5_violations": ro5,
        "qed": qed_value,
    }


def _classify_synthesis_risk(sa_score: float | None) -> str:
    if sa_score is None:
        return "unknown"
    if sa_score < 4:
        return "low"
    if sa_score <= 6:
        return "medium"
    return "high"


PubChemLookup = Callable[[str], int | None]


def _default_pubchem_lookup(inchikey: str) -> int | None:
    if not inchikey:
        return None
    try:
        import httpx

        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/"
            f"{inchikey}/cids/JSON"
        )
        with httpx.Client(timeout=10.0) as client:
            response = client.get(url)
            if response.status_code != 200:
                return None
            payload = response.json()
    except Exception:  # noqa: BLE001 — best-effort, log nothing
        return None
    identifier_list = payload.get("IdentifierList") or {}
    cids = identifier_list.get("CID") or []
    return int(cids[0]) if cids else None


def build_candidate(
    candidate: RankedCandidate,
    *,
    batch_id: str,
    molforge_run_id: str | None,
    pubchem_lookup: PubChemLookup = _default_pubchem_lookup,
    off_target_source: str = "fallback_cache",
) -> WetLabCandidate:
    ligand_smiles = candidate.ligand.smiles
    physchem = _rdkit_physchem(ligand_smiles)
    canonical_smiles = str(physchem.get("canonical_smiles") or ligand_smiles)
    inchikey = str(physchem.get("inchikey") or "")

    # Trim the physchem payload to the spec fields — drop rdkit scratch keys.
    spec_keys = (
        "molecular_weight", "logp", "tpsa", "hba", "hbd",
        "rotatable_bonds", "heavy_atoms", "ro5_violations", "qed",
    )
    physchem_out: dict[str, Any] = {k: physchem[k] for k in spec_keys if k in physchem}

    # SA score is not available on Ligand/RankedCandidate directly in v1 schema —
    # surface from provenance if the generative backend recorded it.
    sa_score: float | None = None
    if isinstance(candidate.provenance, dict):
        value = candidate.provenance.get("sa_score")
        if isinstance(value, (int, float)):
            sa_score = float(value)
    if sa_score is not None:
        physchem_out["sa_score"] = round(sa_score, 4)

    admet_flags = list(candidate.admet.liability_flags)
    admet_summary = {
        "composite_score": round(candidate.composite_score, 4),
        "top_liabilities": admet_flags[:3],
        "endpoint_count": len(candidate.admet.endpoints),
    }

    off_target_warnings = [
        {
            "off_target_gene": hit.off_target_gene,
            "similarity": round(hit.similarity, 4),
            "severity": hit.severity,
            # Codex review honesty: reveal whether warnings came from a live
            # ChEMBL scan or from the packaged fallback cache.
            "source": off_target_source,
        }
        for hit in sorted(
            candidate.off_targets, key=lambda h: (-h.similarity, h.off_target_gene)
        )
    ]

    scoring = {
        "vina_score": None,
        "affinity_log_ki": None,
        "rescored_by": None,
        "rescore_run_id": None,
    }
    if candidate.affinity is not None:
        scoring["vina_score"] = candidate.affinity.vina_score
        scoring["affinity_log_ki"] = candidate.affinity.affinity_log_ki
        if candidate.affinity.affinity_log_ki is not None:
            scoring["rescored_by"] = "boltz-2-reselection"
        else:
            scoring["rescored_by"] = "vina-top1"

    pubchem_cid = pubchem_lookup(inchikey) if inchikey else None

    identifiers = {
        "pubchem_cid": pubchem_cid,
        "chembl_id": candidate.ligand.chembl_id,
        "commercial_available": None,
    }

    order_hint = {
        "supplier_candidates": [],
        "synthesis_risk": _classify_synthesis_risk(sa_score),
        "custom_synthesis_needed": _classify_synthesis_risk(sa_score) == "high",
    }

    provenance = {
        "ligand_source": candidate.ligand.source,
        "molforge_run_id": molforge_run_id,
        "batch_id": batch_id,
    }

    return WetLabCandidate(
        rank=candidate.rank,
        smiles=ligand_smiles,
        canonical_smiles=canonical_smiles,
        inchikey=inchikey,
        physchem=physchem_out,
        admet_summary=admet_summary,
        off_target_warnings=off_target_warnings,
        scoring=scoring,
        identifiers=identifiers,
        order_hint=order_hint,
        provenance=provenance,
    )


def build_report(
    batch_result: BatchResult,
    *,
    batch_id: str,
    disease_context: str,
    upstream: dict[str, Any] | None = None,
    pubchem_lookup: PubChemLookup = _default_pubchem_lookup,
    now: Callable[[], _dt.datetime] = lambda: _dt.datetime.now(_dt.UTC),
    live_chembl_enabled: bool = False,
) -> WetLabBatchReport:
    targets: list[WetLabTarget] = []
    failures: list[dict[str, Any]] = []

    off_target_source = "live_chembl" if live_chembl_enabled else "fallback_cache"

    for row in batch_result.per_target:
        if row.status != "completed" or row.run is None:
            failures.append(
                {
                    "target_gene": row.target_gene,
                    "status": row.status,
                    "error_message": row.error_message,
                }
            )
            continue
        candidates = [
            build_candidate(
                candidate,
                batch_id=batch_id,
                molforge_run_id=row.run_id,
                pubchem_lookup=pubchem_lookup,
                off_target_source=off_target_source,
            )
            for candidate in row.run.candidates
        ]
        targets.append(
            WetLabTarget(
                target_gene=row.target_gene,
                target_score=getattr(row.run.input_target, "score", None),
                run_id=row.run_id,
                candidate_count=len(candidates),
                candidates=candidates,
            )
        )

    upstream_out = dict(upstream or {})
    upstream_out.setdefault("live_chembl_enabled", live_chembl_enabled)
    upstream_out.setdefault("off_target_warnings_source", off_target_source)

    return WetLabBatchReport(
        schema_version=SCHEMA_VERSION,
        generated_at_utc=now().isoformat(),
        batch_id=batch_id,
        disease_context=disease_context,
        upstream=upstream_out,
        targets=targets,
        per_target_failures=failures,
    )


def write_report(report: WetLabBatchReport, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(report)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
