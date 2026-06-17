from __future__ import annotations

import argparse
import json
from pathlib import Path

from .input import load_target_candidates


SCHEMA_VERSION = "small-molecule-decision-assessment/v1"


def run_small_molecule_preflight(
    targets_path: Path,
    *,
    biological_assessment_path: Path | None = None,
    disease: str | None = None,
) -> list[dict[str, object]]:
    """Emit advisory small-molecule suitability assessments.

    This is intentionally lightweight: it does not import RDKit, docking, ADMET,
    Boltz, or other heavy dependencies. Missing biological sidecars degrade to
    `unknown` rather than blocking existing molforge runs.
    """
    targets = load_target_candidates(targets_path, disease=disease)
    biological_by_gene = load_assessments_by_gene(biological_assessment_path)
    assessments: list[dict[str, object]] = []
    for index, target in enumerate(targets):
        biological = biological_by_gene.get(target.gene)
        decision_context = extract_decision_context(biological)
        suitability = assess_modality_suitability(target, biological is not None, decision_context)
        assessments.append(
            {
                "assessment_schema_version": SCHEMA_VERSION,
                "target_ref": {
                    "gene": target.gene,
                    "candidate_index": index,
                    "source": str(targets_path),
                    "disease": target.disease,
                },
                "decision_context": decision_context,
                "layers": {
                    "target_validity": summarize_target_validity(target, biological),
                    "modality_suitability": suitability,
                    "computational_feasibility": assess_computational_feasibility(target),
                    "experimental_readiness": unknown_layer(
                        "Wet-lab assay readiness is outside molforge preflight."
                    ),
                },
                "provenance": {
                    "created_by": "molforge.core.small_molecule_preflight",
                    "targets_path": str(targets_path),
                    "biological_assessment_path": str(biological_assessment_path) if biological_assessment_path else None,
                },
            }
        )
    return assessments


def load_assessments_by_gene(path: Path | None) -> dict[str, dict[str, object]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload if isinstance(payload, list) else [payload]
    result: dict[str, dict[str, object]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        target_ref = item.get("target_ref")
        if not isinstance(target_ref, dict):
            continue
        gene = target_ref.get("gene")
        if isinstance(gene, str) and gene:
            result[gene] = item
    return result


def summarize_target_validity(target: object, biological: dict[str, object] | None) -> dict[str, object]:
    if biological:
        layers = biological.get("layers")
        if isinstance(layers, dict):
            target_validity = layers.get("target_validity")
            if isinstance(target_validity, dict):
                return target_validity
    return unknown_layer("No biocompute biological assessment sidecar supplied.")


def extract_decision_context(biological: dict[str, object] | None) -> dict[str, str]:
    default = {
        "directionality_hypothesis": "unknown",
        "payload_strategy": "unknown",
        "intervention_target": "unknown",
    }
    if not biological:
        return default
    context = biological.get("decision_context")
    if not isinstance(context, dict):
        return default
    return {
        key: str(context.get(key) or default[key])
        for key in default
    }


def assess_modality_suitability(
    target: object,
    has_biological_sidecar: bool,
    decision_context: dict[str, str],
) -> dict[str, object]:
    target_gene = getattr(target, "gene")
    evidence = getattr(target, "evidence")
    pathway = getattr(target, "pathway")
    rationale: list[str] = []
    unknowns: list[str] = []
    score = 0.0
    if getattr(target, "uniprot_id"):
        score += 0.35
        rationale.append("UniProt identifier is available for protein-level lookup.")
    else:
        unknowns.append("UniProt identifier is missing; structure lookup may need fallback mapping.")
    if pathway:
        score += 0.25
        rationale.append("Pathway partners are available for mechanism review.")
    else:
        unknowns.append("No pathway partners supplied for tractability context.")
    if evidence:
        score += 0.2
        rationale.append("Target evidence summaries are available for review.")
    else:
        unknowns.append("No evidence summaries supplied.")
    if has_biological_sidecar:
        score += 0.2
        rationale.append("Biological target-validity sidecar is available.")
    else:
        unknowns.append("Biological sidecar is absent; biology/modality direction remains unresolved.")
    directionality = decision_context["directionality_hypothesis"]
    if directionality == "inhibit":
        rationale.append("Biological sidecar suggests inhibitory intervention; small-molecule antagonism may be directionally compatible.")
    elif directionality in {"activate", "replace"}:
        unknowns.append(f"Biological sidecar suggests {directionality}; small-molecule strategy needs explicit mechanism review.")
    else:
        unknowns.append("Directionality hypothesis is unknown; small-molecule intervention class cannot be selected yet.")
    gate = "conditional" if score >= 0.5 else "unknown"
    return {
        "score": round(score, 3),
        "gate": gate,
        "rationale": rationale or [f"{target_gene} requires small-molecule tractability review."],
        "blockers": [],
        "unknowns": unknowns,
    }


def assess_computational_feasibility(target: object) -> dict[str, object]:
    if getattr(target, "uniprot_id"):
        return {
            "score": 0.35,
            "gate": "conditional",
            "rationale": ["Protein identifier exists; downstream structure/docking checks can proceed."],
            "blockers": [],
            "unknowns": ["Pocket quality, ligand precedent, docking, and ADMET have not run."],
        }
    return unknown_layer("Protein identifier is missing; structure/docking feasibility is unknown.")


def unknown_layer(reason: str) -> dict[str, object]:
    return {
        "score": None,
        "gate": "unknown",
        "rationale": [],
        "blockers": [],
        "unknowns": [reason],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emit molforge small-molecule preflight sidecars.")
    parser.add_argument("targets_json")
    parser.add_argument("--biological-assessment", default=None)
    parser.add_argument("--disease", default=None)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    assessments = run_small_molecule_preflight(
        Path(args.targets_json),
        biological_assessment_path=Path(args.biological_assessment) if args.biological_assessment else None,
        disease=args.disease,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(assessments, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote small-molecule preflight: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
