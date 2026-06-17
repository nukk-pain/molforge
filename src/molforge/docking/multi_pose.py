"""Multi-pose primitives for v3 A2.

Scope A (per `archive/runs/v3-a20-boltz2-scoring-mode/summary.json`):
run Boltz-2 with `--diffusion_samples N`, parse the emitted structure files,
and extract one `PoseWithAffinity` per sample. Boltz-2's affinity head emits
two fields per sample — we retain both.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PoseWithAffinity:
    """A single Boltz diffusion sample.

    Fields cover the affinity head outputs (Boltz-2, future-proof) as well
    as the ligand-interface confidence metric (Boltz-1 compatible). The
    reselection logic prefers affinity_pred_value when present and falls
    back to ligand_iptm (higher = stronger interface) on Boltz-1-era
    releases that have no affinity head.
    """

    sample_index: int
    structure_path: Path  # CIF or PDB emitted by `boltz predict`
    affinity_pred_value: float | None  # specific log-Ki, for optimisation
    affinity_probability_binary: float | None  # binder-vs-decoy, 0..1
    ligand_iptm: float | None = None  # Boltz-1 fallback: interface confidence
    confidence_score: float | None = None
    auxiliary: dict[str, object] = field(default_factory=dict)


def write_boltz_input_yaml(
    *,
    protein_sequence: str,
    ligand_smiles: str,
    protein_chain_id: str = "A",
    ligand_chain_id: str = "B",
    predict_affinity: bool = True,
) -> str:
    """Render the minimal YAML Boltz-2 accepts for one protein + one ligand.

    When `predict_affinity=True` the `properties` block enables the affinity
    head — required for multi-pose reselection since pose ranking is by
    affinity.
    """
    lines = [
        "version: 1",
        "sequences:",
        f"  - protein:",
        f"      id: {protein_chain_id}",
        f"      sequence: {protein_sequence}",
        f"  - ligand:",
        f"      id: {ligand_chain_id}",
        f'      smiles: "{ligand_smiles}"',
    ]
    if predict_affinity:
        lines.extend(
            [
                "properties:",
                "  - affinity:",
                f"      binder: {ligand_chain_id}",
            ]
        )
    return "\n".join(lines) + "\n"


def parse_boltz_samples(
    output_root: Path, *, input_stem: str = "input"
) -> list[PoseWithAffinity]:
    """Locate per-sample structure files + affinity JSON under a Boltz output tree.

    Boltz-2 writes `boltz_results_<input_stem>/` with predictions/ and
    scores/ (or similar) per-sample. This parser is intentionally
    forgiving: it walks the tree, groups files by sample index parsed
    from filenames, and tolerates missing affinity fields by emitting
    None rather than raising.

    The concrete file layout varies across boltz releases, so the
    matching heuristic is kept loose and documented.
    """
    # Expected root per CLI: `<out_dir>/boltz_results_<input_stem>/`
    candidate_roots = list(output_root.glob(f"**/boltz_results_{input_stem}"))
    if not candidate_roots:
        candidate_roots = [output_root]
    elif len(candidate_roots) > 1:
        import warnings

        warnings.warn(
            f"parse_boltz_samples: found {len(candidate_roots)} boltz_results_* "
            f"roots under {output_root}; using {candidate_roots[0]} and ignoring "
            f"siblings {candidate_roots[1:]}. This likely means output tree "
            f"from multiple runs was accidentally stacked together.",
            stacklevel=2,
        )
    base = candidate_roots[0]

    # Structure files: any .cif / .pdb with a digit suffix indicating sample id.
    structure_files = sorted(
        [*base.glob("**/*.cif"), *base.glob("**/*.pdb")],
    )

    # Score sources (priority order):
    # 1) Boltz-2 native affinity file: one `affinity_<stem>.json` directly
    #    under predictions/<id>/ with FLAT indexed keys
    #    (`affinity_pred_value`, `affinity_pred_value1`, `affinity_pred_value2`,
    #    `affinity_probability_binary`, `affinity_probability_binary1`, ...).
    #    Empty suffix = sample 0, numeric suffix = sample index.
    # 2) Boltz-1 era per-sample files (`affinity_0.json`, `aff_N.json`, ...).
    # 3) Confidence JSONs (`confidence_input_model_N.json`) for
    #    `ligand_iptm` / `confidence_score` tiebreakers — structural, NOT
    #    affinity.
    score_records: dict[int, dict[str, float]] = {}

    # (1) Boltz-2 combined affinity file.
    for affinity_path in base.glob("**/affinity_*.json"):
        try:
            payload = json.loads(affinity_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        for key, value in payload.items():
            if not isinstance(value, (int, float)):
                continue
            for base_key in ("affinity_pred_value", "affinity_probability_binary"):
                if not key.startswith(base_key):
                    continue
                suffix = key[len(base_key):]
                if suffix == "":
                    sample_idx: int | None = 0
                elif suffix.isdigit():
                    sample_idx = int(suffix)
                else:
                    sample_idx = None
                if sample_idx is None:
                    continue
                score_records.setdefault(sample_idx, {})[base_key] = float(value)
                break

    # (2)+(3) Per-sample JSONs (skip the combined affinity file we already read).
    for json_path in base.glob("**/*.json"):
        if json_path.name.startswith("affinity_"):
            continue
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        sample_index = _sample_index_from_name(json_path.name)
        if sample_index is None:
            continue
        if isinstance(payload, dict):
            record = score_records.setdefault(sample_index, {})
            for key in (
                "affinity_pred_value",
                "affinity_probability_binary",
                "ligand_iptm",
                "confidence_score",
            ):
                if key in record:
                    continue  # prefer Boltz-2 combined affinity values
                value = payload.get(key)
                if isinstance(value, (int, float)):
                    record[key] = float(value)

    poses: list[PoseWithAffinity] = []
    for path in structure_files:
        sample_index = _sample_index_from_name(path.name)
        if sample_index is None:
            continue
        record = score_records.get(sample_index, {})
        poses.append(
            PoseWithAffinity(
                sample_index=sample_index,
                structure_path=path,
                affinity_pred_value=record.get("affinity_pred_value"),
                affinity_probability_binary=record.get("affinity_probability_binary"),
                ligand_iptm=record.get("ligand_iptm"),
                confidence_score=record.get("confidence_score"),
                auxiliary={"score_json_present": bool(record)},
            )
        )
    # If nothing matched a sample index convention, fall back to an enumeration
    # so at least the structural outputs are returned to the caller.
    if not poses and structure_files:
        poses = [
            PoseWithAffinity(
                sample_index=index,
                structure_path=path,
                affinity_pred_value=None,
                affinity_probability_binary=None,
                auxiliary={"affinity_json_present": False, "parser_fallback": True},
            )
            for index, path in enumerate(structure_files)
        ]
    return poses


def _sample_index_from_name(name: str) -> int | None:
    """Extract a sample index from filenames like `pred_sample_0.cif` or `aff_2.json`."""
    import re

    match = re.search(r"(?:sample|pred|aff|model)[_-]?(\d+)", name.lower())
    if match:
        return int(match.group(1))
    # Fallback: trailing digit group before extension.
    trailing = re.search(r"(\d+)(?=\.[a-z0-9]+$)", name.lower())
    return int(trailing.group(1)) if trailing else None
