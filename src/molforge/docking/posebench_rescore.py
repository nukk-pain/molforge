from __future__ import annotations

import json
from pathlib import Path


def generate_posebench_rescore_summary(
    *,
    baseline_summary_path: Path,
    output_dir: Path,
    rescore_cost_estimate_usd: float = 0.0,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = _build_summary(
        baseline_summary_path=baseline_summary_path,
        rescore_cost_estimate_usd=rescore_cost_estimate_usd,
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _build_summary(
    *, baseline_summary_path: Path, rescore_cost_estimate_usd: float
) -> dict[str, object]:
    if not baseline_summary_path.exists():
        return {
            "gate": "v2-posebench-rescore",
            "status": "blocked",
            "baseline_summary_path": str(baseline_summary_path),
            "vina_baseline_status": None,
            "boltz2_rescored_status": "blocked",
            "rescore_cost_estimate_usd": round(rescore_cost_estimate_usd, 6),
            "diagnosis": (
                "Baseline PoseBench artifact is missing, so the v2 rescoring gate "
                "cannot recompute status yet. Recreate archive/runs/phase2-posebench-pass/summary.json first."
            ),
        }

    payload = json.loads(baseline_summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("PoseBench baseline summary must decode to an object.")

    vina_baseline_status = str(payload.get("status") or "unknown")
    mean_rmsd = _optional_float(payload.get("mean_rmsd"))
    rmsd_under_2a_rate = _optional_float(payload.get("rmsd_under_2a_rate"))
    return {
        "gate": "v2-posebench-rescore",
        "status": "completed",
        "baseline_summary_path": str(baseline_summary_path),
        "vina_baseline_status": vina_baseline_status,
        "boltz2_rescored_status": vina_baseline_status,
        "mean_rmsd_before": mean_rmsd,
        "mean_rmsd_after": mean_rmsd,
        "rmsd_under_2a_rate_before": rmsd_under_2a_rate,
        "rmsd_under_2a_rate_after": rmsd_under_2a_rate,
        "rescore_cost_estimate_usd": round(rescore_cost_estimate_usd, 6),
        "diagnosis": (
            "This gate records the current v2 diagnostic reality: affinity-only Boltz-2 rescoring enriches scores, but the existing Phase 2 artifact is a single-pose RMSD benchmark. Without multi-pose reselection artifacts, the PoseBench status remains unchanged."
        ),
    }


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError("PoseBench summary numeric fields must be numeric when present.")
