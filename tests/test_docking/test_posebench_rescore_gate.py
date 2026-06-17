# pyright: reportMissingImports=false
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from molforge.docking.posebench_rescore import generate_posebench_rescore_summary  # noqa: E402


def test_posebench_rescore_gate_reports_missing_baseline(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    summary = generate_posebench_rescore_summary(
        baseline_summary_path=tmp_path / "missing.json",
        output_dir=output_dir,
        rescore_cost_estimate_usd=0.4,
    )

    assert summary["status"] == "blocked"
    assert summary["boltz2_rescored_status"] == "blocked"
    assert summary["rescore_cost_estimate_usd"] == 0.4
    assert (output_dir / "summary.json").exists()


def test_posebench_rescore_gate_preserves_rmsd_status_without_reselection(
    tmp_path: Path,
) -> None:
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "status": "soft_pass",
                "mean_rmsd": 3.03,
                "rmsd_under_2a_rate": 1 / 3,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    summary = generate_posebench_rescore_summary(
        baseline_summary_path=baseline_path,
        output_dir=tmp_path / "out",
        rescore_cost_estimate_usd=1.2,
    )

    assert summary["status"] == "completed"
    assert summary["vina_baseline_status"] == "soft_pass"
    assert summary["boltz2_rescored_status"] == "soft_pass"
    assert summary["mean_rmsd_before"] == 3.03
    assert summary["mean_rmsd_after"] == 3.03
    assert summary["rescore_cost_estimate_usd"] == 1.2
