# pyright: reportMissingImports=false
"""v5 Track I1 — `molforge fullloop` orchestrator TDD.

Covers 4 scenarios per PLAN AC-I1-7:
  1. happy path — all 3 stages complete
  2. biocompute fails → molforge + neuroregen skipped, summary records failure
  3. molforge partial target failure → stage reported "partial" with
     per-target status, cost still tallied, neuroregen blocked
  4. resume after molforge partial → biocompute stage skipped, molforge
     re-runs only failed targets, neuroregen proceeds after full completion
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from molforge.core.fullloop import (  # noqa: E402
    FullloopResult,
    run_fullloop,
)


# ---------------------------------------------------------------------------
# Mock runner helpers
# ---------------------------------------------------------------------------


def _make_biocompute_runner(
    *, should_succeed: bool = True, neuroregen_targets: list[dict] | None = None
):
    """Returns a mock that writes a biocompute/<run>/neuroregen_targets.json
    under the caller's output_dir and returns a dict matching the real
    adapter contract.
    """
    neuroregen_targets = neuroregen_targets or [
        {
            "gene": "CXCL12",
            "gene_score": 0.85,
            "keyword": "scar",
            "provenance": {"pipeline": "biocompute-mock"},
        }
    ]

    def runner(
        *,
        disease: str,
        description: str | None,
        keywords: tuple,
        generations: int,
        population_size: int,
        output_dir: Path,
        biocompute_dir: Path,
    ) -> dict:
        # Emulate biocompute's own run_dir layout
        run_dir = output_dir / "mock-run-001"
        run_dir.mkdir(parents=True, exist_ok=True)
        targets_path = run_dir / "neuroregen_targets.json"
        targets_path.write_text(json.dumps(neuroregen_targets), encoding="utf-8")
        # stdout/stderr persistence is the adapter's responsibility in the real
        # implementation; the mock just leaves the raw payload.
        (run_dir / "report.md").write_text("# mock biocompute report\n")
        return {
            "success": should_succeed,
            "run_dir": run_dir,
            "neuroregen_targets_path": targets_path,
            "stdout": "mock stdout",
            "stderr": "" if should_succeed else "mock biocompute failure",
            "exit_code": 0 if should_succeed else 1,
            "elapsed_seconds": 1.0,
        }

    return runner


def _make_molforge_runner(
    *, per_target_statuses: list[str] | None = None, cost_per_target: float = 0.5
):
    """Returns a mock stage runner. `per_target_statuses` is a list of
    'completed' / 'failed:<reason>' strings, one per input target."""
    per_target_statuses = per_target_statuses or ["completed"]

    def runner(
        *, neuroregen_targets_path: Path, output_dir: Path, disease: str, top_n: int
    ) -> dict:
        targets = json.loads(neuroregen_targets_path.read_text())
        per_target = []
        completed = 0
        total_cost = 0.0
        for i, status in enumerate(per_target_statuses[: len(targets)]):
            gene = targets[i].get("gene", f"GENE{i}")
            per_target.append(
                {
                    "target_gene": gene,
                    "status": status,
                    "cost_estimate_usd": cost_per_target
                    if status == "completed"
                    else 0.0,
                }
            )
            if status == "completed":
                completed += 1
                total_cost += cost_per_target
        # Write artifact the real adapter would produce
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "cross_target_summary.json").write_text(
            json.dumps({"per_target": per_target, "total_cost_usd": total_cost})
        )
        (output_dir / "wet_lab_report.json").write_text(json.dumps({"candidates": []}))
        return {
            "success": completed > 0,
            "per_target": per_target,
            "total_cost_usd": total_cost,
            "completed_count": completed,
            "wet_lab_report_path": output_dir / "wet_lab_report.json",
            "elapsed_seconds": 1.5,
        }

    return runner


def _make_neuroregen_runner(*, should_succeed: bool = True):
    def runner(
        *,
        neuroregen_targets_path: Path,
        output_dir: Path,
        neuroregen_dir: Path,
        top_n: int,
    ) -> dict:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "stdout.log").write_text("mock neuroregen log\n")
        if should_succeed:
            designs_path = output_dir / "mrna_designs.json"
            designs_path.write_text(
                json.dumps({"designs": [{"gene": "CXCL12", "mrna": "AUG..."}]})
            )
            return {
                "success": True,
                "designs_path": designs_path,
                "stderr": "",
                "exit_code": 0,
                "elapsed_seconds": 2.0,
            }
        return {
            "success": False,
            "designs_path": None,
            "stderr": "mock neuroregen failure",
            "exit_code": 1,
            "elapsed_seconds": 0.5,
        }

    return runner


# ---------------------------------------------------------------------------
# Scenario 1 — happy path
# ---------------------------------------------------------------------------


def test_fullloop_happy_path_all_three_stages_complete(tmp_path):
    result = run_fullloop(
        disease="Myofascial Pain Syndrome",
        description="chronic pain from nerve hyperinnervation",
        keywords=("scar", "hyperinnervation", "fascia"),
        biocompute_dir=tmp_path / "biocompute_repo",
        neuroregen_dir=tmp_path / "neuroregen_repo",
        output_dir=tmp_path / "fullloop_out",
        top_n=3,
        generations=2,
        population_size=3,
        cost_budget_usd=5.0,
        runners={
            "biocompute": _make_biocompute_runner(),
            "molforge": _make_molforge_runner(cost_per_target=0.4),
            "neuroregen": _make_neuroregen_runner(),
        },
    )

    assert isinstance(result, FullloopResult)
    assert result.stage_status["biocompute"] == "completed"
    assert result.stage_status["molforge"] == "completed"
    assert result.stage_status["neuroregen"] == "completed"
    assert result.total_cost_usd == pytest.approx(0.4, abs=1e-6)
    assert result.neuroregen_designs_path is not None
    assert result.neuroregen_designs_path.exists()

    # fullloop_summary.json written
    summary_path = result.output_dir / "fullloop_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert summary["schema_version"] == "fullloop/v2"
    assert summary["stage_status"]["biocompute"] == "completed"


# ---------------------------------------------------------------------------
# Scenario 2 — biocompute stage fails
# ---------------------------------------------------------------------------


def test_fullloop_biocompute_failure_skips_downstream(tmp_path):
    result = run_fullloop(
        disease="Test",
        description="test",
        keywords=(),
        biocompute_dir=tmp_path / "bc",
        neuroregen_dir=tmp_path / "nr",
        output_dir=tmp_path / "out",
        top_n=3,
        generations=1,
        population_size=1,
        cost_budget_usd=5.0,
        runners={
            "biocompute": _make_biocompute_runner(should_succeed=False),
            "molforge": _make_molforge_runner(),
            "neuroregen": _make_neuroregen_runner(),
        },
    )
    assert result.stage_status["biocompute"].startswith("failed:")
    assert result.stage_status["molforge"] == "skipped"
    assert result.stage_status["neuroregen"] == "skipped"
    # No cost accrued because molforge never ran.
    assert result.total_cost_usd == 0.0


# ---------------------------------------------------------------------------
# Scenario 3 — molforge stage partial target failure
# ---------------------------------------------------------------------------


def test_fullloop_molforge_partial_failure_blocks_neuroregen_and_records_detail(
    tmp_path,
):
    result = run_fullloop(
        disease="Test",
        description="test",
        keywords=(),
        biocompute_dir=tmp_path / "bc",
        neuroregen_dir=tmp_path / "nr",
        output_dir=tmp_path / "out",
        top_n=3,
        generations=1,
        population_size=1,
        cost_budget_usd=5.0,
        runners={
            "biocompute": _make_biocompute_runner(
                neuroregen_targets=[
                    {
                        "gene": "A",
                        "gene_score": 0.9,
                        "keyword": "k",
                        "provenance": {"pipeline": "mock"},
                    },
                    {
                        "gene": "B",
                        "gene_score": 0.8,
                        "keyword": "k",
                        "provenance": {"pipeline": "mock"},
                    },
                    {
                        "gene": "C",
                        "gene_score": 0.7,
                        "keyword": "k",
                        "provenance": {"pipeline": "mock"},
                    },
                ]
            ),
            "molforge": _make_molforge_runner(
                per_target_statuses=["completed", "failed:OOM", "completed"],
                cost_per_target=0.3,
            ),
            "neuroregen": _make_neuroregen_runner(),
        },
    )
    assert result.stage_status["molforge"] == "partial"
    assert result.stage_status["neuroregen"] == "skipped"
    # Cost = 2 completed × 0.3 (failed target contributes 0).
    assert result.total_cost_usd == pytest.approx(0.6, abs=1e-6)
    summary = json.loads((result.output_dir / "fullloop_summary.json").read_text())
    molforge_detail = summary["stage_details"]["molforge"]
    assert molforge_detail["target_count"] == 3
    assert molforge_detail["completed_count"] == 2
    assert molforge_detail["pending_target_genes"] == ["B"]


# ---------------------------------------------------------------------------
# Scenario 4 — resume after molforge failed
# ---------------------------------------------------------------------------


def test_fullloop_resume_after_molforge_partial_reruns_only_failed_targets(tmp_path):
    output_dir = tmp_path / "out"
    targets = [
        {
            "gene": "A",
            "gene_score": 0.9,
            "keyword": "k",
            "provenance": {"pipeline": "mock"},
        },
        {
            "gene": "B",
            "gene_score": 0.8,
            "keyword": "k",
            "provenance": {"pipeline": "mock"},
        },
        {
            "gene": "C",
            "gene_score": 0.7,
            "keyword": "k",
            "provenance": {"pipeline": "mock"},
        },
    ]

    first = run_fullloop(
        disease="Test",
        description="test",
        keywords=(),
        biocompute_dir=tmp_path / "bc",
        neuroregen_dir=tmp_path / "nr",
        output_dir=output_dir,
        top_n=3,
        generations=1,
        population_size=1,
        cost_budget_usd=5.0,
        runners={
            "biocompute": _make_biocompute_runner(neuroregen_targets=targets),
            "molforge": _make_molforge_runner(
                per_target_statuses=["completed", "failed:OOM", "completed"],
                cost_per_target=0.3,
            ),
            "neuroregen": _make_neuroregen_runner(),
        },
    )
    assert first.stage_status["biocompute"] == "completed"
    assert first.stage_status["molforge"] == "partial"
    assert first.stage_status["neuroregen"] == "skipped"

    # Second run: resume — biocompute should NOT rerun.
    bc_called: list[int] = []
    rerun_target_lists: list[list[str]] = []

    def _tracking_bc_runner(**kwargs):
        bc_called.append(1)
        return _make_biocompute_runner()(**kwargs)

    def _tracking_molforge_runner(
        *, neuroregen_targets_path, output_dir, disease, top_n
    ):
        targets = json.loads(neuroregen_targets_path.read_text())
        rerun_target_lists.append([target["gene"] for target in targets])
        return _make_molforge_runner(
            per_target_statuses=["completed"],
            cost_per_target=0.4,
        )(
            neuroregen_targets_path=neuroregen_targets_path,
            output_dir=output_dir,
            disease=disease,
            top_n=top_n,
        )

    # Allow a tiny pause so any re-write would have a newer mtime.
    time.sleep(0.05)
    second = run_fullloop(
        disease="Test",
        description="test",
        keywords=(),
        biocompute_dir=tmp_path / "bc",
        neuroregen_dir=tmp_path / "nr",
        output_dir=output_dir,
        top_n=3,
        generations=1,
        population_size=1,
        cost_budget_usd=5.0,
        resume=True,
        runners={
            "biocompute": _tracking_bc_runner,
            "molforge": _tracking_molforge_runner,
            "neuroregen": _make_neuroregen_runner(),
        },
    )
    assert len(bc_called) == 0, "biocompute runner should not be called on resume"
    assert rerun_target_lists == [["B"]]
    assert second.stage_status["biocompute"] == "completed"
    assert second.stage_status["molforge"] == "completed"
    assert second.stage_status["neuroregen"] == "completed"
    summary = json.loads((output_dir / "fullloop_summary.json").read_text())
    molforge_detail = summary["stage_details"]["molforge"]
    assert molforge_detail["completed_target_genes"] == ["A", "B", "C"]
    assert molforge_detail["pending_target_genes"] == []
    assert molforge_detail["resumed_only_target_genes"] == ["B"]


# ---------------------------------------------------------------------------
# Scenario 5 — cost budget exceeded (extra safety AC-I1-cost)
# ---------------------------------------------------------------------------


def test_fullloop_cost_budget_exceeded_triggers_budget_failure(tmp_path):
    result = run_fullloop(
        disease="Test",
        description="test",
        keywords=(),
        biocompute_dir=tmp_path / "bc",
        neuroregen_dir=tmp_path / "nr",
        output_dir=tmp_path / "out",
        top_n=3,
        generations=1,
        population_size=1,
        cost_budget_usd=0.5,  # very low
        runners={
            "biocompute": _make_biocompute_runner(
                neuroregen_targets=[
                    {
                        "gene": f"T{i}",
                        "gene_score": 1.0,
                        "keyword": "k",
                        "provenance": {"pipeline": "mock"},
                    }
                    for i in range(3)
                ]
            ),
            "molforge": _make_molforge_runner(
                per_target_statuses=["completed"] * 3,
                cost_per_target=0.4,  # 3 × 0.4 = 1.2 > 0.5 budget
            ),
            "neuroregen": _make_neuroregen_runner(),
        },
    )
    # AC-I1-cost: total_cost_usd > budget → fullloop flags it (stage marked
    # budget_exceeded) and neuroregen is skipped.
    assert result.stage_status["molforge"] == "budget_exceeded"
    assert result.stage_status["neuroregen"] == "skipped"
    assert result.total_cost_usd > 0.5
