# pyright: reportMissingImports=false
"""CLI subparser tests for `molforge fullloop`.

Verifies argparse wiring, mutual-exclusive flag handling (--skip-mrna
XOR --neuroregen-dir), and exit-code mapping onto FullloopResult
stage_status.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from molforge.cli import build_parser, main  # noqa: E402


# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------


def test_fullloop_subparser_registered():
    parser = build_parser()
    # Parse valid args and ensure `fullloop` subcommand is recognised.
    args = parser.parse_args([
        "fullloop", "Myofascial Pain",
        "--biocompute-dir", "/tmp/bc",
        "--skip-mrna",
    ])
    assert args.command == "fullloop"
    assert args.disease == "Myofascial Pain"
    assert args.biocompute_dir == "/tmp/bc"
    assert args.skip_mrna is True


def test_fullloop_requires_either_skip_mrna_or_neuroregen_dir(capsys):
    # Neither --skip-mrna nor --neuroregen-dir → argparse should reject.
    with pytest.raises(SystemExit):
        main([
            "fullloop", "Test",
            "--biocompute-dir", "/tmp/bc",
            "-d", "desc",
        ])


def test_fullloop_rejects_both_skip_mrna_and_neuroregen_dir(capsys):
    with pytest.raises(SystemExit):
        main([
            "fullloop", "Test",
            "--biocompute-dir", "/tmp/bc",
            "--neuroregen-dir", "/tmp/nr",
            "--skip-mrna",
            "-d", "desc",
        ])


# ---------------------------------------------------------------------------
# Dispatch → run_fullloop + exit codes
# ---------------------------------------------------------------------------


def test_fullloop_cli_happy_path_exit_0(tmp_path):
    output_dir = tmp_path / "out"
    with patch("molforge.cli.run_fullloop") as mock_run:
        fake = type("_F", (), {
            "stage_status": {"biocompute": "completed", "molforge": "completed", "neuroregen": "completed"},
            "stage_wall_seconds": {"biocompute": 1.0, "molforge": 2.0, "neuroregen": 3.0},
            "total_cost_usd": 0.4,
            "output_dir": output_dir,
            "errors": [],
        })()
        mock_run.return_value = fake

        rc = main([
            "fullloop", "Myofascial Pain",
            "-d", "chronic pain",
            "-k", "scar",
            "--biocompute-dir", str(tmp_path / "bc"),
            "--neuroregen-dir", str(tmp_path / "nr"),
            "--output-dir", str(output_dir),
            "--top-n", "3",
            "-g", "2",
            "-p", "2",
            "--cost-budget-usd", "5",
        ])
        assert rc == 0
        assert mock_run.called
        # Check kwargs were wired correctly.
        kwargs = mock_run.call_args.kwargs
        assert kwargs["disease"] == "Myofascial Pain"
        assert kwargs["description"] == "chronic pain"
        assert kwargs["keywords"] == ("scar",)
        assert kwargs["top_n"] == 3


def test_fullloop_cli_budget_exceeded_exit_3(tmp_path):
    output_dir = tmp_path / "out"
    with patch("molforge.cli.run_fullloop") as mock_run:
        fake = type("_F", (), {
            "stage_status": {
                "biocompute": "completed",
                "molforge": "budget_exceeded",
                "neuroregen": "skipped",
            },
            "stage_wall_seconds": {"biocompute": 1.0, "molforge": 2.0, "neuroregen": 0.0},
            "total_cost_usd": 10.0,
            "output_dir": output_dir,
            "errors": [],
        })()
        mock_run.return_value = fake

        rc = main([
            "fullloop", "Test",
            "-d", "desc",
            "--biocompute-dir", str(tmp_path / "bc"),
            "--skip-mrna",
            "--output-dir", str(output_dir),
            "--cost-budget-usd", "1",
        ])
        assert rc == 3, "CLI must exit 3 on cost budget exceeded (AC-I1-cost)"


def test_fullloop_cli_partial_failure_exit_1(tmp_path):
    output_dir = tmp_path / "out"
    with patch("molforge.cli.run_fullloop") as mock_run:
        fake = type("_F", (), {
            "stage_status": {
                "biocompute": "completed",
                "molforge": "failed:RuntimeError",
                "neuroregen": "skipped",
            },
            "stage_wall_seconds": {"biocompute": 1.0, "molforge": 0.5, "neuroregen": 0.0},
            "total_cost_usd": 0.0,
            "output_dir": output_dir,
            "errors": [],
        })()
        mock_run.return_value = fake

        rc = main([
            "fullloop", "Test",
            "-d", "desc",
            "--biocompute-dir", str(tmp_path / "bc"),
            "--skip-mrna",
            "--output-dir", str(output_dir),
        ])
        assert rc == 1
