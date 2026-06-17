# pyright: reportMissingImports=false
"""Regression shield for SOTA rate divergence alarms.

If our benchmark pipeline silently under-performs the published SOTA
rate by 10× or more, the test helper these scripts depend on raises a
`SOTARateDivergenceError` so the CI job fails loudly instead of quietly
archiving a wrong number.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from molforge.utils import (  # noqa: E402
    SOTARateDivergenceError,
    check_sota_rate_divergence,
)


def test_matching_rate_passes():
    r = check_sota_rate_divergence(
        our_rate=0.55,
        reference_rate=0.53,
        source="DiffDock-L paper",
        target_subset="drug-like 5-target",
    )
    assert r.divergence_flagged is False
    assert r.ratio > 1.0


def test_slight_under_reference_passes():
    # Our 40% vs reference 53% is a ratio of 0.75 — well above 0.1 threshold.
    r = check_sota_rate_divergence(
        our_rate=0.40,
        reference_rate=0.53,
        source="DiffDock-L paper",
        target_subset="sugar 3-target",
    )
    assert r.divergence_flagged is False


def test_ten_times_divergence_raises():
    # Our 0.0 vs reference 53% is a hard divergence from the published rate.
    with pytest.raises(SOTARateDivergenceError) as exc_info:
        check_sota_rate_divergence(
            our_rate=0.0,
            reference_rate=0.53,
            source="DiffDock-L paper (Corso et al. 2024)",
            target_subset="PoseBench 3-target sugar subset",
        )
    assert "SOTA RATE DIVERGENCE" in str(exc_info.value)
    assert "verify the benchmark metric itself" in str(exc_info.value)


def test_divergence_without_raise_returns_flagged_result():
    r = check_sota_rate_divergence(
        our_rate=0.0,
        reference_rate=0.53,
        source="test",
        target_subset="test",
        raise_on_divergence=False,
    )
    assert r.divergence_flagged is True
    assert r.ratio == 0.0


def test_validates_rate_ranges():
    with pytest.raises(ValueError):
        check_sota_rate_divergence(
            our_rate=1.5,
            reference_rate=0.5,
            source="x",
            target_subset="x",
        )
    with pytest.raises(ValueError):
        check_sota_rate_divergence(
            our_rate=0.5,
            reference_rate=0.0,
            source="x",
            target_subset="x",
        )


def test_custom_threshold_respected():
    # ratio = 0.15; with min_ratio=0.2 it should flag, with 0.1 it shouldn't.
    r1 = check_sota_rate_divergence(
        our_rate=0.08,
        reference_rate=0.53,
        source="x",
        target_subset="x",
        min_ratio=0.2,
        raise_on_divergence=False,
    )
    assert r1.divergence_flagged is True

    r2 = check_sota_rate_divergence(
        our_rate=0.08,
        reference_rate=0.53,
        source="x",
        target_subset="x",
        min_ratio=0.1,
        raise_on_divergence=False,
    )
    assert r2.divergence_flagged is False
