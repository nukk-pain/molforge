from __future__ import annotations

import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

derive_liability_flags = importlib.import_module(
    "molforge.admet.liability"
).derive_liability_flags


def test_derive_liability_flags_applies_plan_rules() -> None:
    flags = derive_liability_flags(
        {
            "herg": 0.9,
            "ames": 0.8,
            "dili": 0.7,
            "hepatotoxicity": 0.2,
        },
        {"PAINS_alert": 1, "BRENK_alert": 1, "Lipinski": 2},
    )

    assert flags == [
        "BRENK",
        "PAINS",
        "hERG_high",
        "hepatotox",
        "mutagenic",
        "ro5_violations",
    ]


def test_derive_liability_flags_returns_empty_when_no_rules_match() -> None:
    assert derive_liability_flags({"herg": 0.1, "ames": 0.1, "dili": 0.1}, {}) == []
