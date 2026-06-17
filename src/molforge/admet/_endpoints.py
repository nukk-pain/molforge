from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

CONFIG_PATH = Path(__file__).with_name("config.json")


@lru_cache(maxsize=1)
def load_phase4_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    endpoint_names = config.get("endpoint_names", [])
    expected_endpoint_count = config.get("expected_endpoint_count")
    if not isinstance(endpoint_names, list) or not all(
        isinstance(item, str) and item for item in endpoint_names
    ):
        raise ValueError("Phase 4 ADMET config must define non-empty endpoint names.")
    if expected_endpoint_count != len(endpoint_names):
        raise ValueError(
            "Phase 4 ADMET config endpoint count must match expected_endpoint_count."
        )
    return config


def get_expected_endpoint_count() -> int:
    return int(load_phase4_config()["expected_endpoint_count"])


def get_endpoint_names() -> tuple[str, ...]:
    return tuple(load_phase4_config()["endpoint_names"])


def canonicalize_endpoint_name(endpoint_name: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z]+", "_", endpoint_name.strip()).strip("_")
    return normalized.lower()


def get_ranking_weights() -> dict[str, float]:
    return {
        str(key): float(value)
        for key, value in load_phase4_config()["ranking_weights"].items()
    }


def get_liability_thresholds() -> dict[str, float]:
    return {
        str(key): float(value)
        for key, value in load_phase4_config()["liability_thresholds"].items()
    }
