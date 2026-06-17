from __future__ import annotations

import math
from typing import Literal

from ._endpoints import get_endpoint_names, get_expected_endpoint_count

Direction = Literal["higher_is_better", "lower_is_better"]

ENDPOINT_DIRECTION: dict[str, Direction] = {
    "ames": "lower_is_better",
    "bbb_martins": "lower_is_better",
    "bioavailability_ma": "higher_is_better",
    "caco2_wang": "higher_is_better",
    "carcinogens_lagunin": "lower_is_better",
    "clearance_hepatocyte_az": "lower_is_better",
    "clearance_microsome_az": "lower_is_better",
    "clintox": "lower_is_better",
    "cyp1a2_veith": "lower_is_better",
    "cyp2c19_veith": "lower_is_better",
    "cyp2c9_substrate_carbonmangels": "lower_is_better",
    "cyp2c9_veith": "lower_is_better",
    "cyp2d6_substrate_carbonmangels": "lower_is_better",
    "cyp2d6_veith": "lower_is_better",
    "cyp3a4_substrate_carbonmangels": "lower_is_better",
    "cyp3a4_veith": "lower_is_better",
    "dili": "lower_is_better",
    "half_life_obach": "higher_is_better",
    "herg": "lower_is_better",
    "hia_hou": "higher_is_better",
    "hydrationfreeenergy_freesolv": "lower_is_better",
    "ld50_zhu": "higher_is_better",
    "lipophilicity_astrazeneca": "lower_is_better",
    "pampa_ncats": "higher_is_better",
    "pgp_broccatelli": "lower_is_better",
    "ppbr_az": "lower_is_better",
    "solubility_aqsoldb": "higher_is_better",
    "vdss_lombardo": "higher_is_better",
    "skin_reaction": "lower_is_better",
    "sr_are": "lower_is_better",
    "sr_atad5": "lower_is_better",
    "sr_hse": "lower_is_better",
    "sr_mmp": "lower_is_better",
    "sr_p53": "lower_is_better",
    "nr_ar": "lower_is_better",
    "nr_ar_lbd": "lower_is_better",
    "nr_ahr": "lower_is_better",
    "nr_aromatase": "lower_is_better",
    "nr_er": "lower_is_better",
    "nr_er_lbd": "lower_is_better",
    "nr_ppar_gamma": "lower_is_better",
}

EXPECTED_ENDPOINT_NAMES = set(get_endpoint_names())
if set(ENDPOINT_DIRECTION) != EXPECTED_ENDPOINT_NAMES:
    raise ValueError("ENDPOINT_DIRECTION must define every canonical Phase 4 endpoint.")
if len(ENDPOINT_DIRECTION) != get_expected_endpoint_count():
    raise ValueError(
        "ENDPOINT_DIRECTION size mismatch for Phase 4 canonical endpoints."
    )


def normalize_endpoint_score(endpoint_name: str, value: float) -> float:
    direction = ENDPOINT_DIRECTION[endpoint_name]
    bounded_value = squash_score(float(value))
    if direction == "lower_is_better":
        return 1.0 - bounded_value
    return bounded_value


def squash_score(value: float) -> float:
    if 0.0 <= value <= 1.0:
        return value
    return 1.0 / (1.0 + math.exp(-value))
