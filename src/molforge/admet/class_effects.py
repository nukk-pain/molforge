from __future__ import annotations

from contracts.schema import OffTargetHit

CLASS_EFFECTS: dict[str, dict[str, str | float]] = {
    "VEGFR2": {"event": "HTN_risk", "penalty": 0.2},
    "JAK2": {"event": "myelosuppression_risk", "penalty": 0.2},
    "PPARG": {"event": "weight_gain_risk", "penalty": 0.1},
    "CETP": {"event": "lipid_shift_risk", "penalty": 0.1},
    "BRAF": {"event": "paradoxical_mapk_risk", "penalty": 0.2},
    "TNF": {"event": "immune_signal_risk", "penalty": 0.1},
    "CYP3A4": {"event": "ddi_risk", "penalty": 0.2},
}


def class_effect_flags_for_hits(hits: list[OffTargetHit]) -> list[str]:
    flags: list[str] = []
    for hit in hits:
        effect = CLASS_EFFECTS.get(hit.off_target_gene)
        if effect is None:
            continue
        flags.append(f"class_effect:{hit.off_target_gene}_{effect['event']}")
    return sorted(set(flags))
