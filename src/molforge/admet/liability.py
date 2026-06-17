from __future__ import annotations

from ._endpoints import canonicalize_endpoint_name, get_liability_thresholds

LIABILITY_FLAG_ENDPOINTS: dict[str, set[str]] = {
    "hERG_high": {"herg"},
    "mutagenic": {"ames"},
    "hepatotox": {"dili", "hepatotoxicity"},
}


def derive_liability_flags(
    endpoints: dict[str, float],
    physchem: dict[str, float | int | str | bool | None] | None = None,
) -> list[str]:
    thresholds = get_liability_thresholds()
    flags: list[str] = []
    normalized_endpoints = {
        canonicalize_endpoint_name(name): value for name, value in endpoints.items()
    }
    physchem_payload = physchem or {}

    if float(normalized_endpoints.get("herg", 0.0)) > thresholds["herg"]:
        flags.append("hERG_high")
    if float(normalized_endpoints.get("ames", 0.0)) > thresholds["ames"]:
        flags.append("mutagenic")
    if (
        float(normalized_endpoints.get("dili", 0.0)) > thresholds["dili"]
        or float(normalized_endpoints.get("hepatotoxicity", 0.0))
        > thresholds["hepatotoxicity"]
    ):
        flags.append("hepatotox")
    if int(float(physchem_payload.get("PAINS_alert", 0) or 0)) >= 1:
        flags.append("PAINS")
    if int(float(physchem_payload.get("BRENK_alert", 0) or 0)) >= 1:
        flags.append("BRENK")
    if int(float(physchem_payload.get("Lipinski", 0) or 0)) >= int(
        thresholds["ro5_violations"]
    ):
        flags.append("ro5_violations")

    return sorted(set(flags))


def liability_penalty(liability_flags: list[str]) -> float:
    return min(0.5, len(set(liability_flags)) * 0.1)


def excluded_endpoints_for_flags(liability_flags: list[str]) -> set[str]:
    excluded: set[str] = set()
    for flag in liability_flags:
        excluded.update(LIABILITY_FLAG_ENDPOINTS.get(flag, set()))
    return excluded
