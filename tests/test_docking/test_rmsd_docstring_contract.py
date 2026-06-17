# pyright: reportMissingImports=false
"""RMSD function docstrings must declare
their coordinate-frame contract.

Metric bugs can arise when no single RMSD helper in
`pose_rmsd_from_cif.py` documented whether it expects both inputs in
the same coordinate frame or performs receptor alignment itself.
Callers mixed conventions silently for 4 weeks.

This test enforces a minimal contract on any public `rmsd_*` function
exported by the module: the docstring MUST mention either "same frame"
(positional metric) or "receptor aligned" / "Cα superposition"
(aligned metric) or explicitly flag deprecation / non-pose semantics.
If someone adds a new RMSD helper without declaring its convention,
this test fails and forces them to think about it.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from molforge.docking import pose_rmsd_from_cif  # noqa: E402


# Minimum vocabulary a compliant docstring must include. At least one
# keyword from each CATEGORY set must appear. Case-insensitive.
_CONTRACT_KEYWORDS = [
    # Category 1 — coordinate-frame statement
    {
        "same frame",           # positional (no alignment)
        "crystal frame",
        "receptor-aligned",     # Kabsch on Cα first
        "receptor aligned",
        "cα superposition",
        "ca superposition",
        "deprecated",           # opt-out marker for legacy helpers
        "legacy",
        "kabsch",               # naive Kabsch helpers
    },
    # Category 2 — symmetry handling declared
    {
        "symmetry",
        "getbestrms",
        "assumes atom ordering",
        "atom ordering matches",
        "wrong atom order",      # naive helper caveats
    },
]


def _docstring_violates_contract(func) -> list[str] | None:
    """Return list of missing keyword categories, or None if compliant."""
    doc = (func.__doc__ or "").lower()
    missing = []
    for i, category in enumerate(_CONTRACT_KEYWORDS, start=1):
        if not any(kw in doc for kw in category):
            missing.append(
                f"category {i}: need one of {sorted(category)[:4]}..."
            )
    return missing if missing else None


def test_all_rmsd_functions_declare_frame_and_symmetry_convention():
    """Walk every `rmsd_*` callable in the module. Each docstring must
    disclose coordinate frame expectations AND symmetry handling —
    the two dimensions that matter for RMSD correctness."""
    rmsd_funcs = [
        (name, obj)
        for name, obj in inspect.getmembers(pose_rmsd_from_cif)
        if name.startswith("rmsd_") and inspect.isfunction(obj)
    ]
    assert rmsd_funcs, "no rmsd_* functions found to check"

    violations: dict[str, list[str]] = {}
    for name, func in rmsd_funcs:
        missing = _docstring_violates_contract(func)
        if missing:
            violations[name] = missing

    assert not violations, (
        "RMSD helpers missing coordinate-frame / symmetry docstring "
        "contract. Violations:\n"
        + "\n".join(
            f"  {name}: missing {', '.join(missing)}"
            for name, missing in violations.items()
        )
        + "\n\nSee rmsd_positional_symmetry_aware's docstring for the reference "
          "pattern — every new rmsd_* helper must state whether it "
          "expects same-frame input, performs receptor alignment, or is "
          "deprecated/legacy, AND whether it handles atom symmetries."
    )


def test_at_least_one_same_frame_and_one_receptor_aligned_helper_exists():
    """There is no single 'correct' RMSD metric —
    DiffDock-style same-frame and Boltz-style receptor-aligned each
    need their own helper. The module must expose both."""
    names = [
        name for name, obj in inspect.getmembers(pose_rmsd_from_cif)
        if name.startswith("rmsd_") and inspect.isfunction(obj)
    ]
    have_positional = any("positional" in n for n in names)
    have_receptor_aligned = any(
        "receptor" in n or "aligned" in n for n in names
    )
    assert have_positional, (
        "module must export a same-frame positional RMSD helper "
        "(e.g. rmsd_positional_symmetry_aware) for DiffDock-style "
        "crystal-frame inputs"
    )
    assert have_receptor_aligned, (
        "module must export a receptor-aligned RMSD helper "
        "(e.g. rmsd_receptor_aligned_symmetry_aware) for Boltz-style "
        "denovo-folded-protein inputs"
    )
