"""SOTA rate divergence alarm.

Background: an earlier benchmark reported 0/39 <2 Å pose success while
DiffDock-L's published benchmark is 53% and Boltz-2's 44-60% (class-
dependent). A 10×+ gap between our pipeline's reported rate and the
reference literature rate should have raised an alarm immediately;
instead, we formulated increasingly elaborate "sampling ceiling"
hypotheses because we trusted the metric output over the paper's.

This module provides a utility that benchmark scripts can call just
before emitting their gate verdict:

    check_sota_rate_divergence(
        our_rate=our_hard_pass_rate,
        reference_rate=0.53,   # DiffDock-L PDBbind
        source="Corso et al. 2024 (DiffDock-L)",
        target_subset="PoseBench drug-like 5-target",
    )

If `our_rate < reference_rate / 10` (default ≤ 10× divergence
threshold), the utility writes a warning to stderr AND raises
`SOTARateDivergenceError`. Scripts can catch it, dump diagnostics, and
exit non-zero — blocking an accidental-silent-failure publication
pattern.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass


class SOTARateDivergenceError(RuntimeError):
    """Raised when a benchmark result diverges from published SOTA by
    >= the configured factor."""


@dataclass(frozen=True, slots=True)
class DivergenceCheckResult:
    our_rate: float
    reference_rate: float
    source: str
    target_subset: str
    ratio: float  # our_rate / reference_rate
    divergence_threshold: float
    divergence_flagged: bool
    message: str


def check_sota_rate_divergence(
    *,
    our_rate: float,
    reference_rate: float,
    source: str,
    target_subset: str,
    min_ratio: float = 0.1,
    raise_on_divergence: bool = True,
) -> DivergenceCheckResult:
    """Compare our benchmark rate against a published SOTA rate.

    Rationale: large divergence can indicate a metric, symmetry, or coordinate-frame bug.

    Arguments:
      our_rate         — fraction in [0, 1] of our pipeline's success
      reference_rate   — fraction in [0, 1] from the published paper
      source           — citation string for logging (e.g. "Corso
                         et al. 2024 (DiffDock-L)")
      target_subset    — description of our benchmark subset (so the
                         comparison is honest about scope)
      min_ratio        — our_rate / reference_rate must be ≥ this, else
                         flag divergence. Default 0.1 = 10× gap tolerated.
      raise_on_divergence — if True, raise SOTARateDivergenceError when
                         flagged. If False, just return the result with
                         `divergence_flagged=True` for the caller to
                         handle.

    Returns DivergenceCheckResult in all cases (no raise → return).
    """
    if not (0.0 <= our_rate <= 1.0):
        raise ValueError(f"our_rate must be in [0,1], got {our_rate}")
    if not (0.0 < reference_rate <= 1.0):
        raise ValueError(
            f"reference_rate must be in (0,1], got {reference_rate}"
        )

    ratio = our_rate / reference_rate
    flagged = ratio < min_ratio
    if flagged:
        msg = (
            f"⚠️ SOTA RATE DIVERGENCE: our {our_rate:.1%} vs "
            f"reference {reference_rate:.1%} on {target_subset}. "
            f"Ratio {ratio:.2f} < threshold {min_ratio}. "
            f"Reference: {source}. "
            f"Before assuming our "
            f"method is worse, verify the benchmark metric itself "
            f"(RMSD symmetry, coordinate frame, atom ordering)."
        )
    else:
        msg = (
            f"SOTA rate check OK: our {our_rate:.1%} vs reference "
            f"{reference_rate:.1%} on {target_subset} "
            f"(ratio {ratio:.2f} ≥ {min_ratio})."
        )

    result = DivergenceCheckResult(
        our_rate=our_rate,
        reference_rate=reference_rate,
        source=source,
        target_subset=target_subset,
        ratio=ratio,
        divergence_threshold=min_ratio,
        divergence_flagged=flagged,
        message=msg,
    )

    if flagged:
        print(msg, file=sys.stderr)
        if raise_on_divergence:
            raise SOTARateDivergenceError(msg)

    return result
