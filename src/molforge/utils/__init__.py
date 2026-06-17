"""molforge.utils — cross-cutting helpers.

Exports:
  run_streaming_subprocess — subprocess.run() replacement that streams
  stdout/stderr to disk and survives timeouts.
"""

from .sota_rate_check import (
    DivergenceCheckResult,
    SOTARateDivergenceError,
    check_sota_rate_divergence,
)
from .subprocess_helpers import (
    StreamingSubprocessResult,
    run_streaming_subprocess,
    safe_env,
)

__all__ = [
    "DivergenceCheckResult",
    "SOTARateDivergenceError",
    "StreamingSubprocessResult",
    "check_sota_rate_divergence",
    "run_streaming_subprocess",
    "safe_env",
]
