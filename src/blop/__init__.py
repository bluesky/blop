"""A bridge between optimization algorithms and Bluesky."""

from .plan_stubs import list_scan_in_run
from .plans import (
    acquire_baseline,
    default_acquire,
    optimize,
    optimize_in_run,
    optimize_step,
    sample_suggestions,
)

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"

__all__ = [
    "__version__",
    "acquire_baseline",
    "default_acquire",
    "list_scan_in_run",
    "optimize",
    "optimize_in_run",
    "optimize_step",
    "sample_suggestions",
]
