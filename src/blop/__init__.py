from .plans import acquire_baseline, default_acquire, optimize, optimize_step, sample_suggestions
from .xopt import XoptOptimizer

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"

__all__ = [
    "__version__",
    "acquire_baseline",
    "default_acquire",
    "optimize",
    "optimize_step",
    "sample_suggestions",
    "XoptOptimizer",
]
