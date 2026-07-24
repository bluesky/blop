"""blop_sim: Simulation devices for BLOP documentation and tutorials."""

from typing import TYPE_CHECKING

from .backends.simple import SimpleBackend

if TYPE_CHECKING:
    from .backends.tes import TESBackend
    from .backends.xrt import XRTBackend

__all__ = [
    "SimpleBackend",
    "TESBackend",
    "XRTBackend",
]


def __getattr__(name: str):
    # Lazy imports — see blop_sim.backends.__getattr__.
    if name in ("XRTBackend", "TESBackend"):
        from . import backends

        return getattr(backends, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
