"""Backend simulation infrastructure for blop_sim."""

from typing import TYPE_CHECKING

from .core import SimBackend
from .simple import SimpleBackend

if TYPE_CHECKING:
    from .tes import TESBackend
    from .xrt import XRTBackend

__all__ = ["SimBackend", "SimpleBackend", "TESBackend", "XRTBackend"]


def __getattr__(name: str):
    # Lazy imports: xrt pulls in pyopencl (which can fail to import on hosts
    # without OpenCL devices) and tes pulls in torch — neither should break
    # `import blop_sim` for users of the other backends.
    if name == "XRTBackend":
        from .xrt import XRTBackend

        return XRTBackend
    if name == "TESBackend":
        from .tes import TESBackend

        return TESBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
