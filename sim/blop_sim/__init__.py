"""blop_sim: Simulation devices for BLOP documentation and tutorials.

This package provides ophyd-async simulation devices for demonstrating
Bayesian optimization with BLOP. It includes:

- Two simulation backends: SimpleBackend (mathematical) and XRTBackend (ray-tracing)
- KB mirror devices with backend-specific control parameters
- Slit device for aperture control
- Detector device that generates beam images (statistics computed in evaluation functions)

Example usage with XRT backend:

    from blop_sim.backends.xrt import XRTBackend
    from blop_sim.devices import KBMirrorXRT, DetectorDevice
    from ophyd_async.core import DirectoryProvider
    
    backend = XRTBackend()
    kbv = KBMirrorXRT(backend, mirror_index=0, initial_radius=38000, name="kbv")
    kbh = KBMirrorXRT(backend, mirror_index=1, initial_radius=21000, name="kbh")
    det = DetectorDevice(backend, DirectoryProvider("/tmp/blop/sim"), name="det")

Example usage with Simple backend:

    from blop_sim.backends.simple import SimpleBackend
    from blop_sim.devices import KBMirrorSimple, SlitDevice, DetectorDevice
    from ophyd_async.core import DirectoryProvider
    
    backend = SimpleBackend()
    kbh = KBMirrorSimple(backend, orientation="horizontal", name="kbh")
    kbv = KBMirrorSimple(backend, orientation="vertical", name="kbv")
    slit = SlitDevice(backend, name="ssa")
    det = DetectorDevice(backend, DirectoryProvider("/tmp/blop/sim"), name="det")
"""

# Backend exports
from .backends.simple import SimpleBackend
from .backends.xrt import XRTBackend

# Keep handlers for HDF5 file reading
from .handlers import HDF5Handler, get_beam_stats

__all__ = [
    "SimpleBackend",
    "XRTBackend",
    "HDF5Handler",
    "get_beam_stats",
]
