"""KB mirror devices for beam simulation."""

from ophyd_async.core import StandardReadable, soft_signal_rw
from ophyd_async.core import StandardReadableFormat as Format

from ..backends import SimBackend


class KBMirrorSimple(StandardReadable):
    """KB mirror with jack position control (for SimpleBackend).
    
    Exposes two jack positions (upstream/downstream) that control the mirror curvature.
    Used with SimpleBackend for mathematical beam simulation.
    
    Args:
        backend: Simulation backend (should be SimpleBackend)
        orientation: "horizontal" or "vertical" 
        name: Device name
    """
    
    def __init__(self, backend: SimBackend, orientation: str = "horizontal", name: str = ""):
        self._backend = backend
        self._orientation = orientation
        
        # Jack position signals (CONFIG since they're not measurement outputs)
        with self.add_children_as_readables(Format.HINTED_SIGNAL):
            self.upstream = soft_signal_rw(float, 0.0)
            self.downstream = soft_signal_rw(float, 0.0)
        
        super().__init__(name=name)
        
        # Register with backend
        backend.register_device(
            device_name=name,
            device_type="kb_mirror_simple",
            get_state_callback=self._get_state,
        )
    
    async def _get_state(self) -> dict:
        """Get current mirror state for backend (async)."""
        return {
            "orientation": self._orientation,
            "upstream": await self.upstream.get_value(),
            "downstream": await self.downstream.get_value(),
        }


class KBMirrorXRT(StandardReadable):
    """KB mirror with curvature radius control (for XRTBackend).
    
    Exposes a single radius parameter that directly controls the XRT mirror R value.
    Used with XRTBackend for ray-tracing simulation.
    
    Args:
        backend: Simulation backend (should be XRTBackend)
        mirror_index: 0 for first mirror (vertical), 1 for second mirror (horizontal)
        initial_radius: Initial curvature radius in mm
        name: Device name
    """
    
    def __init__(
        self,
        backend: SimBackend,
        mirror_index: int,
        initial_radius: float = 30000.0,
        name: str = "",
    ):
        self._backend = backend
        self._mirror_index = mirror_index
        
        # Curvature radius signal
        with self.add_children_as_readables(Format.HINTED_SIGNAL):
            self.radius = soft_signal_rw(float, initial_radius)
        
        super().__init__(name=name)
        
        # Register with backend
        backend.register_device(
            device_name=name,
            device_type="kb_mirror_xrt",
            get_state_callback=self._get_state,
        )
    
    async def _get_state(self) -> dict:
        """Get current mirror state for backend (async)."""
        return {
            "mirror_index": self._mirror_index,
            "radius": await self.radius.get_value(),
        }


__all__ = ["KBMirrorSimple", "KBMirrorXRT"]
