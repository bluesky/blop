"""Slit device for beam simulation."""

from ophyd_async.core import StandardReadable, soft_signal_rw
from ophyd_async.core import StandardReadableFormat as Format

from ..backends import SimBackend


class SlitDevice(StandardReadable):
    """Four-blade slit device for aperture control.

    Controls a rectangular aperture that clips the beam. The slit is defined by
    four blade positions that create a window in the beam path.

    Args:
        backend: Simulation backend
        name: Device name
    """

    def __init__(self, backend: SimBackend, name: str = ""):
        self._backend = backend

        # Four blade positions (CONFIG since they're not measurement outputs)
        with self.add_children_as_readables(Format.CONFIG_SIGNAL):
            self.inboard = soft_signal_rw(float, -5.0)
            self.outboard = soft_signal_rw(float, 5.0)
            self.lower = soft_signal_rw(float, -5.0)
            self.upper = soft_signal_rw(float, 5.0)

        super().__init__(name=name)

        # Register with backend
        backend.register_device(
            device_name=name,
            device_type="slit",
            get_state_callback=self._get_state,
        )

    async def _get_state(self) -> dict:
        """Get current slit state for backend (async)."""
        return {
            "inboard": await self.inboard.get_value(),
            "outboard": await self.outboard.get_value(),
            "lower": await self.lower.get_value(),
            "upper": await self.upper.get_value(),
        }


__all__ = ["SlitDevice"]
