"""Fixed Mask devices for XRTBackend."""

from ophyd_async.core import StandardReadable, soft_signal_rw
from ophyd_async.core import StandardReadableFormat as Format

from ...backends import SimBackend


class FixedMask(StandardReadable):
    """Fixed mask position control (for XRTBackend).

    Exposes the coordinates that directly control the XRT fixed mask position.
    Used with XRTBackend for ray-tracing simulation.

    Args:
        backend: Simulation backend (should be XRTBackend)
        position: list of floats that define the [x, y, z] position of the mask
        name: Device name
    """

    def __init__(
        self,
        backend: SimBackend,
        initial_position: list[float],
        name: str = "",
    ):
        self._backend = backend

        # Curvature radius signal
        with self.add_children_as_readables(Format.HINTED_SIGNAL):
            self.x = soft_signal_rw(float, initial_position[0])
            self.y = soft_signal_rw(float, initial_position[1])
            self.z = soft_signal_rw(float, initial_position[2])

        super().__init__(name=name)

        # Register with backend
        backend.register_device(
            device_name=name,
            device_type="fixed_mask_xrt",
            get_state_callback=self._get_state,
        )

    async def _get_state(self) -> dict:
        """Get current mirror state for backend (async)."""
        # position = self.position.get_value()
        return {
            "x": await self.x.get_value(),
            "y": await self.y.get_value(),
            "z": await self.z.get_value(),
        }


__all__ = ["FixedMask"]
