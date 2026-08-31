"""KB mirror devices for XRTBackend."""

from ophyd_async.core import StandardReadable, soft_signal_rw
from ophyd_async.core import StandardReadableFormat as Format

from ...backends import SimBackend


class DBHR(StandardReadable):
    """DCM with second crystal roll control (for XRTBioXAS).

    Exposes the extraRoll and extraPitch of the second crystal of the DBHR
    Used with XRTBioXAS for ray-tracing simulation.

    Args:
        backend: Simulation backend (should be XRTBioXAS)
        extraPitch: Additional pitch of second crystal in mm after initial alignment
        extraRoll: Additional roll of second crystal in mm after initial alignment
        name: Device name
    """

    def __init__(
        self,
        backend: SimBackend,
        optic_index: int,
        extraPitch: float = 0,
        extraRoll: float = 0,
        name: str = "",
    ):
        self._optic_index = optic_index
        self._backend = backend

        # second crystal roll
        with self.add_children_as_readables(Format.HINTED_SIGNAL):
            self.extraPitch = soft_signal_rw(float, extraPitch)
            self.extraRoll = soft_signal_rw(float, extraRoll)

        super().__init__(name=name)

        # Register with backend
        backend.register_device(
            device_name=name,
            device_type="dbhm_xrt",
            get_state_callback=self._get_state,
        )

    async def _get_state(self) -> dict:
        """Get current DCM state for backend (async)."""
        return {
            "optic_index": self._optic_index,
            "extraPitch": await self.extraPitch.get_value(),
            "extraRoll": await self.extraRoll.get_value(),
        }


__all__ = ["DBHR"]
