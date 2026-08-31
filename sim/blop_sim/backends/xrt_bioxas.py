"""XRT ray-tracing beam simulation backend."""

import numpy as np

from . import SimBackend
from .models.xrt_bioxas_model import build_beamline, build_histRGB, run_process


class XRTBIOXASBackend(SimBackend):
    """XRT ray-tracing simulation backend.

    Uses the XRT package to perform realistic ray-tracing through a KB mirror pair.
    Much slower than SimpleBackend but more physically accurate.
    """

    def __init__(self, noise: bool = False):
        """Initialize XRT backend."""
        super().__init__()
        self._beamline = None
        self._limits = [[-2.5, 2.5], [-2.5, 2.5]]
        self._noise = noise

    def _ensure_beamline(self):
        """Build XRT beamline if not already built."""
        if self._beamline is None:
            self._beamline = build_beamline()

    async def generate_beam(self) -> np.ndarray:
        """Generate beam using XRT ray-tracing.

        Returns:
            2D numpy array with shape (300, 400)
        """
        self._ensure_beamline()

        # Get KB mirror radii from devices
        mirror_radii = await self._get_mirror_radii()
        self._beamline.Mirror1.R = mirror_radii[0]  # Vertical mirror
        self._beamline.Mirror2.R = mirror_radii[1]  # Horizontal mirror

        # Get information for DBHR devices (pitch and roll for each mirror)
        dbhr_info = await self._get_dbhr_information()
        self._beamline.DBHR1.extraPitch = dbhr_info[0]
        self._beamline.DBHR1.extraRoll = dbhr_info[2]
        self._beamline.DBHR2.extraPitch = dbhr_info[1]
        self._beamline.DBHR2.extraRoll = dbhr_info[3]

        # Run ray tracing
        outDict = run_process(self._beamline)
        lb = outDict["SampleScreen_local"]

        # Build histogram from ray data
        hist2d, _, _ = build_histRGB(lb, lb, limits=self._limits, isScreen=True, shape=[400, 300])
        image = hist2d

        # Add noise if requested
        if self._noise:
            image += 1e-3 * np.abs(np.random.standard_normal(size=image.shape))

        return image

    async def _get_mirror_radii(self) -> list[float]:
        """Get KB mirror radii from registered devices.

        Returns:
            [R1, R2] where R1 is first mirror (vertical), R2 is second mirror (horizontal)
        """
        # Default radii from xrt_bioxas_model.py
        radii = [7120000.0, 2500000.0]

        for name, device in self._device_states.items():
            if device["type"] == "kb_mirror_xrt":
                state = await self._get_device_state(name)
                mirror_index = state["mirror_index"]
                radius = state["radius"]
                if mirror_index < len(radii):
                    radii[mirror_index] = radius

        return radii

    async def _get_dbhr_information(self) -> list[float]:
        pitch = [None, None]
        roll = [None, None]
        for name, device in self._device_states.items():
            if device["type"] == "dbhm_xrt":
                state = await self._get_device_state(name)
                mirror_index = state["optic_index"]
                pitch[mirror_index] = state["extraPitch"]
                roll[mirror_index] = state["extraRoll"]
        return pitch + roll


__all__ = ["XRTBIOXASBackend"]
