"""XRT ray-tracing beam simulation backend."""

import numpy as np
import random

from . import SimBackend
from .models.xrt_bmm_model import build_beamline, build_histRGB, run_process, set_energy

class XRTBMMBackend(SimBackend):
    """XRT ray-tracing simulation backend.

    Uses the XRT package to perform realistic ray-tracing to simulate the BMM beamline.
    """

    def __init__(self, noise: bool = False, dcm_offset = random.uniform(-0.00009, 0.00009)):
        """Initialize XRT backend."""
        super().__init__()
        self._beamline = build_beamline()
        self._limits = [[-5, 5], [-5, 5]]
        self._dcm_offset = dcm_offset
        self._noise = noise

    def _ensure_beamline(self):
        """Build XRT beamline if not already built."""
        if self._beamline is None:
            self._beamline = build_beamline()

    def change_energy(self, ev):
        set_energy(self._beamline, ev, fixedExit=30.0)

    async def generate_beam(self) -> np.ndarray:
        """Generate beam using XRT ray-tracing.

        Returns:
            2D numpy array with shape (300, 400)
        """
        self._ensure_beamline()

        # get and set DCM roll
        dcm_roll = await self._get_dcm_roll()
        self._beamline.DCM.cryst2roll = dcm_roll + self._dcm_offset

        # get and set TFM (m2) yaw
        m2_yaw = await self._get_m2_yaw()
        self._beamline.M2_TFM.yaw = m2_yaw

        # get and set TFM (m2) lateral (x-position)
        m2_lateral = await self._get_m2_lateral()
        self._beamline.M2_TFM.center[0] = m2_lateral

        # Run ray tracing
        outDict = run_process(self._beamline)
        lb = outDict["XAS_SAMPLE_local"]

        # Build histogram from ray data
        image, _, _ = build_histRGB(lb, lb, limits=self._limits, isScreen=True, shape=[400, 300])

        # Add noise if requested (pretty sure XRT already adds noise? don't think this is needed)
        if self._noise:
            image += 1e-3 * np.abs(np.random.standard_normal(size=image.shape))

        return image

    async def _get_dcm_roll(self) -> float:
        for name, device in self._device_states.items():
            if device["type"] == "dcm_xrt":
                state = await self._get_device_state(name)
                return state['roll']
    
    async def _get_m2_yaw(self) -> float:
        for name, device in self._device_states.items():
            if device["type"] == "toroidal_mirror_xrt":
                state = await self._get_device_state(name)
                return state["yaw"]
    
    async def _get_m2_lateral(self) -> float:
        for name, device in self._device_states.items():
            if device["type"] == "toroidal_mirror_xrt":
                state = await self._get_device_state(name)
                return state['lateral']


__all__ = ["XRTBMMBackend"]
