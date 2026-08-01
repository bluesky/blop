"""XRT ray-tracing beam simulation backend."""

import numpy as np

from . import SimBackend
from .models.xrt_bioxas_model import build_beamline, build_histRGB, run_process


class XRTBIOXASBackend(SimBackend):
    """XRT ray-tracing simulation backend.

    Uses the XRT package to perform realistic ray-tracing through a KB mirror pair.
    Much slower than SimpleBackend but more physically accurate.
    """

    available_targets = ("PreM2Screen", "PreDBHRScreen", "SampleScreen")

    def __init__(self, noise: bool = False, target: str = "SampleScreen"):
        """Initialize XRT backend."""
        super().__init__()
        self._beamline = None
        self._limits = [[-2.5, 2.5], [-2.5, 2.5]]
        self._noise = noise
        self.target = target

    @property
    def target(self) -> str:
        """The screen currently rendered by detector acquisitions."""
        return self._target

    @target.setter
    def target(self, target: str) -> None:
        normalized_target = target.removesuffix("_local")
        if normalized_target not in self.available_targets:
            targets = ", ".join(self.available_targets)
            raise ValueError(f"Unknown BioXAS target screen {target!r}. Available targets are: {targets}.")
        self._target = normalized_target

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

        # Get KB mirror settings from devices
        mirror_radii, mirror_extra_pitches = await self._get_mirror_information()
        self._beamline.Mirror1.R = mirror_radii[0]  # Vertical mirror
        self._beamline.Mirror2.R = mirror_radii[1]  # Horizontal mirror
        self._beamline.Mirror1.extraPitch = mirror_extra_pitches[0]
        self._beamline.Mirror2.extraPitch = mirror_extra_pitches[1]

        # Get information for DBHR devices (pitch and roll for each mirror)
        dbhr_info = await self._get_dbhr_information()
        self._beamline.DBHR1.extraPitch = dbhr_info[0]
        self._beamline.DBHR1.extraRoll = dbhr_info[2]
        self._beamline.DBHR2.extraPitch = dbhr_info[1]
        self._beamline.DBHR2.extraRoll = dbhr_info[3]

        # Run ray tracing
        outDict = run_process(self._beamline)
        lb = outDict[f"{self.target}_local"]

        # Build histogram from ray data
        hist2d, _, _ = build_histRGB(lb, lb, limits=self._limits, isScreen=True, shape=[400, 300])
        image = hist2d

        # Add noise if requested
        if self._noise:
            image += 1e-3 * np.abs(np.random.standard_normal(size=image.shape))

        return image

    async def _get_mirror_information(self) -> tuple[list[float], list[float]]:
        """Get KB mirror radii and pitch offsets from registered devices.

        Returns:
            [R1, R2] and [extraPitch1, extraPitch2] where index 0 is the first mirror
            (vertical) and index 1 is the second mirror (horizontal).
        """
        # Default radii from xrt_bioxas_model.py
        radii = [7120000.0, 2500000.0]
        extra_pitches = [0.0, 0.0]

        for name, device in self._device_states.items():
            if device["type"] == "kb_mirror_xrt":
                state = await self._get_device_state(name)
                mirror_index = state["mirror_index"]
                if mirror_index < len(radii):
                    radii[mirror_index] = state["radius"]
                    extra_pitches[mirror_index] = state["extraPitch"]

        return radii, extra_pitches

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
