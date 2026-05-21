"""XRT ray-tracing beam simulation backend."""

import numpy as np

from . import SimBackend
from .models.xrt_kb_model import build_beamline, build_histRGB, run_process
from xrt.backends.raycing import BeamLine



class KBBackend(SimBackend):
    """XRT ray-tracing simulation backend.

    Uses the XRT package to perform realistic ray-tracing through a KB mirror pair.
    Much slower than SimpleBackend but more physically accurate.
    """

    def __init__(self, noise: bool = False):
        """Initialize XRT backend."""
        super().__init__()
        self._beamline = None
        self._limits = [[-0.6, 0.6], [-0.45, 0.45]]
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

        # Update XRT beamline mirror parameters
        self._beamline.toroidMirror01.R = mirror_radii[0]  # Vertical mirror
        self._beamline.toroidMirror02.R = mirror_radii[1]  # Horizontal mirror

        # Run ray tracing
        outDict = run_process(self._beamline)
        lb = outDict["screen01beamLocal01"]

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
        # Default radii from xrt_kb_model.py
        radii = [38245.0, 21035.0]

        for name, device in self._device_states.items():
            if device["type"] == "kb_mirror_xrt":
                state = await self._get_device_state(name)
                mirror_index = state["mirror_index"]
                radius = state["radius"]
                if mirror_index < len(radii):
                    radii[mirror_index] = radius

        return radii


class XRTBackend(SimBackend):
    """XRT ray-tracing simulation backend.

    Uses the XRT package to perform realistic ray-tracing through a KB mirror pair.
    Much slower than SimpleBackend but more physically accurate.
    """

    def __init__(self, file=None, limits=None, noise: bool = False):
        """Initialize XRT backend."""
        super().__init__()
        self._beamline = raycing.BeamLine(fileName=fileName)
        self._limits = limits or [[-0.6, 0.6], [-0.45, 0.45]]
        self._noise = noise

    def _ensure_beamline(self):
        """Build XRT beamline if not already built."""
        if self._beamline is None:
            raise ValueError("beamline has not been initialized")

        if self._elements is None:
            beamLine = self._beamline
            reverse_ind = dict(
                zip([v[0] for v in beamLine.beamNamesDict.values()], [*beamLine.beamNamesDict.keys()], strict=True)
            )
            self._dofs = {reverse_ind[k]: vars(v[0]) for k, v in beamLine.oesDict.items()}

    async def generate_beam(self) -> np.ndarray:
        """Generate beam using XRT ray-tracing.

        Returns:
            2D numpy array with shape (300, 400)
        """
        self._ensure_beamline()

        # Get KB mirror radii from devices
        mirror_radii = await self._get_mirror_radii()

        # Update XRT beamline mirror parameters
        self._beamline.toroidMirror01.R = mirror_radii[0]  # Vertical mirror
        self._beamline.toroidMirror02.R = mirror_radii[1]  # Horizontal mirror

        # Run ray tracing
        outDict = run_process(self._beamline)
        lb = outDict["screen01beamLocal01"]

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
        # Default radii from xrt_kb_model.py
        radii = [38245.0, 21035.0]

        for name, device in self._device_states.items():
            if device["type"] == "kb_mirror_xrt":
                state = await self._get_device_state(name)
                mirror_index = state["mirror_index"]
                radius = state["radius"]
                if mirror_index < len(radii):
                    radii[mirror_index] = radius

        return radii


__all__ = ["XRTBackend"]
