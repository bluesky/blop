"""XRT ray-tracing beam simulation backend."""

import numpy as np

from . import SimBackend
from .models.xrt_xas_model import build_beamline, build_histRGB, run_process


class XRTXASBackend(SimBackend):
    """XRT ray-tracing simulation backend.

    Uses the XRT package to perform realistic ray-tracing.
    Much slower than SimpleBackend but more physically accurate.
    """

    def __init__(self, noise: bool = False):
        """Initialize XRT backend."""
        super().__init__()
        self._beamline = None
        # self._limits = [[-0.6, 0.6], [-0.45, 0.45]]
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
        # mirror_radii = await self._get_mirror_radii()

        # Update XRT beamline mirror parameters
        # self._beamline.toroidMirror01.R = mirror_radii[0]  # Vertical mirror
        # self._beamline.toroidMirror02.R = mirror_radii[1]  # Horizontal mirror

        # Run ray tracing
        outDict = run_process(self._beamline)
        lb = outDict["SampleScreen_local"]

        # Build histogram from ray data
        hist2d, _, _ = build_histRGB(lb, lb, isScreen=True, shape=[400, 300])
        image = hist2d

        # Add noise if requested
        if self._noise:
            image += 1e-3 * np.abs(np.random.standard_normal(size=image.shape))

        return image

__all__ = ["XRTBackend"]
