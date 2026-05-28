"""XRT ray-tracing beam simulation backend."""

import time
from collections import OrderedDict

import numpy as np
import xrt.backends.raycing as raycing
from bluesky.protocols import Readable
from xrt.backends.raycing import BeamLine

from ..devices.xrt import element_to_variables, i
from . import SimBackend
from .models.xrt_kb_model import build_beamline, build_histRGB, run_process


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


class XRTBackend(SimBackend, Readable):
    """XRT ray-tracing simulation backend.

    Uses the XRT package to perform realistic ray-tracing through a KB mirror pair.
    Much slower than SimpleBackend but more physically accurate.
    """

    def __init__(self, file=None, limits=None, noise: bool = False):
        """Initialize XRT backend."""
        super().__init__()
        self._beamline = raycing.BeamLine(fileName=file)
        self._limits = limits or [[-0.6, 0.6], [-0.45, 0.45]]
        self._noise = noise
        self._target = None
        self._ensure_beamline()

    def _ensure_beamline(self):
        """Build XRT beamline if not already built."""
        if self._beamline is None:
            raise ValueError("beamline has not been initialized")

        if self._elements is None:
            beamLine = self._beamline
            self._elements = {k: beamLine.oesDict[v][0] for k, v in beamLine.oenamesToUUIDs.items()}
            self._variables = {k: element_to_variables(v, k) for k, v in self._elements.items()}

    def generate_beam(self) -> np.ndarray:
        """Generate beam using XRT ray-tracing.

        Returns:
            2D numpy array with shape (300, 400)
        """
        self._ensure_beamline()

        # Run ray tracing
        outDict = raycing.run_process_from_file(self._beamline)
        self.render = outDict
        if self.target is not None:
            print("warning: target is not set, please make sure you manage your triggering by setting a primary detector")
            lb = [v for k, v in outDict.items() if self.target in k][0]
        else:
            lb = outDict.values()[0]

        # Build histogram from ray data
        hist2d, _, _ = build_histRGB(lb, lb, limits=self._limits, isScreen=True, shape=self._image_shape)
        image = hist2d

        # Add noise if requested
        if self._noise:
            image += 1e-3 * np.abs(np.random.standard_normal(size=image.shape))

        return image

    @property
    def target(self):
        return self._target

    @target.setter
    def register_target(self, detector):
        self._target = detector

    def read(self):
        if self.render is None:
            self.generate_beam()
        result = OrderedDict
        for name, beam in self.render.items():
            hist2d, _, _ = build_histRGB(beam, beam, limits=self._limits, isScreen=True, shape=self._image_shape)
            result[name] = {"value": hist2d, "timestamp": time.time()}
        return result

    def describe(self):
        return OrderedDict([(k, {"source": k, "dtype": type(v['value']), 'shape': v.shape}) for k, v in self.read().items()])

    def __getitem__(self, key):
        if self.render is None:
            self.generate_beam()
        if key in self.render.keys():
            return self.render[key]
        return [v for k, v in self.render.items() if key in k and 'local' in k]

    @property
    def variables(self):
        return self._variables

    @property
    def elements(self):
        return self._elements
