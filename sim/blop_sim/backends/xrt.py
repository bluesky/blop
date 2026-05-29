"""XRT ray-tracing beam simulation backend."""

import time
from collections import OrderedDict

import numpy as np
import xrt.backends.raycing as raycing
from bluesky.protocols import Readable
from xrt.backends.raycing import BeamLine

from .core import SimBackend
from blop.protocols import MovableHasName, Readable
from pathlib import Path


limits = [[-0.6, 0.6], [-0.45, 0.45]]


def build_histRGB(lb, gb, limits=None, isScreen=False, shape=None):
    if shape is None:
        shape = [256, 256]
    good = (lb.state == 1) | (lb.state == 2)
    if isScreen:
        x, y, z = lb.x[good], lb.z[good], lb.y[good]
    else:
        x, y, z = lb.x[good], lb.y[good], lb.z[good]
    goodlen = len(lb.x[good])
    hist2dRGB = np.zeros((shape[1], shape[0], 3), dtype=np.float64)
    hist2d = np.zeros((shape[1], shape[0]), dtype=np.float64)

    if limits is None and goodlen > 0:
        limits = np.array([[np.min(x), np.max(x)], [np.min(y), np.max(y)], [np.min(z), np.max(z)]])

    if goodlen > 0:
        beamLimits = [limits[1], limits[0]] or None
        flux = gb.Jss[good] + gb.Jpp[good]
        hist2d, _, _ = np.histogram2d(y, x, bins=[shape[1], shape[0]], range=beamLimits, weights=flux)
        hist2dRGB = None
    return hist2d, hist2dRGB, limits


class XRTBackend(SimBackend, Readable):
    """XRT ray-tracing simulation backend.

    Uses the XRT package to perform realistic ray-tracing through a KB mirror pair.
    Much slower than SimpleBackend but more physically accurate.
    """

    def __init__(self, file, limits=None, noise: bool = False):
        """Initialize XRT backend."""
        super().__init__()
        self._beamline = raycing.BeamLine(fileName=file)
        self._name = Path(file).stem
        self._limits = limits or [[-0.6, 0.6], [-0.45, 0.45]]
        self._noise = noise
        self._target = None
        beamLine = self._beamline
        self._elements = {k: beamLine.oesDict[v][0] for k, v in beamLine.oenamesToUUIDs.items()}
        self._ensure_beamline()

    def _ensure_beamline(self):
        """Build XRT beamline if not already built."""
        if self._beamline is None:
            raise ValueError("beamline has not been initialized")


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
            lb = [v for k, v in outDict.items() if self.target.name in k][0]
        else:
            print("warning: target is not set, please make sure you manage your triggering by setting a primary detector")
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
    def target(self, detector):
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

    @property
    def name(self):
        return self._name

    def __getitem__(self, key):
        if self.render is None:
            self.generate_beam()
        if key in self.render.keys():
            return [self.render[key]]
        return [v for k, v in self.render.items() if key in k and 'local' in k]

    @property
    def variables(self):
        return self._variables

    @property
    def elements(self):
        return self._elements

