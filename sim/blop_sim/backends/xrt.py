"""XRT ray-tracing beam simulation backend."""

import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import xrt.backends.raycing as raycing
from bluesky.protocols import Readable

from .core import SimBackend

limits = [[-0.6, 0.6], [-0.45, 0.45]]
cache_dir = Path("/tmp/blop/sim/render")
cache_dir.mkdir(parents=True, exist_ok=True)


def build_histRGB(lb, gb, limits=None, isScreen=False, shape=None) -> tuple[np.array, np.array, np.array]:
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


def _run_cached_process_from_file(beamLine, start_index) -> dict[str, raycing.sources.beams.Beam]:
    outDict = {}
    sequence = list(beamLine.flowU.items())[start_index:]
    # print(f"starting from: {start_index}")
    for oeid, meth in sequence:
        oe = beamLine.oesDict[oeid][0]
        for func, fkwargs in meth.items():
            getattr(oe, func)(**fkwargs)
    for beamName, beamTag in beamLine.beamNamesDict.items():
        outDict[beamName] = beamLine.beamsDictU[beamTag[0]][beamTag[1]]

    return outDict


class XRTBackend(SimBackend, Readable):
    """XRT ray-tracing simulation backend.

    Uses the XRT package to perform realistic ray-tracing through a KB mirror pair.
    Much slower than SimpleBackend but more physically accurate.
    """

    def __init__(self, file, limits=None, noise: bool = False):
        """Initialize XRT backend. requiring at least the beamline description file"""
        super().__init__()
        self._beamline = raycing.BeamLine(fileName=file)
        self._name = Path(file).stem
        self._limits = limits or [[-0.6, 0.6], [-0.45, 0.45]]
        self._noise = noise
        self._target = None
        beamLine = self._beamline
        self._elements = {k: beamLine.oesDict[v][0] for k, v in beamLine.oenamesToUUIDs.items()}
        self._ensure_beamline()
        self._cache_invalidator = [0] * len(beamLine.flowU.items())
        self._render = None

    def _ensure_beamline(self):
        """legacy check to make sure beamline exists"""
        if self._beamline is None:
            raise ValueError("beamline has not been initialized")

    def generate_beam(self) -> np.ndarray:
        """Generate beam using XRT ray-tracing.

        Returns:
            2D numpy histogram array with shape of primary detector
        """
        self._ensure_beamline()
        if self._render is None:
            self._render = {}
            self._cache_invalidator = [0] * len(self._beamline.flowU.keys())

        minvalid_index = self._cache_invalidator.index(1) if 1 in self._cache_invalidator else 0
        outDict = _run_cached_process_from_file(self._beamline, start_index=minvalid_index)
        self._render.update(outDict)
        self._cache_invalidator = [0] * len(self._cache_invalidator)
        self._cache_invalidator[-1] = 1
        if self.target is not None:
            # lb = [v for k, v in outDict.items() if self.target.name in k][0]
            target = self.target.name
        else:
            print("warning: target is not set, please make sure you manage your triggering by setting a primary detector")
            target = list(self._render.keys())[0]

        lb = self[target][0]
        # Build histogram from ray data
        hist2d, _, _ = build_histRGB(lb, lb, limits=self._limits, isScreen=True, shape=self._image_shape)
        image = hist2d

        # Add noise if requested
        if self._noise:
            image += 1e-3 * np.abs(np.random.standard_normal(size=image.shape))

        return image

    def invalidate_cache(self):  # all zeroes or all ones doesn't matter but this is more "forceful" feeling
        self._cache_invalidator = [1] * len(self._beamline.flowU.keys())

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, detector):
        self._target = detector

    # readable interface
    def read(self) -> OrderedDict[str, dict]:
        if self._render is None:
            self.generate_beam()
        result = OrderedDict()
        for name, beam in self._render.items():
            hist2d, _, _ = build_histRGB(beam, beam, limits=self._limits, isScreen=True, shape=self._image_shape)
            result[name] = {"value": hist2d, "timestamp": time.time()}
        return result

    def describe(self):
        return OrderedDict([(k, {"source": k, "dtype": type(v["value"]), "shape": v.shape}) for k, v in self.read().items()])

    @property
    def name(self):
        return self._name

    def __getitem__(self, key):
        if self._render is None:
            self.generate_beam()
        return [v for k, v in self._render.items() if key in k and "global" not in k]

    @property
    def variables(self):
        return self._variables

    @property
    def elements(self):
        return self._elements

    def invalidate(self, id: int | str):
        if type(id) is int and id in range(len(self._cache_invalidator)):
            self._cache_invalidator[id] = 1

        # print(f"invalidating cache element {id} ->", end="")
        if id in self.elements.keys():
            # print("success")
            index = list(self._beamline.flowU.keys()).index(self._beamline.oenamesToUUIDs[id])
            self._cache_invalidator[index] = 1
