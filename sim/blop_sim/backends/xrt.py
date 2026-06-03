"""XRT ray-tracing beam simulation backend."""

import pickle
import shelve
import time
from collections import OrderedDict
from multiprocessing import Pool, shared_memory
from pathlib import Path

import numpy as np
import xrt.backends.raycing as raycing
from bluesky.protocols import Readable

from .core import SimBackend

limits = [[-0.6, 0.6], [-0.45, 0.45]]
cache_dir = Path.cwd() / "tmp" / "render"
cache_dir.mkdir(parents=True, exist_ok=True)


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


def _run_cached_process_from_file(beamLine, start_index):
    outDict = {}
    sequence = list(beamLine.flowU.items())[start_index:]
    print(f"starting from: {start_index}")
    for oeid, meth in sequence:
        oe = beamLine.oesDict[oeid][0]
        for func, fkwargs in meth.items():
            getattr(oe, func)(**fkwargs)
    for beamName, beamTag in beamLine.beamNamesDict.items():
        outDict[beamName] = beamLine.beamsDictU[beamTag[0]][beamTag[1]]

    return outDict


def _run_shelved_process_from_file(beamLine, start_index, shelf):
    filepath = f"tmp/render/iterbuf_{shelf}"
    with shelve.open(filepath, flag="c") as outDict:
        sequence = list(beamLine.flowU.items())[start_index:]
        touched = OrderedDict(sequence).keys()
        print(f"starting {shelf} from: {start_index}:")
        for oeid, meth in sequence:
            oe = beamLine.oesDict[oeid][0]
            for func, fkwargs in meth.items():
                getattr(oe, func)(**fkwargs)
        for beamName, beamTag in beamLine.beamNamesDict.items():
            if beamTag[0] not in touched:
                continue
            outDict[beamName] = beamLine.beamsDictU[beamTag[0]][beamTag[1]]

    return filepath


class XRTBackend(SimBackend, Readable):
    """XRT ray-tracing simulation backend.

    Uses the XRT package to perform realistic ray-tracing through a KB mirror pair.
    Much slower than SimpleBackend but more physically accurate.
    """

    def __init__(self, file, limits=None, noise: bool = False, n_iters=4, n_workers=4):
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
        self.n_iters = n_iters
        self.n_workers = n_workers
        self._cache_invalidator = [0] * len(beamLine.flowU.items())
        self.render = None

    def _ensure_beamline(self):
        """Build XRT beamline if not already built."""
        if self._beamline is None:
            raise ValueError("beamline has not been initialized")

    @staticmethod
    def _render_worker(stuple):
        beamLine, seed, minvalid_index = stuple
        print(f"worker {seed} executing")
        np.random.seed(seed=seed)
        beam = pickle.loads(beamLine.buf[: beamLine.size])
        # outDict = _run_cached_process_from_file(beamLine=beam, start_index=minvalid_index)
        outDict = _run_shelved_process_from_file(beamLine=beam, start_index=minvalid_index, shelf=seed)
        return outDict

    def generate_beam(self) -> np.ndarray:
        """Generate beam using XRT ray-tracing.

        Returns:
            2D numpy array with shape (300, 400)
        """
        self._ensure_beamline()
        if self.render is None:
            self._cache_invalidator = [0] * len(self._beamline.flowU.keys())

        with Pool(processes=self.n_workers) as pool:
            binary = pickle.dumps(self._beamline)
            shm = shared_memory.SharedMemory(create=True, size=len(binary))
            shm.buf[: len(binary)] = binary
            n_disp = self.n_iters - 1
            minvalid_index = [self._cache_invalidator.index(1) if 1 in self._cache_invalidator else 0] * n_disp
            b_refs = [shm] * n_disp
            result = pool.imap_unordered(self._render_worker, zip(b_refs, range(n_disp), minvalid_index, strict=True))
            outDict = [_run_shelved_process_from_file(self._beamline, start_index=minvalid_index[0], shelf=n_disp)]
            # outDict = _run_cached_process_from_file(self._beamline, start_index=minvalid_index[0])
            for out in result:
                # [outDict[k].concatenate(v) for k, v in out.items() if k in outDict.keys()]
                # del out
                outDict.append(out)
            shm.close()
            shm.unlink()
        self.render = outDict
        self._cache_invalidator = [0] * len(self._cache_invalidator)
        self._cache_invalidator[-1] = 1
        if self.target is not None:
            # lb = [v for k, v in outDict.items() if self.target.name in k][0]
            target = self.target.name
        else:
            print("warning: target is not set, please make sure you manage your triggering by setting a primary detector")
            with shelve.open(self.render[0]) as db:
                target = list(db.keys())[0]

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

    def read(self):
        if self.render is None:
            self.generate_beam()
        result = OrderedDict()
        for name, beam in self.render.items():
            hist2d, _, _ = build_histRGB(beam, beam, limits=self._limits, isScreen=True, shape=self._image_shape)
            result[name] = {"value": hist2d, "timestamp": time.time()}
        return result

    def describe(self):
        return OrderedDict([(k, {"source": k, "dtype": type(v["value"]), "shape": v.shape}) for k, v in self.read().items()])

    @property
    def name(self):
        return self._name

    def __getitem__(self, key):
        if self.render is None:
            self.generate_beam()

        outDict = {}
        with shelve.open(self.render[0]) as db:
            # if key in db.keys():
            #     outDict = {self.render[key]}
            for k in db.keys():
                if key in k and "global" not in k:
                    outDict[k] = db[k]

        for shelf in self.render[1:]:
            with shelve.open(shelf) as db:
                index = set(outDict.keys())
                for k in db.keys():
                    if k in index:
                        outDict[k].concatenate(db[k])
        return list(outDict.values())
        # return [v for k, v in self.render.items() if key in k and "global" not in k]

    @property
    def variables(self):
        return self._variables

    @property
    def elements(self):
        return self._elements

    def invalidate(self, id: int | str):
        if type(id) is int and id in range(len(self._cache_invalidator)):
            self._cache_invalidator[id] = 1

        print(f"invalidating cache element {id} ->", end="")
        if id in self.elements.keys():
            print("success")
            index = list(self._beamline.flowU.keys()).index(self._beamline.oenamesToUUIDs[id])
            self._cache_invalidator[index] = 1
