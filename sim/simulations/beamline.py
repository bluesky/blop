import itertools
from collections import deque
from collections.abc import Generator, Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py  # type: ignore[import-untyped]
import numpy as np
import scipy as sp  # type: ignore[import-untyped]
from event_model import StreamRange, compose_resource, compose_stream_resource  # type: ignore[import-untyped]
from ophyd import Component as Cpt  # type: ignore[import-untyped]
from ophyd import Device, Kind, Signal  # type: ignore[import-untyped]
from ophyd.sim import NullStatus, new_uid  # type: ignore[import-untyped]
from ophyd.utils import make_dir_tree  # type: ignore[import-untyped]

from .handlers import get_beam_stats, ExternalFileReference


class DatabrokerDetector(Device):
    sum = Cpt(Signal, kind="hinted")
    max = Cpt(Signal, kind="normal")
    area = Cpt(Signal, kind="normal")
    cen_x = Cpt(Signal, kind="hinted")
    cen_y = Cpt(Signal, kind="hinted")
    wid_x = Cpt(Signal, kind="hinted")
    wid_y = Cpt(Signal, kind="hinted")
    image = Cpt(ExternalFileReference, kind="normal")
    image_shape = Cpt(Signal, value=(300, 400), kind="normal")
    noise = Cpt(Signal, kind="normal")

    def __init__(
        self, root_dir: str = "/tmp/blop/sim", verbose: bool = True, noise: bool = True, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)

        _ = make_dir_tree(datetime.now().year, base_path=root_dir)

        self._root_dir = root_dir
        self._verbose = verbose

        # Used for the emulated cameras only.
        self._img_dir = None

        # Resource/datum docs related variables.
        self._asset_docs_cache: deque[tuple[str, dict[str, Any]]] = deque()
        self._resource_document: dict[str, Any] | None = None
        self._datum_factory: Any | None = None
        self._dataset: h5py.Dataset | None = None
        self._h5file_desc: h5py.File | None = None
        self._counter: Iterator[int] | None = None

        self.noise.put(noise)

    def trigger(self) -> NullStatus:
        if not self._counter:
            raise RuntimeError("Counter not initialized, make sure to call stage() first.")
        if not self._dataset:
            raise RuntimeError("Dataset not initialized, make sure to call stage() first.")
        if not self._datum_factory:
            raise RuntimeError("Datum factory not initialized, make sure to call stage() first.")
        super().trigger()
        raw_image = self.generate_beam(noise=self.noise.get())

        current_frame = next(self._counter)

        self._dataset.resize((current_frame + 1, *self.image_shape.get()))
        self._dataset[current_frame, :, :] = raw_image

        datum_document = self._datum_factory(datum_kwargs={"frame": current_frame})
        self._asset_docs_cache.append(("datum", datum_document))

        stats = get_beam_stats(raw_image)
        self.image.put(datum_document["datum_id"])

        for attr in ["max", "sum", "cen_x", "cen_y", "wid_x", "wid_y"]:
            getattr(self, attr).put(stats[attr])

        super().trigger()

        return NullStatus()

    def stage(self) -> list[Any]:
        devices = super().stage()
        date = datetime.now()
        self._assets_dir = date.strftime("%Y/%m/%d")
        data_file = f"{new_uid()}.h5"

        self._resource_document, self._datum_factory, _ = compose_resource(
            start={"uid": "needed for compose_resource() but will be discarded"},
            spec="HDF5",
            root=self._root_dir,
            resource_path=str(Path(self._assets_dir) / Path(data_file)),
            resource_kwargs={},
        )

        if not self._resource_document:
            raise RuntimeError("Resource document not initialized.")

        self._data_file = str(Path(self._resource_document["root"]) / Path(self._resource_document["resource_path"]))

        # now discard the start uid, a real one will be added later
        self._resource_document.pop("run_start")
        self._asset_docs_cache.append(("resource", self._resource_document))

        self._h5file_desc = h5py.File(self._data_file, "x")
        group = self._h5file_desc.create_group("/entry")
        self._dataset = group.create_dataset(
            "image",
            data=np.full(fill_value=np.nan, shape=(1, *self.image_shape.get())),
            maxshape=(None, *self.image_shape.get()),
            chunks=(1, *self.image_shape.get()),
            dtype="float64",
            compression="lzf",
        )
        self._counter = itertools.count()
        return devices

    def unstage(self) -> list[Any]:
        devices = super().unstage()
        del self._dataset
        if self._h5file_desc:
            self._h5file_desc.close()
        self._resource_document = None
        self._datum_factory = None
        return devices

    def collect_asset_docs(self) -> Generator[tuple[str, dict[str, Any]], None, None]:
        items = list(self._asset_docs_cache)
        self._asset_docs_cache.clear()
        yield from items

    def generate_beam(self, noise: bool = True) -> np.ndarray:
        nx, ny = self.image_shape.get()

        x = np.linspace(-10, 10, ny)
        y = np.linspace(-10, 10, nx)
        X, Y = np.meshgrid(x, y)

        x0 = self.parent.kbh_ush.get() - self.parent.kbh_dsh.get()
        y0 = self.parent.kbv_usv.get() - self.parent.kbv_dsv.get()
        x_width = np.sqrt(0.2 + 5e-1 * (self.parent.kbh_ush.get() + self.parent.kbh_dsh.get() - 1) ** 2)
        y_width = np.sqrt(0.1 + 5e-1 * (self.parent.kbv_usv.get() + self.parent.kbv_dsv.get() - 2) ** 2)

        beam = np.exp(-0.5 * (((X - x0) / x_width) ** 4 + ((Y - y0) / y_width) ** 4)) / (
            np.sqrt(2 * np.pi) * x_width * y_width
        )

        mask = X > self.parent.ssa_inboard.get()
        mask &= X < self.parent.ssa_outboard.get()
        mask &= Y > self.parent.ssa_lower.get()
        mask &= Y < self.parent.ssa_upper.get()
        mask = sp.ndimage.gaussian_filter(mask.astype(float), sigma=1)

        image = beam * mask

        if noise:
            kx = np.fft.fftfreq(n=len(x), d=0.1)
            ky = np.fft.fftfreq(n=len(y), d=0.1)
            KX, KY = np.meshgrid(kx, ky)

            power_spectrum = 1 / (1e-2 + KX**2 + KY**2)

            white_noise = 1e-3 * np.random.standard_normal(size=X.shape)
            pink_noise = 1e-3 * np.real(np.fft.ifft2(power_spectrum * np.fft.fft2(np.random.standard_normal(size=X.shape))))
            # background = 5e-3 * (X - Y) / X.max()

            image += white_noise + pink_noise

        return image


class TiledDetector(Device):
    sum = Cpt(Signal, kind=Kind.hinted)
    max = Cpt(Signal, kind=Kind.normal)
    area = Cpt(Signal, kind=Kind.normal)
    cen_x = Cpt(Signal, kind=Kind.hinted)
    cen_y = Cpt(Signal, kind=Kind.hinted)
    wid_x = Cpt(Signal, kind=Kind.hinted)
    wid_y = Cpt(Signal, kind=Kind.hinted)
    image = Cpt(ExternalFileReference, kind=Kind.omitted)
    image_shape = Cpt(Signal, value=(300, 400), kind=Kind.omitted)
    noise = Cpt(Signal, kind=Kind.normal)

    def __init__(self, root_dir: str = "/tmp/blop/sim", verbose: bool = True, noise: bool = True, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        _ = make_dir_tree(datetime.now().year, base_path=root_dir)

        self._root_dir = root_dir
        self._verbose = verbose

        # Used for the emulated cameras only.
        self._img_dir = None

        # Resource/datum docs related variables.
        self._asset_docs_cache: deque[tuple[str, dict[str, Any]]] = deque()
        self._stream_resource_document: dict[str, Any] | None = None
        self._stream_datum_factory: Any | None = None
        self._dataset: h5py.Dataset | None = None
        self._counter: Iterator[int] | None = None
        self.noise.put(noise)

    def trigger(self):
        if not self._counter:
            raise RuntimeError("Counter not initialized, make sure to call stage() first.")
        if not self._dataset:
            raise RuntimeError("Dataset not initialized, make sure to call stage() first.")
        if not self._stream_datum_factory:
            raise RuntimeError("Datum factory not initialized, make sure to call stage() first.")
        super().trigger()
        raw_image = self.generate_beam(noise=self.noise.get())

        current_frame = next(self._counter)

        self._dataset.resize((current_frame + 1, *self.image_shape.get()))

        self._dataset[current_frame, :, :] = raw_image

        stream_datum_document = self._stream_datum_factory(
            StreamRange(start=current_frame, stop=current_frame + 1),
        )
        self._asset_docs_cache.append(("stream_datum", stream_datum_document))

        stats = get_beam_stats(raw_image)

        for attr in ["max", "sum", "cen_x", "cen_y", "wid_x", "wid_y"]:
            getattr(self, attr).put(stats[attr])

        super().trigger()

        return NullStatus()

    def _generate_file_path(self, date_template="%Y/%m/%d"):
        date = datetime.now()
        assets_dir = date.strftime(date_template)
        data_file = f"{new_uid()}.h5"

        return Path(self._root_dir) / Path(assets_dir) / Path(data_file)

    def stage(self):
        devices = super().stage()

        self._asset_docs_cache.clear()
        full_path = self._generate_file_path()
        image_shape = self.image_shape.get()

        uri = f"file://localhost/{str(full_path).strip('/')}"

        (
            self._stream_resource_document,
            self._stream_datum_factory,
        ) = compose_stream_resource(
            mimetype="application/x-hdf5",
            uri=uri,
            data_key=self.image.name,
            parameters={
                "chunk_shape": (1, *image_shape),
                "dataset": "/entry/image",
            },
        )

        self._data_file = full_path

        self._asset_docs_cache.append(("stream_resource", self._stream_resource_document))

        self._h5file_desc = h5py.File(self._data_file, "x")
        group = self._h5file_desc.create_group("/entry")
        self._dataset = group.create_dataset(
            "image",
            data=np.full(fill_value=np.nan, shape=(1, *image_shape)),
            maxshape=(None, *image_shape),
            chunks=(1, *image_shape),
            dtype="float64",
            compression="lzf",
        )
        self._counter = itertools.count()
        return devices

    def unstage(self) -> list[Any]:
        devices = super().unstage()
        del self._dataset
        if self._h5file_desc:
            self._h5file_desc.close()
        self._resource_document = None
        self._datum_factory = None
        return devices

    def describe(self):
        res = super().describe()
        res[self.image.name] = {
            "shape": [1, *self.image_shape.get()],
            "external": "STREAM:",
            "source": "sim",
            "dtype": "array",
            "dtype_numpy": np.dtype(np.float64).str,
        }  # <i8
        return res

    def collect_asset_docs(self):
        items = list(self._asset_docs_cache)
        self._asset_docs_cache.clear()
        yield from items

    def generate_beam(self, noise: bool = True):
        nx, ny = self.image_shape.get()

        x = np.linspace(-10, 10, ny)
        y = np.linspace(-10, 10, nx)
        X, Y = np.meshgrid(x, y)

        x0 = self.parent.kbh_ush.get() - self.parent.kbh_dsh.get()
        y0 = self.parent.kbv_usv.get() - self.parent.kbv_dsv.get()
        x_width = np.sqrt(0.2 + 5e-1 * (self.parent.kbh_ush.get() + self.parent.kbh_dsh.get() - 1) ** 2)
        y_width = np.sqrt(0.1 + 5e-1 * (self.parent.kbv_usv.get() + self.parent.kbv_dsv.get() - 2) ** 2)

        beam = np.exp(-0.5 * (((X - x0) / x_width) ** 4 + ((Y - y0) / y_width) ** 4)) / (
            np.sqrt(2 * np.pi) * x_width * y_width
        )

        mask = X > self.parent.ssa_inboard.get()
        mask &= X < self.parent.ssa_outboard.get()
        mask &= Y > self.parent.ssa_lower.get()
        mask &= Y < self.parent.ssa_upper.get()
        mask = sp.ndimage.gaussian_filter(mask.astype(float), sigma=1)

        image = beam * mask

        if noise:
            kx = np.fft.fftfreq(n=len(x), d=0.1)
            ky = np.fft.fftfreq(n=len(y), d=0.1)
            KX, KY = np.meshgrid(kx, ky)

            power_spectrum = 1 / (1e-2 + KX**2 + KY**2)

            white_noise = 1e-3 * np.random.standard_normal(size=X.shape)
            pink_noise = 1e-3 * np.real(np.fft.ifft2(power_spectrum * np.fft.fft2(np.random.standard_normal(size=X.shape))))
            # background = 5e-3 * (X - Y) / X.max()

            image += white_noise + pink_noise

        return image


class DatabrokerBeamline(Device):
    det = Cpt(DatabrokerDetector)

    kbh_ush = Cpt(Signal, kind=Kind.hinted)
    kbh_dsh = Cpt(Signal, kind=Kind.hinted)
    kbv_usv = Cpt(Signal, kind=Kind.hinted)
    kbv_dsv = Cpt(Signal, kind=Kind.hinted)

    ssa_inboard = Cpt(Signal, value=-5.0, kind=Kind.hinted)
    ssa_outboard = Cpt(Signal, value=5.0, kind=Kind.hinted)
    ssa_lower = Cpt(Signal, value=-5.0, kind=Kind.hinted)
    ssa_upper = Cpt(Signal, value=5.0, kind=Kind.hinted)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class TiledBeamline(Device):
    det = Cpt(TiledDetector)

    kbh_ush = Cpt(Signal, kind=Kind.hinted)
    kbh_dsh = Cpt(Signal, kind=Kind.hinted)
    kbv_usv = Cpt(Signal, kind=Kind.hinted)
    kbv_dsv = Cpt(Signal, kind=Kind.hinted)

    ssa_inboard = Cpt(Signal, value=-5.0, kind=Kind.hinted)
    ssa_outboard = Cpt(Signal, value=5.0, kind=Kind.hinted)
    ssa_lower = Cpt(Signal, value=-5.0, kind=Kind.hinted)
    ssa_upper = Cpt(Signal, value=5.0, kind=Kind.hinted)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
