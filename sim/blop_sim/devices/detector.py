"""Detector device for beam simulation - images only, NO statistics."""

import itertools
from collections import deque
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import h5py  # type: ignore[import-untyped]
import numpy as np
from event_model import DataKey, StreamDatum, StreamRange, StreamResource, compose_stream_resource  # type: ignore[import-untyped]
from ophyd_async.core import (
    DetectorController,
    DetectorWriter,
    PathProvider,
    StandardDetector,
)

from ..backends import SimBackend


class SimplePathProvider(PathProvider):
    """Simple path provider for detector files.

    Args:
        directory_path: Base directory for detector files
    """

    def __init__(self, directory_path: Path | str):
        self.directory_path = Path(directory_path)
        self._counter = itertools.count()

    def __call__(self, device_name: str | None = None) -> Path:
        """Generate path for next detector file."""
        return self.directory_path / f"det_{next(self._counter):06d}.h5"


class SimDetectorController(DetectorController):
    """Controller for simulated detector - generates images only."""

    def __init__(self, backend: SimBackend):
        self._backend = backend

    def get_deadtime(self, exposure: float | None) -> float:
        """Detector has no deadtime (instant acquisition)."""
        return 0.0

    async def prepare(self, trigger_info: Any) -> None:
        """Prepare for acquisition with trigger info."""
        # Software triggered detector, no preparation needed
        pass

    async def arm(self) -> None:
        """Prepare for acquisition."""
        # Software triggered, no arming needed
        pass

    async def wait_for_idle(self):
        """Wait for acquisition to complete."""
        # Software triggered, always idle
        pass

    async def disarm(self):
        """Clean up after acquisition."""
        pass


class SimDetectorWriter(DetectorWriter):
    """Writer for detector with Tiled streaming."""

    def __init__(
        self,
        backend: SimBackend,
        path_provider: PathProvider,
    ):
        self._backend = backend
        self.path_provider = path_provider
        self._asset_docs_cache: deque[tuple[str, StreamResource | StreamDatum]] = deque()
        self._h5file: h5py.File | None = None
        self._dataset: h5py.Dataset | None = None
        self._counter: itertools.count[int] | None = None
        self._stream_datum_factory: Any | None = None
        self._last_index = 0

    async def open(self, name: str | None = None, exposures_per_event: int = 1) -> dict[str, DataKey]:
        """Open HDF5 file and setup stream resources.

        Args:
            name: Name of detector (optional, uses name_provider if not given)
            exposures_per_event: Number of exposures per event
        """
        # Create directory structure
        full_path = self.path_provider()
        full_path.parent.mkdir(parents=True, exist_ok=True)

        # Get image shape from backend
        image_shape = self._backend.get_image_shape()

        # Create HDF5 file
        self._h5file = h5py.File(full_path, "x")
        group = self._h5file.create_group("/entry")
        self._dataset = group.create_dataset(
            "image",
            data=np.full(fill_value=np.nan, shape=(1, *image_shape)),
            maxshape=(None, *image_shape),
            chunks=(1, *image_shape),
            dtype="float64",
            compression="lzf",
        )

        self._counter = itertools.count()

        # Create stream resource
        uri = f"file://localhost/{str(full_path).strip('/')}"
        (
            stream_resource_doc,
            self._stream_datum_factory,
        ) = compose_stream_resource(
            mimetype="application/x-hdf5",
            uri=uri,
            data_key="image",
            parameters={
                "chunk_shape": (1, *image_shape),
                "dataset": "/entry/image",
            },
        )

        self._asset_docs_cache.append(("stream_resource", stream_resource_doc))

        # Return describe dictionary
        return {
            "image": {
                "source": "sim",
                "shape": [1, *image_shape],
                "dtype": "array",
                "dtype_numpy": np.dtype(np.float64).str,
                "external": "STREAM:",
            }
        }

    async def observe_indices_written(self, timeout: float = float("inf")) -> AsyncIterator[int]:
        """Observe indices as they're written - yield after each frame is generated."""
        # Generate one image immediately (software-triggered, instant acquisition)
        await self._write_single_frame()

        # Yield the index to signal completion
        yield self._last_index

    async def get_indices_written(self) -> int:
        """Get number of indices written so far."""
        return self._last_index

    async def collect_stream_docs(
        self, name: str, indices_written: int
    ) -> AsyncIterator[tuple[str, StreamResource | StreamDatum]]:
        """Collect stream datum documents from the cache.

        Args:
            name: Name of the detector device
            indices_written: Number of indices written
        """
        # Pop all documents from the cache and yield them
        while self._asset_docs_cache:
            yield self._asset_docs_cache.popleft()

    async def close(self) -> None:
        """Close HDF5 file."""
        if self._h5file:
            self._h5file.close()
            self._h5file = None

    async def _write_single_frame(self) -> None:
        """Generate and write a single beam image (internal method)."""
        if self._counter is None or self._dataset is None or self._stream_datum_factory is None:
            raise RuntimeError("Writer not open, call open() first")

        # Generate beam image from backend (async)
        image = await self._backend.generate_beam()

        # Store image
        current_frame = next(self._counter)
        self._dataset.resize((current_frame + 1, *image.shape))
        self._dataset[current_frame, :, :] = image

        # Create stream datum
        stream_datum_doc = self._stream_datum_factory(
            StreamRange(start=current_frame, stop=current_frame + 1),
        )
        self._asset_docs_cache.append(("stream_datum", stream_datum_doc))

        self._last_index = current_frame + 1


class DetectorDevice(StandardDetector):
    """Detector device that generates beam images.

    Args:
        backend: Simulation backend
        path_provider: Provides directory path for HDF5 files
        name: Device name
    """

    def __init__(
        self,
        backend: SimBackend,
        path_provider: PathProvider,
        name: str = "",
    ):
        self._backend = backend

        # Create controller
        controller = SimDetectorController(backend)

        # Create writer
        writer = SimDetectorWriter(backend, path_provider)

        super().__init__(
            controller=controller,
            writer=writer,
            config_sigs=[],
            name=name,
        )

        # Register with backend
        backend.register_device(
            device_name=name,
            device_type="detector",
            get_state_callback=self._get_state,
        )

    async def _get_state(self) -> dict:
        """Get current detector state for backend."""
        return {}


__all__ = ["DetectorDevice", "SimplePathProvider"]
