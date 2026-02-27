"""Detector device for beam simulation - images only, NO statistics."""

import asyncio
import itertools
from collections import deque
from collections.abc import AsyncGenerator
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py  # type: ignore[import-untyped]
import numpy as np
from event_model import DataKey, StreamRange, compose_stream_resource  # type: ignore[import-untyped]
from ophyd_async.core import (
    DetectorController,
    DetectorWriter,
    PathProvider,
    StandardDetector,
    soft_signal_rw,
)
from ophyd_async.core import StandardReadableFormat as Format

from ..backends import SimBackend


class SimplePathProvider(PathProvider):
    """Simple path provider that just returns a fixed directory path."""
    
    def __init__(self, directory_path: Path | str):
        self.directory_path = Path(directory_path)
    
    def __call__(self, device_name: str | None = None) -> Path:
        return self.directory_path


class SimDetectorController(DetectorController):
    """Controller for simulated detector - generates images only."""
    
    def __init__(self, backend: SimBackend):
        self._backend = backend
        self._arm_status: asyncio.Event | None = None
    
    def get_deadtime(self, exposure: float | None) -> float:
        """Detector has no deadtime (instant acquisition)."""
        return 0.0
    
    async def prepare(self, trigger_info: Any) -> None:
        """Prepare for acquisition with trigger info."""
        # Software triggered detector, no preparation needed
        pass
    
    async def arm(self) -> None:
        """Prepare for acquisition."""
        self._arm_status = asyncio.Event()
    
    async def wait_for_idle(self):
        """Wait for acquisition to complete."""
        if self._arm_status:
            await self._arm_status.wait()
    
    async def disarm(self):
        """Clean up after acquisition."""
        self._arm_status = None


class SimDetectorWriter(DetectorWriter):
    """Writer for detector with Tiled streaming."""
    
    def __init__(
        self,
        backend: SimBackend,
        controller: SimDetectorController,
        path_provider: PathProvider,
        name_provider: Any,
        noise_signal: Any,
    ):
        self._backend = backend
        self._controller = controller
        self.path_provider = path_provider
        self._name_provider = name_provider
        self._noise_signal = noise_signal
        self._asset_docs_cache: deque[tuple[str, dict[str, Any]]] = deque()
        self._h5file: h5py.File | None = None
        self._dataset: h5py.Dataset | None = None
        self._counter: itertools.count[int] | None = None
        self._stream_datum_factory: Any | None = None
        self._last_index = 0
    
    async def open(self, multiplier: int = 1) -> dict[str, DataKey]:
        """Open HDF5 file and setup stream resources."""
        # Generate file path
        date = datetime.now()
        assets_dir = date.strftime("%Y/%m/%d")
        filename = f"{self._name_provider()}.h5"
        
        # Create directory structure
        directory_path = self.path_provider.directory_path
        full_path = directory_path / assets_dir / filename
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
    
    async def observe_indices_written(
        self, timeout: float = float("inf")
    ) -> AsyncGenerator[int, None]:
        """Observe indices as they're written - yield after each frame is generated."""
        # Wait for controller to be armed and then generate one frame
        while self._controller._arm_status is None:
            await asyncio.sleep(0.01)
        
        # Generate one image (software-triggered, one per trigger call)
        await self._write_single_frame()
        yield self._last_index
        
        # Signal that we're done
        if self._controller._arm_status:
            self._controller._arm_status.set()
    
    async def get_indices_written(self) -> int:
        """Get number of indices written so far."""
        return self._last_index
    
    async def collect_stream_docs(
        self, indices_written: int
    ) -> AsyncGenerator[tuple[str, dict[str, Any]], None]:
        """Collect stream datum documents."""
        # Yield all cached asset docs
        for doc in list(self._asset_docs_cache):
            yield doc
        self._asset_docs_cache.clear()
    
    async def close(self) -> None:
        """Close HDF5 file."""
        if self._h5file:
            self._h5file.close()
            self._h5file = None
    
    async def _write_single_frame(self) -> None:
        """Generate and write a single beam image (internal method)."""
        if self._counter is None or self._dataset is None or self._stream_datum_factory is None:
            raise RuntimeError("Writer not open, call open() first")
        
        # Get noise setting (simple attribute access)
        noise = self._noise_signal.noise
        
        # Generate beam image from backend
        image = self._backend.generate_beam(noise=noise)
        
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
    
    This detector ONLY stores images - it does NOT compute statistics.
    Statistics (sum, centroid, width, etc.) should be computed in evaluation
    functions, which is more realistic.
    
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
        
        # Create noise signal (simple attribute, not a readable)
        self.noise = True
        
        # Create controller
        controller = SimDetectorController(backend)
        
        # Simple name provider
        _name_counter = itertools.count()
        
        def name_provider():
            return f"det_{next(_name_counter):06d}"
        
        # Create writer
        writer = SimDetectorWriter(backend, controller, path_provider, name_provider, self)
        
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
    
    def _get_state(self) -> dict:
        """Get current detector state for backend."""
        return {
            "noise": self.noise,
        }


__all__ = ["DetectorDevice", "SimplePathProvider"]
