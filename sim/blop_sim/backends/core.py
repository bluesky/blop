from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any

import numpy as np


class SimBackend(ABC):
    """Base class for simulation backends.

    Uses singleton pattern - only one instance per backend type exists.
    All device callbacks are expected to be async.
    """

    _instances: dict[type, "SimBackend"] = {}

    def __new__(cls):
        """Singleton pattern: return existing instance or create new."""
        if cls not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[cls] = instance
            instance._initialized = False
        return cls._instances[cls]

    def __init__(self):
        """Initialize backend state (only runs once due to singleton)."""
        if self._initialized:
            return

        self._device_states: dict[str, dict[str, Any]] = {}
        self._image_shape = (300, 400)
        self._initialized = True

    def register_device(self, device_name: str, device_type: str, get_state_callback: Callable[[], Awaitable[dict]]):
        """Register a device with the backend.

        Args:
            device_name: Unique name for the device
            device_type: Type of device ("kb_mirror_simple", "kb_mirror_xrt", "slit", "detector")
            get_state_callback: Async callable that returns current device state as dict

        Example::

            async def _get_state(self) -> dict:
                return {
                    "radius": await self.radius.get_value(),
                    "position": await self.position.get_value(),
                }
        """
        self._device_states[device_name] = {
            "type": device_type,
            "get_state": get_state_callback,
        }

    async def _get_device_state(self, device_name: str) -> dict:
        """Get device state asynchronously.

        Args:
            device_name: Name of the device

        Returns:
            Device state dictionary
        """
        device = self._device_states[device_name]
        callback = device["get_state"]
        return await callback()

    @abstractmethod
    async def generate_beam(self) -> np.ndarray:
        """Generate beam image based on current device states.

        Returns:
            2D numpy array with shape self._image_shape
        """
        pass

    def get_image_shape(self) -> tuple[int, int]:
        """Return the image shape."""
        return self._image_shape
