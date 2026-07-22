"""Utilities for Bluesky callbacks."""

import math

from bluesky.callbacks import CallbackBase
from event_model import Event, EventDescriptor, RunStart, RunStop


class RunningStats:
    """Accumulates running statistics using Welford's online algorithm.

    Tracks count, min, max, mean, and standard deviation without
    storing individual observations. Numerically stable for variance
    computation.
    """

    __slots__ = ("count", "min", "max", "_mean", "_m2")

    def __init__(self) -> None:
        self.count: int = 0
        self.min: float = math.inf
        self.max: float = -math.inf
        self._mean: float = 0.0
        self._m2: float = 0.0

    def update(self, value: float) -> None:
        """Incorporate a new observation."""
        if math.isnan(value) or math.isinf(value):
            return
        self.count += 1
        self.min = min(self.min, value)
        self.max = max(self.max, value)
        delta = value - self._mean
        self._mean += delta / self.count
        delta2 = value - self._mean
        self._m2 += delta * delta2

    @property
    def mean(self) -> float:
        """Average of the observed values so far."""
        return self._mean if self.count > 0 else math.nan

    @property
    def std(self) -> float:
        """Standard deviation of the observed values so far."""
        if self.count < 2:
            return math.nan
        return math.sqrt(self._m2 / (self.count - 1))


class _PrimaryStreamFilter(CallbackBase):
    """Forwards only ``'primary'`` stream documents to a wrapped callback.

    Descriptor documents whose stream name (``doc["name"]``) is not
    ``"primary"`` are silently dropped. Event documents are forwarded only
    when their corresponding descriptor was accepted. All other document
    types (``start``, ``stop``) are passed through unconditionally, as
    they are run-level rather than stream-level.

    Parameters
    ----------
    callback : CallbackBase
        The wrapped callback to receive filtered documents.
    """

    def __init__(self, callback: CallbackBase) -> None:
        super().__init__()
        self._callback = callback
        self._primary_descriptor_uids: set[str] = set()

    def start(self, doc: RunStart) -> RunStart | None:
        return self._callback.start(doc)

    def descriptor(self, doc: EventDescriptor) -> EventDescriptor | None:
        if doc.get("name") == "primary":
            self._primary_descriptor_uids.add(doc["uid"])
            return self._callback.descriptor(doc)

    def event(self, doc: Event) -> Event:
        if doc.get("descriptor") in self._primary_descriptor_uids:
            return self._callback.event(doc)
        # TODO: Fix typing of `CallbackBase.event` in event-model
        return None  # type: ignore

    def stop(self, doc: RunStop) -> RunStop | None:
        return self._callback.stop(doc)
