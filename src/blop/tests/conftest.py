import threading
import time
from concurrent.futures import Future
from contextlib import contextmanager
from typing import Any

from bluesky.protocols import HasHints, HasParent, Hints, NamedMovable, Readable, Status

from blop.protocols import Checkpointable, Optimizer


class CheckpointableOptimizer(Optimizer, Checkpointable): ...


class AlwaysSuccessfulStatus(Status):
    def add_callback(self, callback) -> None:
        callback(self)

    def exception(self, timeout=0.0):
        return None

    @property
    def done(self) -> bool:
        return True

    @property
    def success(self) -> bool:
        return True


class ReadableSignal(Readable, HasHints, HasParent):
    def __init__(self, name: str) -> None:
        self._name = name
        self._value = 0.0

    @property
    def name(self) -> str:
        return self._name

    @property
    def hints(self) -> Hints:
        return {
            "fields": [self._name],
            "dimensions": [],
            "gridding": "rectilinear",
        }

    @property
    def parent(self) -> Any | None:
        return None

    def read(self):
        return {self._name: {"value": self._value, "timestamp": time.time()}}

    def describe(self):
        return {self._name: {"source": self._name, "dtype": "number", "shape": []}}


class MovableSignal(ReadableSignal, NamedMovable):
    def __init__(self, name: str, initial_value: float = 0.0) -> None:
        super().__init__(name)
        self._value: float = initial_value

    def set(self, value: float) -> Status:
        self._value = value
        return AlwaysSuccessfulStatus()


@contextmanager
def background_call(function, *args, **kwargs):
    future = Future()

    def invoke():
        try:
            result = function(*args, **kwargs)
        except BaseException as error:
            future.set_exception(error)
        else:
            future.set_result(result)

    thread = threading.Thread(target=invoke, daemon=True)
    thread.start()
    try:
        yield future
    finally:
        thread.join(timeout=5)
        assert not thread.is_alive(), "Background public call did not finish"
        future.exception(timeout=5)
