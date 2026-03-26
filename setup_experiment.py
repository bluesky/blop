import logging
import time
import warnings
from typing import Any

import numpy as np
from bluesky.protocols import HasHints, HasParent, Hints, NamedMovable, Readable, Status
from bluesky.run_engine import RunEngine

from blop import Agent, Objective, RangeDOF
from blop.callbacks import BestEffortOptimizationCallback

logging.getLogger("ax.api.client").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")


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


det = ReadableSignal("det")
x1 = MovableSignal("x1")
x2 = MovableSignal("x2")

dofs = [
    RangeDOF(actuator=x1, bounds=(-5, 5), parameter_type="float"),
    RangeDOF(actuator=x2, bounds=(-5, 5), parameter_type="float"),
]
sensors = [det]
objectives = [
    Objective(name="intensity", minimize=False),
    Objective(name="beam_area", minimize=True),
]


def evaluation_function(uid: str, suggestions: list[dict]) -> list[dict]:
    return [{"intensity": np.random.rand(), "beam_area": np.random.rand(), "_id": s["_id"]} for s in suggestions]


agent = Agent(
    sensors=sensors,
    dofs=dofs,
    objectives=objectives,
    evaluation_function=evaluation_function,
)
RE = RunEngine({})
beoc = BestEffortOptimizationCallback()
RE.subscribe(beoc)
