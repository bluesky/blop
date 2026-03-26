"""Interactive test for OptimizationPlotter with a real Blop Agent.

Run in IPython (docs pixi environment) with:

    pixi run -e docs ipython
    %matplotlib tk
    %run -i test_plotter_interactive.py

Then:
    RE(agent.optimize(10))              # 10 iterations, watch plots update
    RE(agent.optimize(10, n_points=3))  # 10 more, batches of 3, data persists
"""

import time
from typing import Any

import numpy as np
from bluesky.protocols import HasHints, HasParent, Hints, NamedMovable, Readable, Status
from bluesky.run_engine import RunEngine

from blop.ax import Agent, Objective, RangeDOF
from blop.callbacks.plotter import OptimizationPlotter


# ── Simulated devices ───────────────────────────────────────────────


class AlwaysSuccessfulStatus(Status):
    def add_callback(self, callback):
        callback(self)

    def exception(self, timeout=0.0):
        return None

    @property
    def done(self):
        return True

    @property
    def success(self):
        return True


class ReadableSignal(Readable, HasHints, HasParent):
    def __init__(self, name):
        self._name = name
        self._value = 0.0

    @property
    def name(self):
        return self._name

    @property
    def hints(self) -> Hints:
        return {"fields": [self._name], "dimensions": [], "gridding": "rectilinear"}

    @property
    def parent(self) -> Any | None:
        return None

    def read(self):
        return {self._name: {"value": self._value, "timestamp": time.time()}}

    def describe(self):
        return {self._name: {"source": self._name, "dtype": "number", "shape": []}}


class MovableSignal(ReadableSignal, NamedMovable):
    def __init__(self, name, initial_value=0.0):
        super().__init__(name)
        self._value = initial_value

    def set(self, value) -> Status:
        self._value = value
        return AlwaysSuccessfulStatus()


# ── Evaluation function (Himmelblau + noise) ────────────────────────


def himmelblau_eval(uid, suggestions):
    outcomes = []
    for s in suggestions:
        x1 = s["x1"]
        x2 = s["x2"]
        val = (x1**2 + x2 - 11) ** 2 + (x1 + x2**2 - 7) ** 2
        val += np.random.normal(0, 0.5)  # small noise
        outcomes.append({"himmelblau": float(val), "_id": s["_id"]})
    return outcomes


# ── Wiring ──────────────────────────────────────────────────────────

RE = RunEngine({})

x1 = MovableSignal("x1", initial_value=0.1)
x2 = MovableSignal("x2", initial_value=0.2)

agent = Agent(
    sensors=[],
    dofs=[
        RangeDOF(actuator=x1, bounds=(-5, 5), parameter_type="float"),
        RangeDOF(actuator=x2, bounds=(-5, 5), parameter_type="float"),
    ],
    objectives=[Objective(name="himmelblau", minimize=True)],
    evaluation_function=himmelblau_eval,
    name="plotter-test",
)

plotter = OptimizationPlotter()
agent.subscribe(plotter)

print("Ready. Try:")
print()
print("  RE(agent.optimize(10))              # 10 iterations")
print("  RE(agent.optimize(10, n_points=3))  # 10 more, batches of 3")
