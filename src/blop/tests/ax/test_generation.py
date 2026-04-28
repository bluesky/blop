"""Tests for blop.bayesian.kernels module.

Tests the public API of LatentKernel - a Matérn kernel with learned affine
transformation using SO(N) parameterization for orthogonal rotations.
"""


import numpy as np
import pytest
from bluesky.run_engine import RunEngine

from blop.ax.agent import Agent
from blop.ax.dof import DOFConstraint, RangeDOF
from blop.ax.objective import Objective

from ..conftest import MovableSignal, ReadableSignal
from ..test_plans import _collect_optimize_events


class TestFunctionEvaluation:
    def __init__(self, tiled_client: Container):
        self.tiled_client = tiled_client

    def __call__(self, uid: str, suggestions: list[dict]) -> list[dict]:
        run = self.tiled_client[uid]
        outcomes = []
        x1_data = run["primary/x1"].read()
        x2_data = run["primary/x2"].read()

        for suggestion in suggestions:
            # Special key to identify a suggestion
            suggestion_id = suggestion["_id"]
            x1 = suggestion["x1"]
            x2 = suggestion["x2"]

            outcomes.append(
                {
                    "test_function_1": 1 - np.exp(-((x1 - 2 * x2 - 1) ** 2) - 1e-3 * (2 * x1 + x2 - 0.5) ** 2),
                    "_id": suggestion_id,
                    "test_function_2": 1 - np.exp(-1e-3 * (x1 - 2 * x2 - 1) ** 2 - (2 * x1 + x2 - 0.5) ** 2),
                    "_id": suggestion_id,
                }
            )

            # outcomes.append({"test_function": (x1 ** 2 + x2 - 11) ** 2 + (x1 + x2 ** 2 - 7) ** 2, "_id": suggestion_id})

        return outcomes


@pytest.fixture(scope="function")
def RE():
    return RunEngine({})


def test_optimize(RE, mock_acquisition_plan, mock_evaluation_function):
    movable1 = MovableSignal(name="test_movable1")
    movable2 = MovableSignal(name="test_movable2")
    readable = ReadableSignal(name="test_readable")
    dof1 = RangeDOF(actuator=movable1, bounds=(0, 10), parameter_type="float")
    dof2 = RangeDOF(actuator=movable2, bounds=(0, 10), parameter_type="float")
    constraint = DOFConstraint(constraint="x1 + x2 <= 10", x1=dof1, x2=dof2)
    objective = Objective(name="test_objective", minimize=False)
    agent = Agent(
        sensors=[readable],
        dofs=[dof1, dof2],
        objectives=[objective],
        evaluation_function=mock_evaluation_function,
        dof_constraints=[constraint],
        acquisition_plan=mock_acquisition_plan,
        name="test_experiment",
    )

    callback, events = _collect_optimize_events()
    RE.subscribe(callback)
    try:
        RE(agent.optimize(20))
    except RuntimeError:
        ...
    finally:
        RE.unsubscribe(callback)
