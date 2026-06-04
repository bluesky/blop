from unittest.mock import MagicMock

import pytest
from bluesky.run_engine import RunEngine

xopt = pytest.importorskip("xopt")

from xopt.generators.random import RandomGenerator

from blop.ax.dof import RangeDOF
from blop.ax.objective import Objective
from blop.tests.conftest import MovableSignal, ReadableSignal
from blop.xopt.agent import XoptAgent


@pytest.fixture(scope="function")
def RE():
    return RunEngine({})


def test_xopt_agent_init_and_suggest():
    movable = MovableSignal(name="x")
    readable = ReadableSignal(name="det")
    dof = RangeDOF(actuator=movable, bounds=(0.0, 1.0), parameter_type="float")
    objective = Objective(name="score", minimize=True)

    evaluation_function = MagicMock(return_value=[{"_id": 0, "score": 0.0}])
    agent = XoptAgent(
        sensors=[readable],
        dofs=[dof],
        objectives=[objective],
        evaluation_function=evaluation_function,
        generator=RandomGenerator,
    )

    suggestions = agent.suggest(1)
    assert len(suggestions) == 1
    assert "_id" in suggestions[0]
    assert "x" in suggestions[0]


def test_xopt_agent_optimize_runs(RE):
    movable = MovableSignal(name="x")
    readable = ReadableSignal(name="det")
    dof = RangeDOF(actuator=movable, bounds=(0.0, 1.0), parameter_type="float")
    objective = Objective(name="score", minimize=True)

    def evaluate(uid, suggestions):
        return [{"_id": suggestion["_id"], "score": float(suggestion["x"])} for suggestion in suggestions]

    agent = XoptAgent(
        sensors=[readable],
        dofs=[dof],
        objectives=[objective],
        evaluation_function=evaluate,
        generator=RandomGenerator,
    )

    RE(agent.optimize(iterations=2, n_points=1))

    assert agent.optimizer.generator.data is not None
    assert len(agent.optimizer.generator.data) == 2
    assert len(agent.get_best_points()) >= 1
