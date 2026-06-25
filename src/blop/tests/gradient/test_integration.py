import time
from unittest.mock import MagicMock

import pytest
from bluesky import RunEngine

from blop.ax import Objective, RangeDOF
from blop.gradient import Scipy, ScipyCFG, ScipyOptimizer
from blop.protocols import ID_KEY, AcquisitionPlan, EvaluationFunction

from ..conftest import MovableSignal, ReadableSignal


def test_integrated_iteration():
    movable = MovableSignal(name="test_movable")
    readable = ReadableSignal(name="test_readable")
    dof = RangeDOF(actuator=movable, bounds=(0, 1E-4), parameter_type="float")
    objective = Objective(name="test_objective", minimize=False)
    config = ScipyCFG(dofs=[dof], objective=objective)

    class deflating_evaluation(EvaluationFunction):
        def __init__(self):
            self.counter = 0
            super().__init__()
        def __call__(self, uid, suggestions):
            self.counter += 1
            return [s | {objective.name: 2**(-.5 * self.counter)} for s in suggestions]

    agent = Scipy(
        sensors=[readable],
        config=config,
        evaluation_function=deflating_evaluation(),
        timeout=5,
    )
    agent._optimizer.force_resiliance = True
    RE = RunEngine({})
    RE(agent.optimize(20))
    time.sleep(.1)
    assert agent._optimizer.final is not None
    assert agent._optimizer.intermediate is not None
    assert not agent._optimizer._active
    RE(agent.optimize(20))
    assert agent.get_best_points() is not None
