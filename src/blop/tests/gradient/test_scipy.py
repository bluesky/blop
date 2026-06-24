import time
from unittest.mock import MagicMock

import pytest

from blop.ax import Objective, RangeDOF
from blop.gradient import SCP, Scipy, ScipyCFG, ScipyOptimizer
from blop.protocols import ID_KEY, AcquisitionPlan, EvaluationFunction

from ..conftest import MovableSignal, ReadableSignal


@pytest.fixture(scope="function")
def mock_evaluation_function():
    return MagicMock(spec=EvaluationFunction)


@pytest.fixture(scope="function")
def mock_acquisition_plan():
    return MagicMock(spec=AcquisitionPlan)

# agent._optimizer.close() is called so the standard timeout doesnt make the testing take forever


@pytest.fixture(scope="function")
def agent_prep(mock_evaluation_function, mock_acquisition_plan):
    movable1 = MovableSignal(name="test_movable1")
    movable2 = MovableSignal(name="test_movable2")
    readable = ReadableSignal(name="test_readable")
    dof1 = RangeDOF(actuator=movable1, bounds=(0, 10), parameter_type="float")
    dof2 = RangeDOF(actuator=movable2, bounds=(0, 10), parameter_type="float")
    objective = Objective(name="test_objective", minimize=False)
    config = ScipyCFG(
        dofs=[dof1, dof2],
        objective=objective,
        threads=4
    )
    agent = Scipy(
        sensors=[readable],
        config=config,
        evaluation_function=mock_evaluation_function,
        acquisition_plan=mock_acquisition_plan,
        name="test_experiment",
        timeout=5
    )
    time.sleep(.1)
    return agent


def test_general_init(mock_evaluation_function, mock_acquisition_plan):
    """Test that the simple Scipy can be initialized."""
    movable1 = MovableSignal(name="test_movable1")
    movable2 = MovableSignal(name="test_movable2")
    readable = ReadableSignal(name="test_readable")
    dof1 = RangeDOF(actuator=movable1, bounds=(0, 10), parameter_type="float")
    dof2 = RangeDOF(actuator=movable2, bounds=(0, 10), parameter_type="float")
    objective = Objective(name="test_objective", minimize=False)
    config = ScipyCFG(
        dofs=[dof1, dof2],
        objective=objective,
    )
    agent = Scipy(
        sensors=[readable],
        config=config,
        evaluation_function=mock_evaluation_function,
        acquisition_plan=mock_acquisition_plan,
        name="test_experiment",
        timeout=5
    )
    assert agent.sensors == [readable]
    assert agent.actuators == [dof1.actuator, dof2.actuator]
    assert agent.evaluation_function == mock_evaluation_function
    assert agent.acquisition_plan == mock_acquisition_plan
    agent._optimizer.close()


def test_agent_init(mock_evaluation_function, mock_acquisition_plan):
    """Test that the agent can be initialized."""
    movable1 = MovableSignal(name="test_movable1")
    movable2 = MovableSignal(name="test_movable2")
    readable = ReadableSignal(name="test_readable")
    dof1 = RangeDOF(actuator=movable1, bounds=(0, 10), parameter_type="float")
    dof2 = RangeDOF(actuator=movable2, bounds=(0, 10), parameter_type="float")
    objective = Objective(name="test_objective", minimize=False)
    agent = Scipy.Agent(
        sensors=[readable],
        dofs=[dof1, dof2],
        objectives=[objective],
        evaluation_function=mock_evaluation_function,
        acquisition_plan=mock_acquisition_plan,
        name="test_experiment",
        timeout=5
    )
    assert agent.sensors == [readable]
    assert agent.actuators == [dof1.actuator, dof2.actuator]
    assert agent.evaluation_function == mock_evaluation_function
    assert agent.acquisition_plan == mock_acquisition_plan
    agent._optimizer.close()


def test_agent_to_optimization_problem(mock_evaluation_function):
    """Test that the agent can be converted to an optimization problem."""
    movable1 = MovableSignal(name="test_movable1")
    movable2 = MovableSignal(name="test_movable2")
    dof1 = RangeDOF(actuator=movable1, bounds=(0, 10), parameter_type="float")
    dof2 = RangeDOF(actuator=movable2, bounds=(0, 10), parameter_type="float")
    objective = Objective(name="test_objective", minimize=False)
    agent = Scipy.Agent(
        sensors=[],
        dofs=[dof1, dof2],
        objectives=[objective],
        evaluation_function=mock_evaluation_function,
        timeout=5
    )
    optimization_problem = agent.to_optimization_problem()
    assert optimization_problem.evaluation_function == mock_evaluation_function
    assert optimization_problem.actuators == [movable1, movable2]
    assert optimization_problem.sensors == []
    assert isinstance(optimization_problem.optimizer, ScipyOptimizer)
    assert optimization_problem.acquisition_plan is None
    agent._optimizer.close()


def test_agent_suggest(agent_prep):
    parameterizations = agent_prep.suggest(1)
    assert len(parameterizations) == 1
    assert parameterizations[0]["_id"] == 0
    assert "test_movable1" in parameterizations[0]
    assert "test_movable2" in parameterizations[0]
    assert isinstance(parameterizations[0]["test_movable1"], (int, float))
    assert isinstance(parameterizations[0]["test_movable2"], (int, float))
    agent_prep._optimizer.close()


def test_agent_ingest(agent_prep):
    agent_prep.suggest()
    agent_prep.ingest([{"test_movable1": 0.1, "test_movable2": 0.2, "test_objective": 0.3, ID_KEY: 0}])
    agent_prep._optimizer.close()


def test_agent_multithread(agent_prep):
    agent_prep.suggest(1)
    agent_prep.ingest([{"test_movable1": 0.1, "test_movable2": 0.2, "test_objective": 0.3, ID_KEY: 0}])
    time.sleep(.1)
    params = agent_prep.suggest(4)
    print(agent_prep._optimizer._active)
    assert len(params) > 1
    agent_prep._optimizer.close()
