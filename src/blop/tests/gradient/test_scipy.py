from unittest.mock import MagicMock, patch

import pytest

import blop.gradient.Scipy as scp
from blop.ax.dof import RangeDOF
from blop.ax.objective import Objective
from blop.protocols import AcquisitionPlan, EvaluationFunction

from ..conftest import MovableSignal, ReadableSignal


@pytest.fixture(scope="function")
def mock_evaluation_function():
    return MagicMock(spec=EvaluationFunction)


@pytest.fixture(scope="function")
def mock_acquisition_plan():
    return MagicMock(spec=AcquisitionPlan)


@pytest.fixture(scope="function")
def agent_prep():
    movable1 = MovableSignal(name="test_movable1")
    movable2 = MovableSignal(name="test_movable2")
    readable = ReadableSignal(name="test_readable")
    dof1 = RangeDOF(actuator=movable1, bounds=(0, 10), parameter_type="float")
    dof2 = RangeDOF(actuator=movable2, bounds=(0, 10), parameter_type="float")
    objective = Objective(name="test_objective", minimize=False)
    config = scp.ScipyCFG(
        dofs=[dof1, dof2],
        objective=objective,
    )
    agent = scp.Scipy(
        sensors=[readable],
        config=config,
        evaluation_function=mock_evaluation_function,
        acquisition_plan=mock_acquisition_plan,
        name="test_experiment",
    )
    return agent


def test_general_init(mock_evaluation_function, mock_acquisition_plan):
    """Test that the simple Scipy can be initialized."""
    movable1 = MovableSignal(name="test_movable1")
    movable2 = MovableSignal(name="test_movable2")
    readable = ReadableSignal(name="test_readable")
    dof1 = RangeDOF(actuator=movable1, bounds=(0, 10), parameter_type="float")
    dof2 = RangeDOF(actuator=movable2, bounds=(0, 10), parameter_type="float")
    objective = Objective(name="test_objective", minimize=False)
    config = scp.ScipyCFG(
        dofs=[dof1, dof2],
        objective=objective,
    )
    agent = scp.Scipy(
        sensors=[readable],
        config=config,
        evaluation_function=mock_evaluation_function,
        acquisition_plan=mock_acquisition_plan,
        name="test_experiment",
    )
    assert agent.sensors == [readable]
    assert agent.actuators == [dof1.actuator, dof2.actuator]
    assert agent.evaluation_function == mock_evaluation_function
    assert agent.acquisition_plan == mock_acquisition_plan


def test_agent_init(mock_evaluation_function, mock_acquisition_plan):
    """Test that the agent can be initialized."""
    movable1 = MovableSignal(name="test_movable1")
    movable2 = MovableSignal(name="test_movable2")
    readable = ReadableSignal(name="test_readable")
    dof1 = RangeDOF(actuator=movable1, bounds=(0, 10), parameter_type="float")
    dof2 = RangeDOF(actuator=movable2, bounds=(0, 10), parameter_type="float")
    objective = Objective(name="test_objective", minimize=False)
    agent = scp.Scipy.Agent(
        sensors=[readable],
        dofs=[dof1, dof2],
        objectives=[objective],
        evaluation_function=mock_evaluation_function,
        acquisition_plan=mock_acquisition_plan,
        name="test_experiment",
    )
    assert agent.sensors == [readable]
    assert agent.actuators == [dof1.actuator, dof2.actuator]
    assert agent.evaluation_function == mock_evaluation_function
    assert agent.acquisition_plan == mock_acquisition_plan


def test_agent_to_optimization_problem(mock_evaluation_function):
    """Test that the agent can be converted to an optimization problem."""
    movable1 = MovableSignal(name="test_movable1")
    movable2 = MovableSignal(name="test_movable2")
    dof1 = RangeDOF(actuator=movable1, bounds=(0, 10), parameter_type="float")
    dof2 = RangeDOF(actuator=movable2, bounds=(0, 10), parameter_type="float")
    objective = Objective(name="test_objective", minimize=False)
    agent = scp.Scipy.Agent(
        sensors=[],
        dofs=[dof1, dof2],
        objectives=[objective],
        evaluation_function=mock_evaluation_function,
    )
    optimization_problem = agent.to_optimization_problem()
    assert optimization_problem.evaluation_function == mock_evaluation_function
    assert optimization_problem.actuators == [movable1, movable2]
    assert optimization_problem.sensors == []
    assert isinstance(optimization_problem.optimizer, scp.ScipyOptimizer)
    assert optimization_problem.acquisition_plan is None


def test_agent_suggest(agent_prep):
    parameterizations = agent_prep.suggest(1)
    assert len(parameterizations) == 1
    assert parameterizations[0]["_id"] == 0
    assert "test_movable1" in parameterizations[0]
    assert "test_movable2" in parameterizations[0]
    assert isinstance(parameterizations[0]["test_movable1"], (int, float))
    assert isinstance(parameterizations[0]["test_movable2"], (int, float))


def test_agent_ingest(mock_evaluation_function):
    movable1 = MovableSignal(name="test_movable1")
    movable2 = MovableSignal(name="test_movable2")
    dof1 = RangeDOF(actuator=movable1, bounds=(0, 10), parameter_type="float")
    dof2 = RangeDOF(actuator=movable2, bounds=(0, 10), parameter_type="float")
    objective = Objective(name="test_objective", minimize=False)
    agent = scp.Scipy.Agent(sensors=[], dofs=[dof1, dof2], objectives=[objective], evaluation_function=mock_evaluation_function)

    agent.ingest([{"test_movable1": 0.1, "test_movable2": 0.2, "test_objective": 0.3}])
