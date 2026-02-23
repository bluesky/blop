from unittest.mock import MagicMock

import numpy as np
import pytest
from ax import Client

from blop.ax.agent import Agent
from blop.ax.dof import DOFConstraint, RangeDOF
from blop.ax.objective import Objective
from blop.ax.optimizer import AxOptimizer
from blop.protocols import AcquisitionPlan, EvaluationFunction

from ..conftest import MovableSignal, ReadableSignal


@pytest.fixture(scope="function")
def mock_evaluation_function():
    return MagicMock(spec=EvaluationFunction)


@pytest.fixture(scope="function")
def mock_acquisition_plan():
    return MagicMock(spec=AcquisitionPlan)


def test_agent_init(mock_evaluation_function, mock_acquisition_plan):
    """Test that the agent can be initialized."""
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
    assert agent.sensors == [readable]
    assert agent.actuators == [dof1.actuator, dof2.actuator]
    assert agent.evaluation_function == mock_evaluation_function
    assert agent.acquisition_plan == mock_acquisition_plan
    assert isinstance(agent.ax_client, Client)


def test_agent_checkpoint(mock_evaluation_function, mock_acquisition_plan, tmp_path):
    checkpoint_path = tmp_path / "checkpoint.json"
    readable = ReadableSignal(name="test_readable")
    agent = Agent(
        sensors=[ReadableSignal(name="test_readable")],
        dofs=[RangeDOF(name="x1", bounds=(0, 10), parameter_type="float")],
        objectives=[Objective(name="test_objective", minimize=False)],
        evaluation_function=mock_evaluation_function,
        acquisition_plan=mock_acquisition_plan,
        checkpoint_path=str(checkpoint_path),
    )

    assert agent.checkpoint_path == str(checkpoint_path)
    assert not checkpoint_path.exists()
    agent.ingest([{"x1": 0.1, "test_objective": 0.2}])
    agent.ax_client.configure_generation_strategy()
    agent.checkpoint()
    assert checkpoint_path.exists()

    agent = Agent.from_checkpoint(
        str(checkpoint_path),
        sensors=[readable],
        actuators=[],
        evaluation_function=mock_evaluation_function,
        acquisition_plan=mock_acquisition_plan,
    )
    assert len(agent.ax_client.summarize()) == 1


def test_agent_to_optimization_problem(mock_evaluation_function):
    """Test that the agent can be converted to an optimization problem."""
    movable1 = MovableSignal(name="test_movable1")
    movable2 = MovableSignal(name="test_movable2")
    dof1 = RangeDOF(actuator=movable1, bounds=(0, 10), parameter_type="float")
    dof2 = RangeDOF(actuator=movable2, bounds=(0, 10), parameter_type="float")
    constraint = DOFConstraint(constraint="x1 + x2 <= 10", x1=dof1, x2=dof2)
    objective = Objective(name="test_objective", minimize=False)
    agent = Agent(
        sensors=[],
        dofs=[dof1, dof2],
        objectives=[objective],
        evaluation_function=mock_evaluation_function,
        dof_constraints=[constraint],
    )
    optimization_problem = agent.to_optimization_problem()
    assert optimization_problem.evaluation_function == mock_evaluation_function
    assert optimization_problem.actuators == [movable1, movable2]
    assert optimization_problem.sensors == []
    assert isinstance(optimization_problem.optimizer, AxOptimizer)
    assert optimization_problem.acquisition_plan is None


def test_agent_suggest(mock_evaluation_function):
    movable1 = MovableSignal(name="test_movable1")
    movable2 = MovableSignal(name="test_movable2")
    dof1 = RangeDOF(actuator=movable1, bounds=(0, 10), parameter_type="float")
    dof2 = RangeDOF(actuator=movable2, bounds=(0, 10), parameter_type="float")
    objective = Objective(name="test_objective", minimize=False)
    agent = Agent(sensors=[], dofs=[dof1, dof2], objectives=[objective], evaluation_function=mock_evaluation_function)

    parameterizations = agent.suggest(1)
    assert len(parameterizations) == 1
    assert parameterizations[0]["_id"] == 0
    assert "test_movable1" in parameterizations[0]
    assert "test_movable2" in parameterizations[0]
    assert isinstance(parameterizations[0]["test_movable1"], (int, float))
    assert isinstance(parameterizations[0]["test_movable2"], (int, float))


def test_agent_suggest_multiple(mock_evaluation_function):
    movable1 = MovableSignal(name="test_movable1")
    movable2 = MovableSignal(name="test_movable2")
    dof1 = RangeDOF(actuator=movable1, bounds=(0, 10), parameter_type="float")
    dof2 = RangeDOF(actuator=movable2, bounds=(0, 10), parameter_type="float")
    objective = Objective(name="test_objective", minimize=False)
    agent = Agent(sensors=[], dofs=[dof1, dof2], objectives=[objective], evaluation_function=mock_evaluation_function)

    parameterizations = agent.suggest(5)
    assert len(parameterizations) == 5
    for i in range(5):
        assert parameterizations[i]["_id"] == i
        assert "test_movable1" in parameterizations[i]
        assert "test_movable2" in parameterizations[i]
        assert isinstance(parameterizations[i]["test_movable1"], (int, float))
        assert isinstance(parameterizations[i]["test_movable2"], (int, float))


def test_agent_ingest(mock_evaluation_function):
    movable1 = MovableSignal(name="test_movable1")
    movable2 = MovableSignal(name="test_movable2")
    dof1 = RangeDOF(actuator=movable1, bounds=(0, 10), parameter_type="float")
    dof2 = RangeDOF(actuator=movable2, bounds=(0, 10), parameter_type="float")
    objective = Objective(name="test_objective", minimize=False)
    agent = Agent(sensors=[], dofs=[dof1, dof2], objectives=[objective], evaluation_function=mock_evaluation_function)

    agent.ingest([{"test_movable1": 0.1, "test_movable2": 0.2, "test_objective": 0.3}])

    agent.ax_client.configure_generation_strategy()
    summary_df = agent.ax_client.summarize()
    assert len(summary_df) == 1
    assert np.all(summary_df["test_movable1"].values == [0.1])
    assert np.all(summary_df["test_movable2"].values == [0.2])
    assert np.all(summary_df["test_objective"].values == [0.3])


def test_agent_ingest_multiple(mock_evaluation_function):
    movable1 = MovableSignal(name="test_movable1")
    movable2 = MovableSignal(name="test_movable2")
    dof1 = RangeDOF(actuator=movable1, bounds=(0, 10), parameter_type="float")
    dof2 = RangeDOF(actuator=movable2, bounds=(0, 10), parameter_type="float")
    objective = Objective(name="test_objective", minimize=False)
    agent = Agent(sensors=[], dofs=[dof1, dof2], objectives=[objective], evaluation_function=mock_evaluation_function)

    agent.ingest(
        [
            {"test_movable1": 0.1, "test_movable2": 0.2, "test_objective": 0.3},
            {"test_movable1": 1.1, "test_movable2": 1.2, "test_objective": 1.3},
        ]
    )
    agent.ax_client.configure_generation_strategy()
    summary_df = agent.ax_client.summarize()
    assert len(summary_df) == 2
    assert np.all(summary_df["test_movable1"].values == [0.1, 1.1])
    assert np.all(summary_df["test_movable2"].values == [0.2, 1.2])
    assert np.all(summary_df["test_objective"].values == [0.3, 1.3])


def test_ingest_baseline(mock_evaluation_function):
    movable1 = MovableSignal(name="test_movable1")
    movable2 = MovableSignal(name="test_movable2")
    dof1 = RangeDOF(actuator=movable1, bounds=(0, 10), parameter_type="float")
    dof2 = RangeDOF(actuator=movable2, bounds=(0, 10), parameter_type="float")
    objective = Objective(name="test_objective", minimize=False)
    agent = Agent(sensors=[], dofs=[dof1, dof2], objectives=[objective], evaluation_function=mock_evaluation_function)

    agent.ingest([{"test_movable1": 0.1, "test_movable2": 0.2, "test_objective": 0.3, "_id": "baseline"}])

    agent.ax_client.configure_generation_strategy()
    summary_df = agent.ax_client.summarize()
    assert len(summary_df) == 1
    assert summary_df["arm_name"].values[0] == "baseline"
