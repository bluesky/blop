from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ax import Client
from bluesky.callbacks import CallbackBase
from bluesky_queueserver_api.zmq import REManagerAPI

from blop.ax.agent import Agent, QueueserverAgent
from blop.ax.dof import DOFConstraint, RangeDOF
from blop.ax.objective import Objective, ScalarizedObjective
from blop.ax.optimizer import AxOptimizer
from blop.callbacks.logger import OptimizationLogger
from blop.protocols import AcquisitionPlan, EvaluationFunction

from ..conftest import MovableSignal, ReadableSignal


@pytest.fixture(scope="function")
def mock_evaluation_function():
    return MagicMock(spec=EvaluationFunction)


@pytest.fixture(scope="function")
def mock_acquisition_plan():
    return MagicMock(spec=AcquisitionPlan)


@pytest.fixture(scope="function")
def mock_re_manager_api():
    return MagicMock(spec=REManagerAPI)


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


def test_agent_suggest_fixed_dofs(mock_evaluation_function):
    movable1 = MovableSignal(name="test_movable1")
    movable2 = MovableSignal(name="test_movable2")
    dof1 = RangeDOF(actuator=movable1, bounds=(0, 10), parameter_type="float")
    dof2 = RangeDOF(actuator=movable2, bounds=(0, 10), parameter_type="float")
    objective = Objective(name="test_objective", minimize=False)
    agent = Agent(
        sensors=[],
        dofs=[dof1, dof2],
        objectives=[objective],
        evaluation_function=mock_evaluation_function,
    )
    # Keys must be a DOF object
    with pytest.raises(AttributeError):
        agent.fixed_dofs = {"test_movable1": 3}

    # Valid updates should fix the DOF
    agent.fixed_dofs = {dof2: 4}
    parameterizations = agent.suggest(5)
    for i in range(5):
        assert "test_movable2" in parameterizations[i]
        if i != 0:  # first trial will default to CenterOfSearchSpace and override any fixed parameters
            assert parameterizations[i]["test_movable2"] == 4


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


def test_reconfigure_search_space(mock_evaluation_function):
    movable1 = MovableSignal(name="test_movable1")
    movable2 = MovableSignal(name="test_movable2")
    dof1 = RangeDOF(actuator=movable1, bounds=(0, 10), parameter_type="float")
    dof2 = RangeDOF(actuator=movable2, bounds=(0, 10), parameter_type="float")
    objective = Objective(name="test_objective", minimize=False)
    agent = Agent(
        sensors=[],
        dofs=[dof1, dof2],
        objectives=[objective],
        evaluation_function=mock_evaluation_function,
    )
    # Keys must be DOF objects, not parameter names
    with pytest.raises(AttributeError):
        agent.reconfigure_search_space({"test_movable1": (3, 6)})

    # Valid update should restrict the search space
    agent.reconfigure_search_space({dof1: (3, 6)})
    parameterizations = agent.suggest(10)
    for i in range(10):
        assert 3 <= parameterizations[i]["test_movable1"] <= 6


def test_agent_init_actuator_string_raises(mock_evaluation_function):
    dof1 = RangeDOF(actuator="test_movable1", bounds=(0, 10), parameter_type="float")
    dof2 = RangeDOF(actuator="test_movable2", bounds=(0, 10), parameter_type="float")
    objective = Objective(name="test_objective", minimize=False)

    with pytest.raises(ValueError, match="not strings"):
        Agent(sensors=[], dofs=[dof1, dof2], objectives=[objective], evaluation_function=mock_evaluation_function)


def test_agent_scalarized_objective(mock_evaluation_function):
    dof1 = RangeDOF(name="test1", bounds=(0, 10), parameter_type="float")
    scalarized_objective = ScalarizedObjective(
        "-50 * x + 100 * y",
        minimize=False,
        x="obj1",
        y="obj2",
    )

    # Validate that the call passes the arguments properly
    with patch("blop.ax.optimizer.Client.configure_optimization") as mock_configure_optimization:
        Agent(
            sensors=[],
            dofs=[dof1],
            objectives=scalarized_objective,
            evaluation_function=mock_evaluation_function,
        )

        mock_configure_optimization.assert_called_once_with(
            objective="-50 * obj1 + 100 * obj2",
            outcome_constraints=None,
        )

    # Validate that Ax can actually use this
    Agent(
        sensors=[],
        dofs=[dof1],
        objectives=scalarized_objective,
        evaluation_function=mock_evaluation_function,
    )


def test_queueserver_agent_init(mock_re_manager_api, mock_evaluation_function):
    dof1 = RangeDOF(actuator="test_motor1", bounds=(0, 10), parameter_type="float")
    dof2 = RangeDOF(actuator="test_motor2", bounds=(0, 10), parameter_type="float")
    agent = QueueserverAgent(
        mock_re_manager_api,
        "inproc://test",
        ["det"],
        [dof1, dof2],
        [Objective(name="obj1", minimize=False)],
        mock_evaluation_function,
    )
    assert agent.sensors == ["det"]
    assert agent.actuators == [dof1.actuator, dof2.actuator]
    assert agent.evaluation_function == mock_evaluation_function
    assert agent.acquisition_plan is None
    assert isinstance(agent.ax_client, Client)

    problem = agent.to_optimization_problem()
    assert problem.acquisition_plan is None
    assert problem.actuators == [dof1.parameter_name, dof2.parameter_name]
    assert problem.sensors == ["det"]
    assert problem.evaluation_function == mock_evaluation_function


def test_queueserver_agent_init_actuator_instance(mock_re_manager_api, mock_evaluation_function):
    movable1 = MovableSignal(name="test_movable1")
    dof1 = RangeDOF(actuator=movable1, bounds=(0, 10), parameter_type="float")
    dof2 = RangeDOF(actuator="test_movable2", bounds=(0, 10), parameter_type="float")
    agent = QueueserverAgent(
        mock_re_manager_api,
        "inproc://test",
        ["det"],
        [dof1, dof2],
        [Objective(name="obj1", minimize=False)],
        mock_evaluation_function,
    )

    assert agent.actuators == [movable1.name, dof2.parameter_name]


@patch("blop.ax.agent.QueueserverClient")
@patch("blop.ax.agent.QueueserverOptimizationRunner")
def test_queueserver_agent_run(
    mock_queueserver_runner_cls, mock_queueserver_client_cls, mock_re_manager_api, mock_evaluation_function
):
    dof1 = RangeDOF(actuator="test_motor1", bounds=(0, 10), parameter_type="float")
    dof2 = RangeDOF(actuator="test_motor2", bounds=(0, 10), parameter_type="float")
    agent = QueueserverAgent(
        mock_re_manager_api,
        "inproc://test",
        ["det"],
        [dof1, dof2],
        [Objective(name="obj1", minimize=False)],
        mock_evaluation_function,
    )
    mock_queueserver_client_cls.assert_called_once()
    mock_queueserver_runner_cls.assert_called_once()

    agent.run()
    mock_queueserver_runner_cls.return_value.run.assert_called_once_with(1, 1)


@patch("blop.ax.agent.QueueserverClient")
@patch("blop.ax.agent.QueueserverOptimizationRunner")
def test_queueserver_agent_submit_suggestions(
    mock_queueserver_runner_cls, mock_queueserver_client_cls, mock_re_manager_api, mock_evaluation_function
):
    dof1 = RangeDOF(actuator="test_motor1", bounds=(0, 10), parameter_type="float")
    dof2 = RangeDOF(actuator="test_motor2", bounds=(0, 10), parameter_type="float")
    agent = QueueserverAgent(
        mock_re_manager_api,
        "inproc://test",
        ["det"],
        [dof1, dof2],
        [Objective(name="obj1", minimize=False)],
        mock_evaluation_function,
    )
    mock_queueserver_client_cls.assert_called_once()
    mock_queueserver_runner_cls.assert_called_once()

    suggestions = [{"test_motor1": 5, "test_motor2": 9}]
    agent.submit_suggestions(suggestions)
    mock_queueserver_runner_cls.return_value.submit_suggestions.assert_called_once_with(suggestions)


@pytest.fixture()
def agent(mock_evaluation_function):
    """A minimal Agent for testing callback management."""
    dof = RangeDOF(name="x1", bounds=(0, 10), parameter_type="float")
    objective = Objective(name="obj", minimize=False)
    return Agent(
        sensors=[],
        dofs=[dof],
        objectives=[objective],
        evaluation_function=mock_evaluation_function,
    )


class _StubCallback(CallbackBase):
    """Minimal CallbackBase for test assertions."""


def test_default_callbacks(agent):
    """A new agent should have exactly one OptimizationLogger by default."""
    assert len(agent.callbacks) == 1
    assert isinstance(agent.callbacks[0], OptimizationLogger)


def test_subscribe(agent):
    cb = _StubCallback()
    agent.subscribe(cb)
    assert cb in agent.callbacks
    assert len(agent.callbacks) == 2


def test_subscribe_duplicate_raises(agent):
    cb = _StubCallback()
    agent.subscribe(cb)
    with pytest.raises(ValueError, match="already subscribed"):
        agent.subscribe(cb)


def test_unsubscribe(agent):
    cb = _StubCallback()
    agent.subscribe(cb)
    agent.unsubscribe(cb)
    assert cb not in agent.callbacks


def test_unsubscribe_not_subscribed_raises(agent):
    cb = _StubCallback()
    with pytest.raises(ValueError):
        agent.unsubscribe(cb)


def test_unsubscribe_default_logger(agent):
    """Users should be able to remove the default OptimizationLogger."""
    logger = agent.callbacks[0]
    agent.unsubscribe(logger)
    assert len(agent.callbacks) == 0


def test_callbacks_clear(agent):
    """Clearing the list should disable all callbacks."""
    agent.callbacks.clear()
    assert len(agent.callbacks) == 0


def test_subscribe_after_clear(agent):
    """Subscribing after clearing should work normally."""
    agent.callbacks.clear()
    cb = _StubCallback()
    agent.subscribe(cb)
    assert agent.callbacks == [cb]


def test_from_checkpoint_has_default_callbacks(mock_evaluation_function, tmp_path):
    """An agent loaded from a checkpoint should have the default callbacks."""
    checkpoint_path = tmp_path / "checkpoint.json"
    original = Agent(
        sensors=[ReadableSignal(name="s")],
        dofs=[RangeDOF(name="x1", bounds=(0, 10), parameter_type="float")],
        objectives=[Objective(name="obj", minimize=False)],
        evaluation_function=mock_evaluation_function,
        checkpoint_path=str(checkpoint_path),
    )
    original.ingest([{"x1": 1.0, "obj": 2.0}])
    original.ax_client.configure_generation_strategy()
    original.checkpoint()

    restored = Agent.from_checkpoint(
        str(checkpoint_path),
        sensors=[ReadableSignal(name="s")],
        actuators=[],
        evaluation_function=mock_evaluation_function,
    )
    assert len(restored.callbacks) == 1
    assert isinstance(restored.callbacks[0], OptimizationLogger)
