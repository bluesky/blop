from unittest.mock import MagicMock, patch

import pytest
from ax import Client
from bluesky.callbacks.zmq import RemoteDispatcher
from bluesky_queueserver_api.zmq import REManagerAPI

from blop.ax.dof import RangeDOF
from blop.ax.objective import Objective
from blop.ax.queueserver_agent import QueueserverAgent
from blop.protocols import AcquisitionPlan, EvaluationFunction

from ..conftest import MovableSignal


@pytest.fixture(scope="function")
def mock_evaluation_function():
    return MagicMock(spec=EvaluationFunction)


@pytest.fixture(scope="function")
def mock_acquisition_plan():
    return MagicMock(spec=AcquisitionPlan)


@pytest.fixture(scope="function")
def mock_document_dispatcher():
    return MagicMock(spec=RemoteDispatcher)


@pytest.fixture(scope="function")
def mock_re_manager_api():
    return MagicMock(spec=REManagerAPI)


def test_queueserver_agent_init(mock_re_manager_api, mock_document_dispatcher, mock_evaluation_function):
    dof1 = RangeDOF(actuator="test_motor1", bounds=(0, 10), parameter_type="float")
    dof2 = RangeDOF(actuator="test_motor2", bounds=(0, 10), parameter_type="float")
    agent = QueueserverAgent(
        mock_re_manager_api,
        mock_document_dispatcher,
        ["det"],
        [dof1, dof2],
        [Objective(name="obj1", minimize=False)],
        mock_evaluation_function,
    )
    assert agent.sensors == ["det"]
    assert agent.actuators == [dof1.actuator, dof2.actuator]
    assert agent.evaluation_function == mock_evaluation_function
    assert agent.acquisition_plan is None
    assert agent.current_iteration == 0
    assert isinstance(agent.ax_client, Client)

    problem = agent.to_optimization_problem()
    assert problem.acquisition_plan is None
    assert problem.actuators == [dof1.parameter_name, dof2.parameter_name]
    assert problem.sensors == ["det"]
    assert problem.evaluation_function == mock_evaluation_function


def test_queueserver_agent_init_acquisition_plan_kwargs(
    mock_re_manager_api, mock_document_dispatcher, mock_evaluation_function
):
    dof1 = RangeDOF(actuator="test_motor1", bounds=(0, 10), parameter_type="float")
    agent = QueueserverAgent(
        mock_re_manager_api,
        mock_document_dispatcher,
        ["det"],
        [dof1],
        [Objective(name="obj1", minimize=False)],
        mock_evaluation_function,
        acquisition_plan_kwargs={"exposure_time": 0.5, "num_frames": 10},
    )

    problem = agent.to_optimization_problem()
    assert problem.acquisition_plan_kwargs == {"exposure_time": 0.5, "num_frames": 10}


def test_queueserver_agent_init_actuator_instance(mock_re_manager_api, mock_document_dispatcher, mock_evaluation_function):
    movable1 = MovableSignal(name="test_movable1")
    dof1 = RangeDOF(actuator=movable1, bounds=(0, 10), parameter_type="float")
    dof2 = RangeDOF(actuator="test_movable2", bounds=(0, 10), parameter_type="float")
    agent = QueueserverAgent(
        mock_re_manager_api,
        mock_document_dispatcher,
        ["det"],
        [dof1, dof2],
        [Objective(name="obj1", minimize=False)],
        mock_evaluation_function,
    )

    assert agent.actuators == [movable1.name, dof2.parameter_name]


@patch("blop.ax.queueserver_agent.QueueserverClient")
@patch("blop.ax.queueserver_agent.QueueserverOptimizationRunner")
def test_queueserver_agent_run(
    mock_queueserver_runner_cls,
    mock_queueserver_client_cls,
    mock_re_manager_api,
    mock_document_dispatcher,
    mock_evaluation_function,
):
    dof1 = RangeDOF(actuator="test_motor1", bounds=(0, 10), parameter_type="float")
    dof2 = RangeDOF(actuator="test_motor2", bounds=(0, 10), parameter_type="float")
    agent = QueueserverAgent(
        mock_re_manager_api,
        mock_document_dispatcher,
        ["det"],
        [dof1, dof2],
        [Objective(name="obj1", minimize=False)],
        mock_evaluation_function,
    )
    mock_queueserver_client_cls.assert_called_once_with(mock_re_manager_api, mock_document_dispatcher)
    mock_queueserver_runner_cls.assert_called_once()

    agent.run()
    mock_queueserver_runner_cls.return_value.run.assert_called_once_with(
        iterations=1, num_points=1, checkpoint_interval=None
    )


@patch("blop.ax.queueserver_agent.QueueserverClient")
@patch("blop.ax.queueserver_agent.QueueserverOptimizationRunner")
def test_queueserver_agent_run_with_checkpoint_interval(
    mock_queueserver_runner_cls,
    mock_queueserver_client_cls,
    mock_re_manager_api,
    mock_document_dispatcher,
    mock_evaluation_function,
):
    dof1 = RangeDOF(actuator="test_motor1", bounds=(0, 10), parameter_type="float")
    dof2 = RangeDOF(actuator="test_motor2", bounds=(0, 10), parameter_type="float")
    agent = QueueserverAgent(
        mock_re_manager_api,
        mock_document_dispatcher,
        ["det"],
        [dof1, dof2],
        [Objective(name="obj1", minimize=False)],
        mock_evaluation_function,
    )
    mock_queueserver_client_cls.assert_called_once_with(mock_re_manager_api, mock_document_dispatcher)
    mock_queueserver_runner_cls.assert_called_once()

    agent.run(checkpoint_interval=1)
    mock_queueserver_runner_cls.return_value.run.assert_called_once_with(iterations=1, num_points=1, checkpoint_interval=1)


@patch("blop.ax.queueserver_agent.QueueserverClient")
@patch("blop.ax.queueserver_agent.QueueserverOptimizationRunner")
def test_queueserver_agent_submit_suggestions(
    mock_queueserver_runner_cls,
    mock_queueserver_client_cls,
    mock_re_manager_api,
    mock_document_dispatcher,
    mock_evaluation_function,
):
    dof1 = RangeDOF(actuator="test_motor1", bounds=(0, 10), parameter_type="float")
    dof2 = RangeDOF(actuator="test_motor2", bounds=(0, 10), parameter_type="float")
    agent = QueueserverAgent(
        mock_re_manager_api,
        mock_document_dispatcher,
        ["det"],
        [dof1, dof2],
        [Objective(name="obj1", minimize=False)],
        mock_evaluation_function,
    )
    mock_queueserver_client_cls.assert_called_once_with(mock_re_manager_api, mock_document_dispatcher)
    mock_queueserver_runner_cls.assert_called_once()

    suggestions = [{"test_motor1": 5, "test_motor2": 9}]
    agent.submit_suggestions(suggestions)
    mock_queueserver_runner_cls.return_value.submit_suggestions.assert_called_once_with(suggestions)


@patch("blop.ax.queueserver_agent.QueueserverClient")
@patch("blop.ax.queueserver_agent.QueueserverOptimizationRunner")
def test_queueserver_agent_stop(
    mock_queueserver_runner_cls,
    mock_queueserver_client_cls,
    mock_re_manager_api,
    mock_document_dispatcher,
    mock_evaluation_function,
):
    dof1 = RangeDOF(actuator="test_motor1", bounds=(0, 10), parameter_type="float")
    dof2 = RangeDOF(actuator="test_motor2", bounds=(0, 10), parameter_type="float")
    agent = QueueserverAgent(
        mock_re_manager_api,
        mock_document_dispatcher,
        ["det"],
        [dof1, dof2],
        [Objective(name="obj1", minimize=False)],
        mock_evaluation_function,
    )
    mock_queueserver_client_cls.assert_called_once_with(mock_re_manager_api, mock_document_dispatcher)
    mock_queueserver_runner_cls.assert_called_once()

    agent.stop()
    mock_queueserver_runner_cls.return_value.stop.assert_called_once()
