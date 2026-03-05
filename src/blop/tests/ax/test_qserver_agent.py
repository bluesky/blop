from unittest.mock import MagicMock, patch

import pytest

from blop.ax.dof import RangeDOF
from blop.ax.objective import Objective
from blop.ax.qserver_agent import BlopQserverAgent, ConsumerCallback, ZMQConsumer
from blop.protocols import EvaluationFunction

from ..conftest import MovableSignal, ReadableSignal


@pytest.fixture(scope="function")
def mock_evaluation_function():
    return MagicMock(spec=EvaluationFunction)


@pytest.fixture(scope="function")
def basic_dofs():
    movable1 = MovableSignal(name="test_movable1")
    movable2 = MovableSignal(name="test_movable2")
    dof1 = RangeDOF(actuator=movable1, bounds=(0, 10), parameter_type="float")
    dof2 = RangeDOF(actuator=movable2, bounds=(0, 10), parameter_type="float")
    return [dof1, dof2]


@pytest.fixture(scope="function")
def basic_sensors():
    return [ReadableSignal(name="test_readable")]


@pytest.fixture(scope="function")
def basic_objective():
    return Objective(name="test_objective", minimize=False)


@patch("blop.ax.qserver_agent.REManagerAPI")
@patch("blop.ax.qserver_agent.ZMQConsumer")
def test_qserver_agent_init(mock_zmq_consumer, mock_re_manager, mock_evaluation_function, basic_dofs, basic_sensors, basic_objective):
    """Test that the qserver agent can be initialized with proper components."""
    agent = BlopQserverAgent(
        sensors=basic_sensors,
        dofs=basic_dofs,
        objectives=[basic_objective],
        evaluation_function=mock_evaluation_function,
        acquisition_plan="test_plan",
        qserver_control_addr="tcp://localhost:60615",
        qserver_info_addr="tcp://localhost:60625",
        zmq_consumer_ip="localhost",
        zmq_consumer_port="5578",
    )

    # Test public properties
    assert agent.sensors == basic_sensors
    assert list(agent.dofs) == basic_dofs
    assert agent.continuous_suggestion is True
    assert agent.num_itterations == 30
    assert agent.n_points == 1
    assert agent.current_itteration == 0

    # Verify REManagerAPI was initialized with correct addresses
    mock_re_manager.assert_called_once_with(
        zmq_control_addr="tcp://localhost:60615",
        zmq_info_addr="tcp://localhost:60625",
    )
    # Verify ZMQ consumer was started
    mock_zmq_consumer.assert_called_once()
    mock_zmq_consumer.return_value.start_zmq_listener_thread.assert_called_once()


@patch("blop.ax.qserver_agent.REManagerAPI")
@patch("blop.ax.qserver_agent.ZMQConsumer")
def test_qserver_agent_optimize_validation(mock_zmq_consumer, mock_re_manager, mock_evaluation_function, basic_dofs, basic_sensors, basic_objective):
    """Test that optimize validates qserver environment, devices, and plans."""
    # Test 1: Error when qserver environment is not open
    agent = BlopQserverAgent(
        sensors=basic_sensors,
        dofs=basic_dofs,
        objectives=[basic_objective],
        evaluation_function=mock_evaluation_function,
    )
    agent.RM.status.return_value = {"worker_environment_exists": False}

    with pytest.raises(ValueError, match="queueserver environment is not open"):
        agent.optimize(iterations=1, n_points=1)

    # Test 2: Error when required device is not in qserver (uses actuator-based DOFs)
    agent2 = BlopQserverAgent(
        sensors=basic_sensors,
        dofs=basic_dofs,
        objectives=[basic_objective],
        evaluation_function=mock_evaluation_function,
    )
    agent2.RM.status.return_value = {"worker_environment_exists": True}
    agent2.RM.devices_allowed.return_value = {"devices_allowed": {}}

    with pytest.raises(ValueError, match="device test_movable1 is not in the Queueserver Environment"):
        agent2.optimize(iterations=1, n_points=1)

    # Test 3: Error when acquisition plan is not in qserver
    agent3 = BlopQserverAgent(
        sensors=basic_sensors,
        dofs=basic_dofs,
        objectives=[basic_objective],
        evaluation_function=mock_evaluation_function,
        acquisition_plan="missing_plan",
    )
    agent3.RM.status.return_value = {"worker_environment_exists": True}
    agent3.RM.devices_allowed.return_value = {
        "devices_allowed": {"test_movable1": {}, "test_movable2": {}, "test_readable": {}}
    }
    agent3.RM.plans_allowed.return_value = {"plans_allowed": {}}

    with pytest.raises(ValueError, match="plan missing_plan is not in the Queueserver Environment"):
        agent3.optimize(iterations=1, n_points=1)


@patch("blop.ax.qserver_agent.REManagerAPI")
@patch("blop.ax.qserver_agent.ZMQConsumer")
def test_qserver_agent_acquire(mock_zmq_consumer, mock_re_manager, mock_evaluation_function, basic_dofs, basic_sensors, basic_objective):
    """Test that acquire submits a plan to the qserver with proper metadata and starts queue."""
    agent = BlopQserverAgent(
        sensors=basic_sensors,
        dofs=basic_dofs,
        objectives=[basic_objective],
        evaluation_function=mock_evaluation_function,
        acquisition_plan="acquire",
    )

    trials = {0: {"test_movable1": 5.0, "test_movable2": 3.0}}
    uid = agent.acquire(trials)

    # Verify uid is returned
    assert uid is not None
    assert isinstance(uid, str)
    assert agent.acquisition_finished is False

    # Verify plan was submitted to qserver
    agent.RM.item_add.assert_called_once()
    call_args = agent.RM.item_add.call_args
    bplan = call_args[0][0]
    assert bplan.name == "acquire"
    assert bplan.kwargs["md"]["agent_suggestion_uid"] == uid
    assert bplan.kwargs["md"]["blop_suggestions"] == [{"_id": 0, "test_movable1": 5.0, "test_movable2": 3.0}]

    # Verify queue was started (autostart is enabled by default)
    agent.RM.wait_for_idle_or_paused.assert_called_once_with(timeout=600)
    agent.RM.queue_start.assert_called_once()


@patch("blop.ax.qserver_agent.REManagerAPI")
@patch("blop.ax.qserver_agent.ZMQConsumer")
def test_qserver_agent_stop(mock_zmq_consumer, mock_re_manager, mock_evaluation_function, basic_dofs, basic_sensors, basic_objective):
    """Test that stop prevents the agent from auto-starting the queue on acquire."""
    agent = BlopQserverAgent(
        sensors=basic_sensors,
        dofs=basic_dofs,
        objectives=[basic_objective],
        evaluation_function=mock_evaluation_function,
    )

    # Stop the agent
    agent.stop()

    # Now acquire should not start the queue
    trials = {0: {"test_movable1": 5.0, "test_movable2": 3.0}}
    agent.acquire(trials)

    # Verify plan was still submitted
    agent.RM.item_add.assert_called_once()
    # But queue was NOT started
    agent.RM.queue_start.assert_not_called()


def test_consumer_callback():
    """Test ConsumerCallback caches start doc, calls callback on stop, and clears cache."""
    mock_callback = MagicMock()
    callback = ConsumerCallback(callback=mock_callback, enable=True)
    start_doc = {"uid": "test-uid", "time": 123}
    stop_doc = {"uid": "test-uid", "exit_status": "success"}

    # Test caching on start
    callback.start(start_doc)
    assert callback.start_doc_cache == start_doc

    # Test callback invocation on stop and cache clearing
    callback.stop(stop_doc)
    mock_callback.assert_called_once_with(start_doc, stop_doc)
    assert callback.start_doc_cache is None

    # Test disabled callback does nothing
    disabled_callback = ConsumerCallback(callback=MagicMock(), enable=False)
    disabled_callback.start(start_doc)
    disabled_callback.stop(stop_doc)
    assert disabled_callback.start_doc_cache is None
    disabled_callback.callback.assert_not_called()


@patch("blop.ax.qserver_agent.RemoteDispatcher")
def test_zmq_consumer(mock_remote_dispatcher):
    """Test ZMQConsumer initialization and thread startup."""
    mock_callback = MagicMock()

    consumer = ZMQConsumer(
        zmq_consumer_ip_address="localhost",
        zmq_consumer_port="5578",
        callback=mock_callback,
    )

    # Test public attributes
    assert consumer.zmq_consumer_ip_address == "localhost"
    assert consumer.zmq_consumer_port == "5578"

    # Verify RemoteDispatcher was initialized and subscribed
    mock_remote_dispatcher.assert_called_once_with("localhost:5578")
    mock_remote_dispatcher.return_value.subscribe.assert_called_once()

    # Test thread startup (verify it doesn't raise)
    consumer.start_zmq_listener_thread()
