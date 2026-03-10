from unittest.mock import MagicMock, patch

import pytest

from blop.protocols import CanRegisterSuggestions, Optimizer, QueueserverOptimizationProblem
from blop.queueserver import CORRELATION_UID_KEY, ConsumerCallback, QueueserverClient, QueueserverOptimizationRunner


@pytest.fixture(scope="function")
def mock_optimization_problem():
    """Create a mock OptimizationProblem with necessary components."""
    mock_optimizer = MagicMock()
    mock_optimizer.suggest.return_value = [
        {"_id": 0, "motor1": 5.0, "motor2": 3.0},
    ]

    mock_eval_func = MagicMock()
    mock_eval_func.return_value = [{"_id": 0, "objective": 1.0}]

    return QueueserverOptimizationProblem(
        optimizer=mock_optimizer,
        actuators=["motor1", "motor2"],
        sensors=["detector"],
        evaluation_function=mock_eval_func,
    )


def test_consumer_callback_caches_start_and_calls_on_stop():
    """Test ConsumerCallback caches start doc and calls callback on stop."""
    mock_callback = MagicMock()
    callback = ConsumerCallback(callback=mock_callback)
    start_doc = {"uid": "test-uid", CORRELATION_UID_KEY: "123", "time": 123}
    stop_doc = {"uid": "test-uid", "exit_status": "success"}

    callback.start(start_doc)
    mock_callback.assert_not_called()

    callback.stop(stop_doc)
    mock_callback.assert_called_once_with(start_doc, stop_doc)


def test_consumer_callback_clears_cache_after_stop():
    """Test ConsumerCallback clears cache after stop is called."""
    mock_callback = MagicMock()
    callback = ConsumerCallback(callback=mock_callback)
    start_doc = {"uid": "test-uid", CORRELATION_UID_KEY: "123"}
    stop_doc = {"uid": "test-uid"}

    callback.start(start_doc)
    callback.stop(stop_doc)

    # Second stop should not call callback (no cached start doc)
    callback.stop(stop_doc)
    assert mock_callback.call_count == 1


@patch("blop.queueserver.REManagerAPI")
def test_queueserver_client_check_environment_raises_when_not_ready(mock_re_manager):
    """Test check_environment raises RuntimeError when environment not open."""
    mock_re_manager.status.return_value = {"worker_environment_exists": False}
    client = QueueserverClient(mock_re_manager, "inproc://test")

    with pytest.raises(RuntimeError, match="queueserver environment is not open"):
        client.check_environment()


@patch("blop.queueserver.REManagerAPI")
def test_queueserver_client_check_devices_raises_for_missing_device(mock_re_manager):
    """Test check_devices_available raises ValueError for missing devices."""
    mock_re_manager.devices_allowed.return_value = {"devices_allowed": {"motor1": {}}}
    client = QueueserverClient(mock_re_manager, "inproc://test")

    with pytest.raises(ValueError, match="Device 'motor2' is not available"):
        client.check_devices_available(["motor1", "motor2"])


@patch("blop.queueserver.REManagerAPI")
def test_queueserver_client_check_plan_raises_for_missing_plan(mock_re_manager):
    """Test check_plan_available raises ValueError for missing plan."""
    mock_re_manager.plans_allowed.return_value = {"plans_allowed": {"other_plan": {}}}
    client = QueueserverClient(mock_re_manager, "inproc://test")

    with pytest.raises(ValueError, match="Plan 'my_plan' is not available"):
        client.check_plan_available("my_plan")


@patch("blop.queueserver.REManagerAPI")
def test_queueserver_client_submit_plan_with_autostart(mock_re_manager):
    """Test submit_plan adds item and starts queue when autostart=True."""
    client = QueueserverClient(mock_re_manager, "inproc://test")
    mock_plan = MagicMock()

    client.submit_plan(mock_plan, autostart=True)

    mock_re_manager.item_add.assert_called_once_with(mock_plan)
    mock_re_manager.wait_for_idle_or_paused.assert_called_once()
    mock_re_manager.queue_start.assert_called_once()


@patch("blop.queueserver.REManagerAPI")
def test_queueserver_client_submit_plan_without_autostart(mock_re_manager):
    """Test submit_plan only adds item when autostart=False."""
    client = QueueserverClient(mock_re_manager, "inproc://test")
    mock_plan = MagicMock()

    client.submit_plan(mock_plan, autostart=False)

    mock_re_manager.item_add.assert_called_once_with(mock_plan)
    mock_re_manager.queue_start.assert_not_called()


@patch("blop.queueserver.threading.Thread")
@patch("blop.queueserver.RemoteDispatcher")
@patch("blop.queueserver.REManagerAPI")
def test_queueserver_client_start_listener(mock_re_manager, mock_dispatcher_cls, mock_thread_cls):
    """Test start_listener creates dispatcher, subscribes callback, and starts thread."""
    mock_re_manager.status.return_value = {"worker_environment_exists": True}
    mock_re_manager.devices_allowed.return_value = {"devices_allowed": {"motor1": {}, "detector": {}}}
    mock_re_manager.plans_allowed.return_value = {"plans_allowed": {"default_acquire": {}}}

    client = QueueserverClient(mock_re_manager, "tcp://localhost:5578")
    mock_callback = MagicMock()

    client.start_listener(on_stop=mock_callback)

    mock_dispatcher_cls.assert_called_once_with("tcp://localhost:5578")
    mock_dispatcher = mock_dispatcher_cls.return_value
    mock_dispatcher.subscribe.assert_called_once()
    subscribed_callback = mock_dispatcher.subscribe.call_args[0][0]
    assert isinstance(subscribed_callback, ConsumerCallback)
    assert subscribed_callback._callback is mock_callback

    mock_thread_cls.assert_called_once()
    call_kwargs = mock_thread_cls.call_args[1]
    assert call_kwargs["target"] == mock_dispatcher.start
    mock_thread_cls.return_value.start.assert_called_once()


@patch("blop.queueserver.threading.Thread")
@patch("blop.queueserver.RemoteDispatcher")
@patch("blop.queueserver.REManagerAPI")
def test_queueserver_client_start_listener_already_running_returns_early(
    mock_re_manager, mock_dispatcher_cls, mock_thread_cls
):
    """Test start_listener returns early when listener is already running."""
    mock_re_manager.status.return_value = {"worker_environment_exists": True}
    mock_re_manager.devices_allowed.return_value = {"devices_allowed": {"motor1": {}, "detector": {}}}
    mock_re_manager.plans_allowed.return_value = {"plans_allowed": {"default_acquire": {}}}

    client = QueueserverClient(mock_re_manager, "tcp://localhost:5578")
    client._listener_thread = MagicMock()  # Simulate already running

    client.start_listener(on_stop=MagicMock())

    mock_dispatcher_cls.assert_not_called()
    mock_thread_cls.assert_not_called()


@patch("blop.queueserver.threading.Thread")
@patch("blop.queueserver.RemoteDispatcher")
@patch("blop.queueserver.REManagerAPI")
def test_queueserver_client_stop_listener(mock_re_manager, mock_dispatcher_cls, mock_thread_cls):
    """Test stop_listener stops dispatcher and clears state."""
    mock_re_manager.status.return_value = {"worker_environment_exists": True}
    mock_re_manager.devices_allowed.return_value = {"devices_allowed": {"motor1": {}, "detector": {}}}
    mock_re_manager.plans_allowed.return_value = {"plans_allowed": {"default_acquire": {}}}

    client = QueueserverClient(mock_re_manager, "tcp://localhost:5578")
    client.start_listener(on_stop=MagicMock())

    client.stop_listener()

    mock_dispatcher_cls.return_value.stop.assert_called_once()
    assert client._dispatcher is None
    assert client._consumer_callback is None
    assert client._listener_thread is None


@patch("blop.queueserver.REManagerAPI")
def test_queueserver_client_stop_listener_when_not_started(mock_re_manager):
    """Test stop_listener is safe to call when listener was never started."""
    client = QueueserverClient(mock_re_manager, "inproc://test")

    client.stop_listener()  # Should not raise

    assert client._dispatcher is None
    assert client._listener_thread is None


def test_runner_run_validates_environment(mock_optimization_problem):
    """Test run() validates queueserver environment before starting."""
    mock_client = MagicMock(spec=QueueserverClient)
    mock_client.check_environment.side_effect = RuntimeError("not open")

    runner = QueueserverOptimizationRunner(
        optimization_problem=mock_optimization_problem,
        queueserver_client=mock_client,
    )

    with pytest.raises(RuntimeError, match="not open"):
        runner.run(iterations=1)

    mock_client.check_environment.assert_called_once()


def test_runner_run_submits_suggestions_to_queueserver():
    """Test run() gets suggestions from optimizer and submits plan to queueserver."""
    mock_client = MagicMock(spec=QueueserverClient)
    mock_optimization_problem = QueueserverOptimizationProblem(
        optimizer=MagicMock(),
        actuators=["motor1"],
        sensors=["det"],
        evaluation_function=MagicMock(),
        acquisition_plan="my_acquire",
    )
    runner = QueueserverOptimizationRunner(
        optimization_problem=mock_optimization_problem,
        queueserver_client=mock_client,
    )
    assert runner.optimization_problem == mock_optimization_problem

    runner.run(iterations=1, num_points=1)

    # Verify optimizer.suggest was called
    mock_optimization_problem.optimizer.suggest.assert_called_once_with(1)

    # Verify plan was submitted
    mock_client.submit_plan.assert_called_once()
    submitted_plan = mock_client.submit_plan.call_args[0][0]
    assert submitted_plan.name == "my_acquire"


def test_runner_stop_sets_finished_state(mock_optimization_problem):
    """Test stop() marks the runner as finished and stops listener."""
    mock_client = MagicMock(spec=QueueserverClient)
    runner = QueueserverOptimizationRunner(
        optimization_problem=mock_optimization_problem,
        queueserver_client=mock_client,
    )

    # The acquisiton completion callback never fires here due to the mocked client, therefore
    # the first plan runs forever
    runner.run(10)
    assert runner.is_running is True

    runner.stop()
    assert runner.is_running is False
    mock_client.stop_listener.assert_called()


def test_runner_submit_suggestions_to_queueserver():
    """Test run() gets suggestions from optimizer and submits plan to queueserver."""
    mock_client = MagicMock(spec=QueueserverClient)

    class CustomOptimizer(Optimizer, CanRegisterSuggestions): ...

    mock_optimization_problem = QueueserverOptimizationProblem(
        optimizer=MagicMock(spec=CustomOptimizer),
        actuators=["motor1"],
        sensors=["det"],
        evaluation_function=MagicMock(),
        acquisition_plan="my_acquire",
    )
    runner = QueueserverOptimizationRunner(
        optimization_problem=mock_optimization_problem,
        queueserver_client=mock_client,
    )

    suggestions = [{"motor1": 5}]
    runner.submit_suggestions(suggestions)

    # Verify optimizer.suggest was NOT called
    mock_optimization_problem.optimizer.suggest.assert_not_called()
    mock_optimization_problem.optimizer.register_suggestions.assert_called_once_with(suggestions)

    # Verify plan was submitted
    mock_client.submit_plan.assert_called_once()
    submitted_plan = mock_client.submit_plan.call_args[0][0]
    assert submitted_plan.name == "my_acquire"


def test_runner_run_full_cycle(mock_optimization_problem):
    """Test run() completes full suggest -> acquire -> ingest cycle across 3 iterations."""
    # Configure for num_points=2: suggest returns 2 items, evaluation_function returns 2 outcomes
    mock_optimization_problem.optimizer.suggest.return_value = [
        {"_id": 0, "motor1": 5.0, "motor2": 3.0},
        {"_id": 1, "motor1": 6.0, "motor2": 4.0},
    ]
    mock_optimization_problem.evaluation_function.return_value = [
        {"_id": 0, "objective": 1.0},
        {"_id": 1, "objective": 2.0},
    ]

    mock_client = MagicMock(spec=QueueserverClient)

    def capture_callback(on_stop):
        mock_client._on_stop = on_stop

    mock_client.start_listener.side_effect = capture_callback

    runner = QueueserverOptimizationRunner(
        optimization_problem=mock_optimization_problem,
        queueserver_client=mock_client,
    )

    runner.run(iterations=3, num_points=2)

    # Simulate 3 acquisition completions by invoking the captured callback
    for _ in range(3):
        current_uid = runner._state.current_uid
        uid = f"fake-uid-{_}"
        start_doc = {"uid": uid, CORRELATION_UID_KEY: current_uid}
        stop_doc = {"uid": uid}
        mock_client._on_stop(start_doc, stop_doc)

    assert mock_client.submit_plan.call_count == 3
    assert mock_optimization_problem.optimizer.suggest.call_count == 3
    assert mock_optimization_problem.optimizer.ingest.call_count == 3
    assert mock_optimization_problem.evaluation_function.call_count == 3
    assert runner.is_running is False


def test_runner_on_acquisition_complete_raises_on_uid_mismatch(mock_optimization_problem):
    """Test _on_acquisition_complete raises RuntimeError when blop_correlation_uid does not match."""
    mock_client = MagicMock(spec=QueueserverClient)

    def capture_callback(on_stop):
        mock_client._on_stop = on_stop

    mock_client.start_listener.side_effect = capture_callback

    runner = QueueserverOptimizationRunner(
        optimization_problem=mock_optimization_problem,
        queueserver_client=mock_client,
    )

    runner.run(iterations=1, num_points=1)

    # Callback with wrong blop_correlation_uid should raise
    start_doc = {"uid": "fake-uid", CORRELATION_UID_KEY: "wrong-uid"}
    stop_doc = {"uid": "fake-uid"}

    with pytest.raises(RuntimeError, match="current_uid did not match start document"):
        mock_client._on_stop(start_doc, stop_doc)
