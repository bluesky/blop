from unittest.mock import MagicMock, patch

import pytest

from blop.protocols import OptimizationProblem
from blop.queueserver import ConsumerCallback, QServerClient, QServerOptimizationRunner


@pytest.fixture(scope="function")
def mock_optimization_problem():
    """Create a mock OptimizationProblem with necessary components."""
    mock_optimizer = MagicMock()
    mock_optimizer.suggest.return_value = [
        {"_id": 0, "motor1": 5.0, "motor2": 3.0},
    ]

    mock_actuator1 = MagicMock()
    mock_actuator1.name = "motor1"
    mock_actuator2 = MagicMock()
    mock_actuator2.name = "motor2"

    mock_sensor = MagicMock()
    mock_sensor.name = "detector"

    mock_eval_func = MagicMock()
    mock_eval_func.return_value = [{"_id": 0, "objective": 1.0}]

    return OptimizationProblem(
        optimizer=mock_optimizer,
        actuators=[mock_actuator1, mock_actuator2],
        sensors=[mock_sensor],
        evaluation_function=mock_eval_func,
    )


def test_consumer_callback_caches_start_and_calls_on_stop():
    """Test ConsumerCallback caches start doc and calls callback on stop."""
    mock_callback = MagicMock()
    callback = ConsumerCallback(callback=mock_callback)
    start_doc = {"uid": "test-uid", "time": 123}
    stop_doc = {"uid": "test-uid", "exit_status": "success"}

    callback.start(start_doc)
    mock_callback.assert_not_called()

    callback.stop(stop_doc)
    mock_callback.assert_called_once_with(start_doc, stop_doc)


def test_consumer_callback_clears_cache_after_stop():
    """Test ConsumerCallback clears cache after stop is called."""
    callback = ConsumerCallback(callback=MagicMock())
    start_doc = {"uid": "test-uid"}
    stop_doc = {"uid": "test-uid"}

    callback.start(start_doc)
    callback.stop(stop_doc)

    # Second stop should not call callback (no cached start doc)
    callback.stop(stop_doc)
    assert callback._callback.call_count == 1


@patch("blop.queueserver.REManagerAPI")
def test_qserver_client_check_environment_raises_when_not_ready(mock_re_manager):
    """Test check_environment raises RuntimeError when environment not open."""
    client = QServerClient()
    client._rm.status.return_value = {"worker_environment_exists": False}

    with pytest.raises(RuntimeError, match="queueserver environment is not open"):
        client.check_environment()


@patch("blop.ax.queueserver.REManagerAPI")
def test_qserver_client_check_devices_raises_for_missing_device(mock_re_manager):
    """Test check_devices_available raises ValueError for missing devices."""
    client = QServerClient()
    client._rm.devices_allowed.return_value = {"devices_allowed": {"motor1": {}}}

    with pytest.raises(ValueError, match="Device 'motor2' is not available"):
        client.check_devices_available(["motor1", "motor2"])


@patch("blop.ax.queueserver.REManagerAPI")
def test_qserver_client_check_plan_raises_for_missing_plan(mock_re_manager):
    """Test check_plan_available raises ValueError for missing plan."""
    client = QServerClient()
    client._rm.plans_allowed.return_value = {"plans_allowed": {"other_plan": {}}}

    with pytest.raises(ValueError, match="Plan 'my_plan' is not available"):
        client.check_plan_available("my_plan")


@patch("blop.ax.queueserver.REManagerAPI")
def test_qserver_client_submit_plan_with_autostart(mock_re_manager):
    """Test submit_plan adds item and starts queue when autostart=True."""
    client = QServerClient()
    mock_plan = MagicMock()

    client.submit_plan(mock_plan, autostart=True)

    client._rm.item_add.assert_called_once_with(mock_plan)
    client._rm.wait_for_idle_or_paused.assert_called_once()
    client._rm.queue_start.assert_called_once()


@patch("blop.ax.queueserver.REManagerAPI")
def test_qserver_client_submit_plan_without_autostart(mock_re_manager):
    """Test submit_plan only adds item when autostart=False."""
    client = QServerClient()
    mock_plan = MagicMock()

    client.submit_plan(mock_plan, autostart=False)

    client._rm.item_add.assert_called_once_with(mock_plan)
    client._rm.queue_start.assert_not_called()


@patch("blop.ax.queueserver.REManagerAPI")
def test_runner_run_validates_environment(mock_re_manager, mock_optimization_problem):
    """Test run() validates qserver environment before starting."""
    mock_client = MagicMock(spec=QServerClient)
    mock_client.check_environment.side_effect = RuntimeError("not open")

    runner = QServerOptimizationRunner(
        optimization_problem=mock_optimization_problem,
        qserver_client=mock_client,
    )

    with pytest.raises(RuntimeError, match="not open"):
        runner.run(iterations=1)

    mock_client.check_environment.assert_called_once()


@patch("blop.ax.queueserver.REManagerAPI")
def test_runner_run_submits_suggestions_to_qserver(mock_re_manager, mock_optimization_problem):
    """Test run() gets suggestions from optimizer and submits plan to qserver."""
    mock_client = MagicMock(spec=QServerClient)
    runner = QServerOptimizationRunner(
        optimization_problem=mock_optimization_problem,
        qserver_client=mock_client,
        acquisition_plan_name="my_acquire",
    )

    runner.run(iterations=1, num_points=1)

    # Verify optimizer.suggest was called
    mock_optimization_problem.optimizer.suggest.assert_called_once_with(1)

    # Verify plan was submitted
    mock_client.submit_plan.assert_called_once()
    submitted_plan = mock_client.submit_plan.call_args[0][0]
    assert submitted_plan.name == "my_acquire"


@patch("blop.ax.queueserver.REManagerAPI")
def test_runner_stop_sets_finished_state(mock_re_manager, mock_optimization_problem):
    """Test stop() marks the runner as finished and stops listener."""
    mock_client = MagicMock(spec=QServerClient)
    runner = QServerOptimizationRunner(
        optimization_problem=mock_optimization_problem,
        qserver_client=mock_client,
    )

    runner.run(iterations=10)
    assert runner.is_running is True

    runner.stop()

    assert runner.is_running is False
    mock_client.stop_listener.assert_called()


@patch("blop.ax.queueserver.REManagerAPI")
def test_runner_ingests_outcomes_on_acquisition_complete(mock_re_manager, mock_optimization_problem):
    """Test that outcomes are ingested into optimizer when acquisition completes."""
    mock_client = MagicMock(spec=QServerClient)
    runner = QServerOptimizationRunner(
        optimization_problem=mock_optimization_problem,
        qserver_client=mock_client,
    )

    # Start the runner (sets up state)
    runner.run(iterations=1, num_points=1)

    # Simulate acquisition completion callback
    runner._on_acquisition_complete(
        start_doc={"uid": "run-uid"},
        stop_doc={"exit_status": "success"},
    )

    # Verify evaluation function was called
    mock_optimization_problem.evaluation_function.assert_called_once()

    # Verify outcomes were ingested
    mock_optimization_problem.optimizer.ingest.assert_called_once()
