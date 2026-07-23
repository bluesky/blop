import threading
from unittest.mock import MagicMock, patch

import pytest
from bluesky.callbacks.zmq import RemoteDispatcher

from blop.protocols import CanRegisterSuggestions, Optimizer, QueueserverOptimizationProblem, TrialFaultAware
from blop.queueserver import (
    CORRELATION_UID_KEY,
    ConsumerCallback,
    OptimizationResult,
    QueueserverClient,
    QueueserverOptimizationRunner,
)

from .conftest import CheckpointableOptimizer


@pytest.fixture(scope="function")
def mock_document_dispatcher():
    """Create a mock document dispatcher."""
    return MagicMock(spec=RemoteDispatcher)


@pytest.fixture(scope="function")
def mock_optimization_problem():
    """Create a mock OptimizationProblem with necessary components."""
    mock_optimizer = MagicMock(spec=Optimizer)
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
    run_uid = "test-uid"
    start_doc = {"uid": run_uid, CORRELATION_UID_KEY: "123", "time": 123}
    stop_doc = {"uid": "test-uid2", "run_start": run_uid, "exit_status": "success"}

    callback.start(start_doc)
    mock_callback.assert_not_called()

    callback.stop(stop_doc)
    mock_callback.assert_called_once_with(start_doc, stop_doc)


def test_consumer_callback_clears_cache_after_stop():
    """Test ConsumerCallback clears cache after stop is called."""
    mock_callback = MagicMock()
    callback = ConsumerCallback(callback=mock_callback)
    run_uid = "test-uid"
    start_doc = {"uid": run_uid, CORRELATION_UID_KEY: "123"}
    stop_doc = {"uid": "test-uid2", "run_start": run_uid}

    callback.start(start_doc)
    callback.stop(stop_doc)

    # Second stop should not call callback (no cached start doc)
    callback.stop(stop_doc)
    assert mock_callback.call_count == 1


def test_consumer_callback_ignores_start_without_correlation_uid():
    """Test ConsumerCallback ignores non-Blop start documents."""
    mock_callback = MagicMock()
    callback = ConsumerCallback(callback=mock_callback)
    run_uid = "test-uid"
    start_doc = {"uid": run_uid}
    stop_doc = {"uid": "test-uid2", "run_start": run_uid}

    callback.start(start_doc)
    callback.stop(stop_doc)

    mock_callback.assert_not_called()


def test_consumer_callback_matches_stop_to_cached_start_by_run_uid():
    """Test ConsumerCallback matches stops to the correct cached Blop start."""
    mock_callback = MagicMock()
    callback = ConsumerCallback(callback=mock_callback)
    start_doc_1 = {"uid": "run-1", CORRELATION_UID_KEY: "one"}
    start_doc_2 = {"uid": "run-2", CORRELATION_UID_KEY: "two"}
    stop_doc = {"uid": "stop-2", "run_start": "run-2"}

    callback.start(start_doc_1)
    callback.start(start_doc_2)
    callback.stop(stop_doc)

    mock_callback.assert_called_once_with(start_doc_2, stop_doc)


@patch("blop.queueserver.bluesky_queueserver_api.http.REManagerAPI")
def test_queueserver_client_check_environment_raises_when_not_ready(mock_re_manager, mock_document_dispatcher):
    """Test check_environment raises RuntimeError when environment not open."""
    mock_re_manager.status.return_value = {"worker_environment_exists": False}
    client = QueueserverClient(mock_re_manager, mock_document_dispatcher)

    with pytest.raises(RuntimeError, match="queueserver environment is not open"):
        client.check_environment()


@patch("blop.queueserver.bluesky_queueserver_api.http.REManagerAPI")
def test_queueserver_client_check_devices_raises_for_missing_device(mock_re_manager, mock_document_dispatcher):
    """Test check_devices_available raises ValueError for missing devices."""
    mock_re_manager.devices_allowed.return_value = {"devices_allowed": {"motor1": {}}}
    client = QueueserverClient(mock_re_manager, mock_document_dispatcher)

    with pytest.raises(ValueError, match="Device 'motor2' is not available"):
        client.check_devices_available(["motor1", "motor2"])


@patch("blop.queueserver.bluesky_queueserver_api.http.REManagerAPI")
def test_queueserver_client_check_plan_raises_for_missing_plan(mock_re_manager, mock_document_dispatcher):
    """Test check_plan_available raises ValueError for missing plan."""
    mock_re_manager.plans_allowed.return_value = {"plans_allowed": {"other_plan": {}}}
    client = QueueserverClient(mock_re_manager, mock_document_dispatcher)

    with pytest.raises(ValueError, match="Plan 'my_plan' is not available"):
        client.check_plan_available("my_plan")


@patch("blop.queueserver.bluesky_queueserver_api.http.REManagerAPI")
def test_queueserver_client_submit_plan_with_autostart(mock_re_manager, mock_document_dispatcher):
    """Test submit_plan adds item and starts queue when autostart=True."""
    client = QueueserverClient(mock_re_manager, mock_document_dispatcher)
    mock_plan = MagicMock()

    client.submit_plan(mock_plan, autostart=True)

    mock_re_manager.item_add.assert_called_once_with(mock_plan)
    mock_re_manager.wait_for_idle_or_paused.assert_called_once()
    mock_re_manager.queue_start.assert_called_once()


@patch("blop.queueserver.bluesky_queueserver_api.http.REManagerAPI")
def test_queueserver_client_submit_plan_without_autostart(mock_re_manager, mock_document_dispatcher):
    """Test submit_plan only adds item when autostart=False."""
    client = QueueserverClient(mock_re_manager, mock_document_dispatcher)
    mock_plan = MagicMock()

    client.submit_plan(mock_plan, autostart=False)

    mock_re_manager.item_add.assert_called_once_with(mock_plan)
    mock_re_manager.queue_start.assert_not_called()


@patch("blop.queueserver.threading.Thread")
@patch("blop.queueserver.bluesky_queueserver_api.http.REManagerAPI")
def test_queueserver_client_start_listener(mock_re_manager, mock_thread_cls, mock_document_dispatcher):
    """Test start_listener creates dispatcher, subscribes callback, and starts thread."""
    mock_re_manager.status.return_value = {"worker_environment_exists": True}
    mock_re_manager.devices_allowed.return_value = {"devices_allowed": {"motor1": {}, "detector": {}}}
    mock_re_manager.plans_allowed.return_value = {"plans_allowed": {"default_acquire": {}}}

    client = QueueserverClient(mock_re_manager, mock_document_dispatcher)
    mock_callback = MagicMock()

    client.start_listener(on_stop=mock_callback)

    mock_document_dispatcher.subscribe.assert_called_once()
    subscribed_callback = mock_document_dispatcher.subscribe.call_args[0][0]
    assert isinstance(subscribed_callback, ConsumerCallback)
    assert subscribed_callback._callback is mock_callback

    mock_thread_cls.assert_called_once()
    call_kwargs = mock_thread_cls.call_args[1]
    assert call_kwargs["target"] == mock_document_dispatcher.start
    mock_thread_cls.return_value.start.assert_called_once()


@patch("blop.queueserver.threading.Thread")
@patch("blop.queueserver.bluesky_queueserver_api.http.REManagerAPI")
def test_queueserver_client_start_listener_already_running_returns_early(
    mock_re_manager, mock_thread_cls, mock_document_dispatcher
):
    """Test start_listener returns early when listener is already running."""
    mock_re_manager.status.return_value = {"worker_environment_exists": True}
    mock_re_manager.devices_allowed.return_value = {"devices_allowed": {"motor1": {}, "detector": {}}}
    mock_re_manager.plans_allowed.return_value = {"plans_allowed": {"default_acquire": {}}}

    client = QueueserverClient(mock_re_manager, mock_document_dispatcher)
    client._listener_thread = MagicMock()  # Simulate already running

    client.start_listener(on_stop=MagicMock())

    mock_document_dispatcher.subscribe.assert_not_called()
    mock_thread_cls.assert_not_called()


@patch("blop.queueserver.threading.Thread")
@patch("blop.queueserver.bluesky_queueserver_api.http.REManagerAPI")
def test_queueserver_client_stop_listener(mock_re_manager, mock_thread_cls, mock_document_dispatcher):
    """Test stop_listener stops dispatcher and clears state."""
    mock_re_manager.status.return_value = {"worker_environment_exists": True}
    mock_re_manager.devices_allowed.return_value = {"devices_allowed": {"motor1": {}, "detector": {}}}
    mock_re_manager.plans_allowed.return_value = {"plans_allowed": {"default_acquire": {}}}

    client = QueueserverClient(mock_re_manager, mock_document_dispatcher)
    client.start_listener(on_stop=MagicMock())

    client.stop_listener()

    mock_document_dispatcher.stop.assert_called_once()
    assert client._dispatcher is mock_document_dispatcher
    assert client._consumer_callback is None
    assert client._listener_thread is None


@patch("blop.queueserver.bluesky_queueserver_api.http.REManagerAPI")
def test_queueserver_client_stop_listener_when_not_started(mock_re_manager, mock_document_dispatcher):
    """Test stop_listener is safe to call when listener was never started."""
    client = QueueserverClient(mock_re_manager, mock_document_dispatcher)

    client.stop_listener()  # Should not raise

    mock_document_dispatcher.stop.assert_not_called()
    assert client._dispatcher is mock_document_dispatcher
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

    future = runner.run(iterations=1, num_points=1)

    # Verify listener is started once during runner construction, not per run
    mock_client.start_listener.assert_called_once_with(on_stop=runner._on_acquisition_complete)

    # Verify optimizer.suggest was called
    mock_optimization_problem.optimizer.suggest.assert_called_once_with(1)

    # Verify plan was submitted
    mock_client.submit_plan.assert_called_once()
    submitted_plan = mock_client.submit_plan.call_args[0][0]
    assert submitted_plan.name == "my_acquire"

    # Future should be pending (acquisition callback has not fired)
    assert not future.done()


def test_runner_run_passes_acquisition_plan_kwargs_to_bplan():
    """Test that acquisition_plan_kwargs are forwarded to the submitted BPlan."""
    mock_client = MagicMock(spec=QueueserverClient)
    mock_optimization_problem = QueueserverOptimizationProblem(
        optimizer=MagicMock(),
        actuators=["motor1"],
        sensors=["det"],
        evaluation_function=MagicMock(),
        acquisition_plan="my_acquire",
        acquisition_plan_kwargs={"exposure_time": 0.5, "num_frames": 10},
    )
    runner = QueueserverOptimizationRunner(
        optimization_problem=mock_optimization_problem,
        queueserver_client=mock_client,
    )

    runner.run(iterations=1, num_points=1)

    submitted_plan = mock_client.submit_plan.call_args[0][0]
    assert submitted_plan.kwargs["exposure_time"] == 0.5
    assert submitted_plan.kwargs["num_frames"] == 10


def test_runner_run_twice_fails():
    """Test 2 calls to run() fails."""
    submit_event = threading.Event()

    def set_event(*args, **kwargs):
        submit_event.set()

    mock_client = MagicMock(spec=QueueserverClient)
    mock_client.submit_plan.side_effect = set_event
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
    submit_event.wait(timeout=5)
    if not submit_event.is_set():
        pytest.fail("Submit event timed out")

    with pytest.raises(RuntimeError, match="already running"):
        runner.run(iterations=1, num_points=1)

    with pytest.raises(RuntimeError, match="already running"):
        suggestions = [{"motor1": 5}]
        runner.submit_suggestions(suggestions)


def test_runner_stop_returns_partial_result(mock_optimization_problem):
    """Test stop() marks the runner as finished, stops listener, and resolves the future."""
    mock_client = MagicMock(spec=QueueserverClient)
    runner = QueueserverOptimizationRunner(
        optimization_problem=mock_optimization_problem,
        queueserver_client=mock_client,
    )

    # The acquisition completion callback never fires here due to the mocked client,
    # so the first plan runs forever until stop() is called.
    future = runner.run(10)
    assert not future.done()

    runner.stop()

    assert future.done()
    # TODO: possible stopping bug in remote dispatcher
    mock_client.stop_listener.assert_not_called()
    result = future.result()
    assert isinstance(result, OptimizationResult)
    assert result.iterations_completed == 0
    assert result.uids == ()


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
    future = runner.submit_suggestions(suggestions)

    # Verify listener is started once during runner construction, not per submission
    mock_client.start_listener.assert_called_once_with(on_stop=runner._on_acquisition_complete)

    # Verify optimizer.suggest was NOT called
    mock_optimization_problem.optimizer.suggest.assert_not_called()
    mock_optimization_problem.optimizer.register_suggestions.assert_called_once_with(suggestions)

    # Verify plan was submitted
    mock_client.submit_plan.assert_called_once()
    submitted_plan = mock_client.submit_plan.call_args[0][0]
    assert submitted_plan.name == "my_acquire"

    assert not future.done()


def test_runner_submit_suggestions_register_fails():
    """Test run() gets suggestions from optimizer and submits plan to queueserver."""
    mock_client = MagicMock(spec=QueueserverClient)

    mock_optimization_problem = QueueserverOptimizationProblem(
        optimizer=MagicMock(spec=Optimizer),
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
    with pytest.raises(ValueError, match="'_id'"):
        runner.submit_suggestions(suggestions)

    # Verify optimizer.suggest was NOT called
    mock_optimization_problem.optimizer.suggest.assert_not_called()


def test_runner_submit_suggestions_twice_fails():
    """Test 2 calls to submit_suggestions() fails."""
    submit_event = threading.Event()

    def set_event(*args, **kwargs):
        submit_event.set()

    mock_client = MagicMock(spec=QueueserverClient)
    mock_client.submit_plan.side_effect = set_event

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
    submit_event.wait(timeout=5)
    if not submit_event.is_set():
        pytest.fail("Submit event timed out")

    with pytest.raises(RuntimeError, match="already running"):
        runner.submit_suggestions(suggestions)

    with pytest.raises(RuntimeError, match="already running"):
        runner.run(iterations=1)


def _make_runner_with_captured_callback(mock_optimization_problem, iterations=3, checkpoint_interval=None):
    """Helper: build a runner and capture the on_stop callback via start_listener side-effect."""
    mock_client = MagicMock(spec=QueueserverClient)

    def capture_callback(on_stop):
        mock_client._on_stop = on_stop

    mock_client.start_listener.side_effect = capture_callback

    runner = QueueserverOptimizationRunner(
        optimization_problem=mock_optimization_problem,
        queueserver_client=mock_client,
    )
    future = runner.run(iterations=iterations, num_points=1, checkpoint_interval=checkpoint_interval)
    return runner, mock_client, future


def _fire_callback(runner, mock_client, iteration: int, exit_status: str = "success", reason: str = "") -> None:
    """Fire the on_stop callback with a matching start/stop document pair."""
    current_uid = runner._state.current_uid
    uid = f"fake-uid-{iteration}"
    start_doc = {"uid": uid, CORRELATION_UID_KEY: current_uid}
    stop_doc = {"uid": f"stop-{iteration}", "run_start": uid, "exit_status": exit_status, "reason": reason}
    mock_client._on_stop(start_doc, stop_doc)


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

    mock_client.start_listener.assert_called_once()

    future = runner.run(iterations=3, num_points=2)

    # Simulate 3 acquisition completions by invoking the captured callback
    uids = []
    for i in range(3):
        current_uid = runner._state.current_uid
        uid = f"fake-uid-{i}"
        uids.append(uid)
        start_doc = {"uid": uid, CORRELATION_UID_KEY: current_uid}
        stop_doc = {"uid": "other-fake-uid", "run_start": uid, "exit_status": "success"}
        mock_client._on_stop(start_doc, stop_doc)

    assert mock_client.submit_plan.call_count == 3
    mock_client.start_listener.assert_called_once()
    assert mock_optimization_problem.optimizer.suggest.call_count == 3
    assert mock_optimization_problem.optimizer.ingest.call_count == 3
    assert mock_optimization_problem.evaluation_function.call_count == 3

    # Verify the future resolved with the correct result
    assert future.done()
    result = future.result()
    assert isinstance(result, OptimizationResult)
    assert result.iterations_completed == 3
    assert result.num_points == 2
    assert result.uids == tuple(uids)


def test_runner_on_acquisition_complete_ignores_other_blop_runs(mock_optimization_problem):
    """Test _on_acquisition_complete ignores Blop documents for other correlation UIDs."""
    runner, mock_client, future = _make_runner_with_captured_callback(mock_optimization_problem)

    start_doc = {"uid": "fake-uid", CORRELATION_UID_KEY: "wrong-uid"}
    stop_doc = {"uid": "other-fake-uid", "run_start": "fake-uid"}

    mock_client._on_stop(start_doc, stop_doc)

    assert future.done()
    exc = future.exception()
    assert isinstance(exc, RuntimeError)
    assert "current_uid did not match start document" in str(exc)


def test_runner_private_method_calls_before_run(mock_optimization_problem):
    mock_client = MagicMock(spec=QueueserverClient)
    runner = QueueserverOptimizationRunner(
        optimization_problem=mock_optimization_problem,
        queueserver_client=mock_client,
    )

    with pytest.raises(RuntimeError, match="run()"):
        runner._build_plan([{}])

    # _on_acquisition_complete catches the error and stores it in the future;
    # since there is no active future yet, verify the runner handles this gracefully.
    runner._on_acquisition_complete({}, {})  # type: ignore[arg-type]
    # No future to check, but the runner must not crash the caller's thread.
    assert runner._current_future is None


def test_runner_error_in_evaluation_function_sets_future_exception(mock_optimization_problem):
    """Exception in evaluation_function stops the loop and stores the error in the future."""
    error = ValueError("bad data")
    mock_optimization_problem.evaluation_function.side_effect = error

    runner, mock_client, future = _make_runner_with_captured_callback(mock_optimization_problem)
    _fire_callback(runner, mock_client, 0)

    assert future.done()
    assert future.exception() is error


def test_runner_error_calls_register_failures_when_optimizer_supports_it():
    """register_failures is called on TrialFaultAware optimizers when an error occurs."""

    class FaultAwareOptimizer(Optimizer, TrialFaultAware): ...

    mock_optimization_problem = QueueserverOptimizationProblem(
        optimizer=MagicMock(spec=FaultAwareOptimizer),
        actuators=["motor1", "motor2"],
        sensors=["detector"],
        evaluation_function=MagicMock(side_effect=RuntimeError("boom")),
    )
    mock_optimization_problem.optimizer.suggest.return_value = [{"_id": 0, "motor1": 5.0, "motor2": 3.0}]

    runner, mock_client, future = _make_runner_with_captured_callback(mock_optimization_problem)
    _fire_callback(runner, mock_client, 0)

    assert future.exception() is not None
    mock_optimization_problem.optimizer.register_failures.assert_called_once()


def test_runner_register_failures_raises_original_error_preserved_in_future():
    """If register_failures() itself raises, the original acquisition error is still in the future."""

    class FaultAwareOptimizer(Optimizer, TrialFaultAware): ...

    acquisition_error = RuntimeError("evaluation failed")
    register_error = RuntimeError("register_failures exploded")

    mock_optimization_problem = QueueserverOptimizationProblem(
        optimizer=MagicMock(spec=FaultAwareOptimizer),
        actuators=["motor1"],
        sensors=["detector"],
        evaluation_function=MagicMock(side_effect=acquisition_error),
    )
    mock_optimization_problem.optimizer.suggest.return_value = [{"_id": 0, "motor1": 5.0}]
    mock_optimization_problem.optimizer.register_failures.side_effect = register_error

    runner, mock_client, future = _make_runner_with_captured_callback(mock_optimization_problem, iterations=1)

    # register_failures re-raises after logging, so it propagates out of the callback
    with pytest.raises(RuntimeError, match="register_failures exploded"):
        _fire_callback(runner, mock_client, 0)

    assert future.done()
    # The original acquisition error is what the caller sees
    assert future.exception() is acquisition_error
    mock_optimization_problem.optimizer.register_failures.assert_called_once()


def test_runner_error_does_not_call_register_failures_when_optimizer_lacks_support(mock_optimization_problem):
    """register_failures is NOT called on optimizers that don't implement TrialFaultAware."""
    mock_optimization_problem.evaluation_function.side_effect = RuntimeError("boom")
    assert not isinstance(mock_optimization_problem.optimizer, TrialFaultAware)

    runner, mock_client, future = _make_runner_with_captured_callback(mock_optimization_problem)
    _fire_callback(runner, mock_client, 0)

    assert future.exception() is not None


def test_runner_error_in_ingest_sets_future_exception(mock_optimization_problem):
    """Exception in optimizer.ingest stops the loop and stores the error in the future."""
    error = RuntimeError("ingest failed")
    mock_optimization_problem.optimizer.ingest.side_effect = error

    runner, mock_client, future = _make_runner_with_captured_callback(mock_optimization_problem)
    _fire_callback(runner, mock_client, 0)

    assert future.done()
    assert future.exception() is error


def test_runner_future_resolves_none_on_successful_run(mock_optimization_problem):
    """Future resolves to an OptimizationResult (not an exception) after a clean run."""
    runner, mock_client, future = _make_runner_with_captured_callback(mock_optimization_problem, iterations=1)
    _fire_callback(runner, mock_client, 0)

    assert future.done()
    assert future.exception() is None
    result = future.result()
    assert isinstance(result, OptimizationResult)
    assert result.iterations_completed == 1
    assert result.uids == ("fake-uid-0",)


@pytest.mark.parametrize("exit_status", ["fail", "abort"])
def test_runner_plan_failure_sets_future_exception(mock_optimization_problem, exit_status):
    """A failed/aborted plan stores a RuntimeError in the future."""
    runner, mock_client, future = _make_runner_with_captured_callback(mock_optimization_problem, iterations=3)
    _fire_callback(runner, mock_client, 0)  # one success
    _fire_callback(runner, mock_client, 1, exit_status=exit_status, reason="hardware fault")

    assert future.done()
    exc = future.exception()
    assert isinstance(exc, RuntimeError)
    assert exit_status in str(exc)
    assert "hardware fault" in str(exc)


@pytest.mark.parametrize("exit_status", ["fail", "abort"])
def test_runner_plan_failure_calls_register_failures_when_supported(exit_status):
    """register_failures is called on TrialFaultAware optimizers when a plan fails."""

    class FaultAwareOptimizer(Optimizer, TrialFaultAware): ...

    mock_optimization_problem = QueueserverOptimizationProblem(
        optimizer=MagicMock(spec=FaultAwareOptimizer),
        actuators=["motor1"],
        sensors=["detector"],
        evaluation_function=MagicMock(),
    )
    mock_optimization_problem.optimizer.suggest.return_value = [{"_id": 0, "motor1": 5.0}]

    runner, mock_client, future = _make_runner_with_captured_callback(mock_optimization_problem)
    _fire_callback(runner, mock_client, 0, exit_status=exit_status, reason="beam lost")

    assert future.done()
    assert isinstance(future.exception(), RuntimeError)
    mock_optimization_problem.optimizer.register_failures.assert_called_once()


@pytest.mark.parametrize("exit_status", ["fail", "abort"])
def test_runner_plan_failure_does_not_call_register_failures_when_unsupported(mock_optimization_problem, exit_status):
    """register_failures is NOT called on optimizers that don't implement TrialFaultAware."""
    assert not isinstance(mock_optimization_problem.optimizer, TrialFaultAware)

    runner, mock_client, future = _make_runner_with_captured_callback(mock_optimization_problem)
    _fire_callback(runner, mock_client, 0, exit_status=exit_status)

    assert future.done()
    assert isinstance(future.exception(), RuntimeError)


def test_runner_stop_races_final_callback_does_not_raise(mock_optimization_problem):
    """stop() called just after the last iteration completes does not raise InvalidStateError."""
    runner, mock_client, future = _make_runner_with_captured_callback(mock_optimization_problem, iterations=1)

    # Fire the final callback — this resolves the future
    _fire_callback(runner, mock_client, 0)
    assert future.done()

    # stop() should be safe to call even though the future is already resolved
    runner.stop()  # Must not raise


def test_runner_init_listener_error_reraises(mock_optimization_problem):
    """An exception from start_listener in __init__ is re-raised."""
    error = RuntimeError("connection refused")
    mock_client = MagicMock(spec=QueueserverClient)
    mock_client.start_listener.side_effect = error

    with pytest.raises(RuntimeError, match="connection refused"):
        QueueserverOptimizationRunner(
            optimization_problem=mock_optimization_problem,
            queueserver_client=mock_client,
        )


def test_runner_run_submit_error_fails_future_and_reraises(mock_optimization_problem):
    """An exception from submit_plan in run() fails the future and re-raises."""
    error = RuntimeError("connection refused")
    mock_client = MagicMock(spec=QueueserverClient)
    mock_client.submit_plan.side_effect = error

    runner = QueueserverOptimizationRunner(
        optimization_problem=mock_optimization_problem,
        queueserver_client=mock_client,
    )

    with pytest.raises(RuntimeError, match="connection refused"):
        runner.run(iterations=1, num_points=1)

    future = runner._current_future
    assert future is not None
    assert future.done()
    assert future.exception() is error


def test_runner_submit_suggestions_submit_error_fails_future_and_reraises(mock_optimization_problem):
    """An exception from submit_plan in submit_suggestions() fails the future and re-raises."""
    error = RuntimeError("connection refused")
    mock_client = MagicMock(spec=QueueserverClient)
    mock_client.submit_plan.side_effect = error

    runner = QueueserverOptimizationRunner(
        optimization_problem=mock_optimization_problem,
        queueserver_client=mock_client,
    )

    suggestions = [{"_id": 0, "motor1": 1.0}]
    with pytest.raises(RuntimeError, match="connection refused"):
        runner.submit_suggestions(suggestions)

    future = runner._current_future
    assert future is not None
    assert future.done()
    assert future.exception() is error


def test_runner_checkpoints(mock_optimization_problem):
    """A checkpoint is taken for each iteration."""

    mock_optimizer = MagicMock(spec=CheckpointableOptimizer)
    mock_optimizer.suggest.return_value = [
        {"_id": 0, "motor1": 5.0, "motor2": 3.0},
    ]
    mock_optimization_problem = QueueserverOptimizationProblem(
        optimizer=mock_optimizer,
        actuators=mock_optimization_problem.actuators,
        sensors=mock_optimization_problem.sensors,
        evaluation_function=mock_optimization_problem.evaluation_function,
    )
    runner, mock_client, future = _make_runner_with_captured_callback(mock_optimization_problem, checkpoint_interval=1)
    mock_optimizer.checkpoint.assert_not_called()
    _fire_callback(runner, mock_client, 0)
    mock_optimizer.checkpoint.assert_called_once()
    _fire_callback(runner, mock_client, 1)
    assert mock_optimizer.checkpoint.call_count == 2
    _fire_callback(runner, mock_client, 2)
    assert mock_optimizer.checkpoint.call_count == 3


def test_runner_skip_checkpoints(mock_optimization_problem):
    """No checkpoint is taken for each iteration because no interval configured."""

    mock_optimizer = MagicMock(spec=CheckpointableOptimizer)
    mock_optimizer.suggest.return_value = [
        {"_id": 0, "motor1": 5.0, "motor2": 3.0},
    ]
    mock_optimization_problem = QueueserverOptimizationProblem(
        optimizer=mock_optimizer,
        actuators=mock_optimization_problem.actuators,
        sensors=mock_optimization_problem.sensors,
        evaluation_function=mock_optimization_problem.evaluation_function,
    )
    runner, mock_client, future = _make_runner_with_captured_callback(mock_optimization_problem, checkpoint_interval=None)
    mock_optimizer.checkpoint.assert_not_called()
    _fire_callback(runner, mock_client, 0)
    mock_optimizer.checkpoint.assert_not_called()
    _fire_callback(runner, mock_client, 1)
    mock_optimizer.checkpoint.assert_not_called()
    _fire_callback(runner, mock_client, 2)
    mock_optimizer.checkpoint.assert_not_called()


def test_runner_raises_not_checkpointable(mock_optimization_problem):
    """No checkpoint is taken for each iteration because optimize does not support it."""
    runner, mock_client, future = _make_runner_with_captured_callback(mock_optimization_problem, checkpoint_interval=1)
    _fire_callback(runner, mock_client, 0)
    with pytest.raises(ValueError, match="optimizer is not checkpointable"):
        raise future.exception()
