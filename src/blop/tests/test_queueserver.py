import json
import logging
import sys
import threading
from concurrent.futures import Future
from contextlib import closing
from dataclasses import replace
from itertools import count
from unittest.mock import MagicMock

import pytest
from bluesky.run_engine import Dispatcher
from bluesky_queueserver_api import BPlan
from bluesky_queueserver_api.zmq import REManagerAPI
from event_model import DocumentNames

from blop.protocols import CanRegisterSuggestions, Optimizer, QueueserverOptimizationProblem, TrialFaultAware
from blop.queueserver import (
    CORRELATION_UID_KEY,
    ConsumerCallback,
    DocumentStreamEvaluator,
    OptimizationResult,
    QueueserverAcquisition,
    QueueserverClient,
    QueueserverOptimizationRunner,
)

from .conftest import CheckpointableOptimizer, background_call


@pytest.fixture(scope="function")
def mock_re_manager_api():
    manager = MagicMock(spec=REManagerAPI)
    manager.status.return_value = {"worker_environment_exists": True}
    item_numbers = count(1)
    manager.item_add.side_effect = lambda plan: {"success": True, "item": {"item_uid": f"item-{next(item_numbers)}"}}
    return manager


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


class RegisteringOptimizer(Optimizer, CanRegisterSuggestions): ...


class FaultAwareOptimizer(Optimizer, TrialFaultAware): ...


class FaultAwareCheckpointableOptimizer(CheckpointableOptimizer, TrialFaultAware): ...


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


def _dispatch_completion(dispatcher, correlation_uid, run_uid, *, exit_status="success", reason=""):
    start_doc = {"uid": run_uid, "time": 0.0}
    if correlation_uid is not None:
        start_doc[CORRELATION_UID_KEY] = correlation_uid
    stop_doc = {
        "uid": f"stop-{run_uid}",
        "run_start": run_uid,
        "time": 1.0,
        "exit_status": exit_status,
        "reason": reason,
    }
    dispatcher.process(DocumentNames.start, start_doc)
    dispatcher.process(DocumentNames.stop, stop_doc)


def test_document_stream_evaluator_accepts_early_completion():
    dispatcher = Dispatcher()
    suggestions = [{"_id": 7, "motor": 3.0}]
    outcomes = [{"_id": 7, "objective": 9.0}]
    evaluate = MagicMock(return_value=outcomes)
    uid = QueueserverAcquisition("target", "item-1", "acquire")

    with closing(DocumentStreamEvaluator(dispatcher, evaluate, timeout=0)) as adapter:
        _dispatch_completion(dispatcher, None, "non-blop", exit_status="fail")
        _dispatch_completion(dispatcher, "other", "unrelated", exit_status="abort")
        _dispatch_completion(dispatcher, uid.correlation_uid, "actual-run-uid")
        _dispatch_completion(dispatcher, uid.correlation_uid, "later-run", exit_status="fail")

        assert adapter(uid, suggestions) is outcomes
        evaluate.assert_called_once_with("actual-run-uid", suggestions)
        with pytest.raises(TimeoutError, match="Timed out waiting for acquisition 'target'"):
            adapter(uid, suggestions)


def test_document_stream_evaluator_requires_matching_completion():
    dispatcher = Dispatcher()
    evaluate = MagicMock(return_value=[{"_id": 1, "objective": 2.0}])
    uid = QueueserverAcquisition("target", None, "acquire")

    with closing(DocumentStreamEvaluator(dispatcher, evaluate, timeout=0)) as adapter:
        _dispatch_completion(dispatcher, None, "non-blop")
        _dispatch_completion(dispatcher, "other", "unrelated", exit_status="fail")
        with pytest.raises(TimeoutError, match="Timed out waiting for acquisition 'target'"):
            adapter(uid, [])
        evaluate.assert_not_called()

        _dispatch_completion(dispatcher, uid.correlation_uid, "matching-run")
        assert adapter(uid, []) == [{"_id": 1, "objective": 2.0}]


@pytest.mark.parametrize("exit_status, reason", [("fail", "hardware fault"), ("abort", "")])
def test_document_stream_evaluator_rejects_unsuccessful_stop(exit_status, reason):
    dispatcher = Dispatcher()
    evaluate = MagicMock()
    uid = QueueserverAcquisition("target", None, "acquire")

    with closing(DocumentStreamEvaluator(dispatcher, evaluate, timeout=0)) as adapter:
        _dispatch_completion(dispatcher, uid.correlation_uid, "failed-run", exit_status=exit_status, reason=reason)
        with pytest.raises(RuntimeError) as exc_info:
            adapter(uid, [{"_id": 1}])
        assert str(exc_info.value) == (
            f"Acquisition run 'failed-run' ended with status {exit_status!r}: {reason or '(no reason given)'}"
        )
        evaluate.assert_not_called()


def test_document_stream_evaluator_preserves_wrapped_error():
    dispatcher = Dispatcher()
    error = ValueError("Invalid acquired data")
    evaluate = MagicMock(side_effect=error)
    uid = QueueserverAcquisition("target", None, "acquire")

    with closing(DocumentStreamEvaluator(dispatcher, evaluate, timeout=0)) as adapter:
        _dispatch_completion(dispatcher, uid.correlation_uid, "completed-run")
        with pytest.raises(ValueError) as exc_info:
            adapter(uid, [{"_id": 1}])
        assert exc_info.value is error


def test_document_stream_evaluator_close_releases_waiter():
    dispatcher = Dispatcher()
    evaluate = MagicMock()
    adapter = DocumentStreamEvaluator(dispatcher, evaluate)
    uid = QueueserverAcquisition("target", None, "acquire")
    entered = threading.Event()

    def wait_for_acquisition():
        entered.set()
        return adapter(uid, [{"_id": 1}])

    with background_call(wait_for_acquisition) as future:
        try:
            assert entered.wait(timeout=5)
            assert not future.done()
            with background_call(adapter.close) as close_future:
                close_future.result(timeout=5)
            with pytest.raises(RuntimeError, match="Document stream evaluator is closed"):
                future.result(timeout=5)
            evaluate.assert_not_called()
        finally:
            adapter.close()


def test_document_stream_evaluator_close_preserves_other_subscribers():
    dispatcher = Dispatcher()
    documents = []
    dispatcher.subscribe(lambda name, doc: documents.append((name, doc)))
    evaluate = MagicMock()
    adapter = DocumentStreamEvaluator(dispatcher, evaluate, timeout=0)
    uid = QueueserverAcquisition("target", None, "acquire")
    _dispatch_completion(dispatcher, uid.correlation_uid, "buffered-run")
    documents.clear()

    adapter.close()
    adapter.close()
    _dispatch_completion(dispatcher, uid.correlation_uid, "late-run")

    assert [(name, doc["uid"]) for name, doc in documents] == [("start", "late-run"), ("stop", "stop-late-run")]
    with pytest.raises(RuntimeError, match="Document stream evaluator is closed"):
        adapter(uid, [])
    evaluate.assert_not_called()


def test_document_stream_evaluator_close_does_not_interrupt_wrapped_evaluation():
    dispatcher = Dispatcher()
    entered = threading.Event()
    release = threading.Event()
    outcomes = [{"_id": 1, "objective": 4.0}]

    def evaluate(run_uid, suggestions):
        entered.set()
        assert release.wait(timeout=5)
        return outcomes

    adapter = DocumentStreamEvaluator(dispatcher, evaluate)
    uid = QueueserverAcquisition("target", None, "acquire")
    _dispatch_completion(dispatcher, uid.correlation_uid, "completed-run")

    with background_call(adapter, uid, [{"_id": 1}]) as future:
        try:
            assert entered.wait(timeout=5)
            with background_call(adapter.close) as close_future:
                close_future.result(timeout=5)
            assert not future.done()
            release.set()
            assert future.result(timeout=5) is outcomes
        finally:
            release.set()
            adapter.close()


def test_queueserver_client_check_environment_raises_when_not_ready(mock_re_manager_api):
    mock_re_manager_api.status.return_value = {"worker_environment_exists": False}
    client = QueueserverClient(mock_re_manager_api)

    with pytest.raises(RuntimeError, match="queueserver environment is not open"):
        client.check_environment()


@pytest.mark.parametrize("autostart", [True, False])
def test_queueserver_client_submit_plan_returns_item_uid(mock_re_manager_api, autostart):
    client = QueueserverClient(mock_re_manager_api, autostart=autostart)
    plan = BPlan("count", ["detector"])

    assert client.submit_plan(plan) == "item-1"
    mock_re_manager_api.queue_autostart.assert_called_once_with(autostart)
    mock_re_manager_api.item_add.assert_called_once_with(plan)


def test_queueserver_client_submit_plan_propagates_item_add_rejection(mock_re_manager_api):
    response = {
        "success": False,
        "msg": "Failed to add an item: plan rejected by RE Manager",
        "qsize": None,
        "item": None,
    }
    error = REManagerAPI.RequestFailedError(request={"method": "queue_item_add"}, response=response)
    mock_re_manager_api.item_add.side_effect = error
    client = QueueserverClient(mock_re_manager_api)
    plan = BPlan("count", ["det_forbidden"], num=1)

    with pytest.raises(REManagerAPI.RequestFailedError) as exc_info:
        client.submit_plan(plan)

    assert exc_info.value is error
    assert exc_info.value.response is response
    mock_re_manager_api.item_add.assert_called_once_with(plan)


def test_queueserver_client_requires_authoritative_item_uid(mock_re_manager_api):
    mock_re_manager_api.item_add.side_effect = None
    mock_re_manager_api.item_add.return_value = {"success": True, "item": {}}
    client = QueueserverClient(mock_re_manager_api)

    with pytest.raises(KeyError) as exc_info:
        client.submit_plan(BPlan("count", ["detector"]))
    assert exc_info.value.args == ("item_uid",)


@pytest.mark.parametrize("manual", [False, True])
def test_runner_run_validates_environment(mock_optimization_problem, mock_re_manager_api, manual):
    mock_re_manager_api.status.return_value = {"worker_environment_exists": False}
    runner = QueueserverOptimizationRunner(mock_optimization_problem, QueueserverClient(mock_re_manager_api))

    with pytest.raises(RuntimeError, match="queueserver environment is not open"):
        if manual:
            runner.submit_suggestions([{"_id": 1, "motor1": 2.0}])
        else:
            runner.run()
    mock_re_manager_api.item_add.assert_not_called()
    mock_optimization_problem.optimizer.suggest.assert_not_called()


def _assert_runner_rejects_work(runner):
    with background_call(runner.run) as call:
        with pytest.raises(RuntimeError, match="Optimization loop is already running"):
            call.result(timeout=5)
    with background_call(runner.submit_suggestions, [{"_id": 99, "motor1": 1.0}]) as call:
        with pytest.raises(RuntimeError, match="Optimization loop is already running"):
            call.result(timeout=5)


@pytest.mark.filterwarnings("error::pytest.PytestUnhandledThreadExceptionWarning")
@pytest.mark.parametrize(
    "interruption_point, error_type",
    [
        ("before_start", RuntimeError),
        ("before_start", KeyboardInterrupt),
        ("before_work", KeyboardInterrupt),
        ("during_evaluation", KeyboardInterrupt),
        ("after_completion", KeyboardInterrupt),
    ],
)
def test_runner_startup_error_preserves_settlement_ownership(
    mock_optimization_problem, mock_re_manager_api, interruption_point, error_type
):
    worker_entered, release_worker = threading.Event(), threading.Event()
    evaluation_entered, release_evaluation = threading.Event(), threading.Event()
    startup_error = error_type("interrupted startup")
    acquisitions = []
    worker = None
    start_returned = False
    start_code = threading.Thread.start.__code__
    run_code = threading.Thread.run.__code__

    def evaluate(uid, suggestions):
        acquisitions.append(uid)
        if interruption_point == "during_evaluation" and len(acquisitions) == 1:
            evaluation_entered.set()
            assert release_evaluation.wait(timeout=5)
        return [{"_id": 0, "objective": 4.0}]

    # Inject at the real thread boundary; neither the worker nor Thread is mocked.
    def trace_worker(frame, event, arg):
        if frame.f_code is run_code and event == "call" and frame.f_locals["self"] is worker:
            worker_entered.set()
            if interruption_point == "before_work":
                assert release_worker.wait(timeout=5)
        return None

    def trace_start(frame, event, arg):
        nonlocal worker, start_returned
        if frame.f_code is not start_code:
            return None
        if event == "call":
            worker = frame.f_locals["self"]
            if interruption_point == "before_start":
                raise startup_error
        elif event == "return":
            start_returned = True
            if interruption_point == "before_work":
                assert worker_entered.wait(timeout=5)
            elif interruption_point == "during_evaluation":
                assert evaluation_entered.wait(timeout=5)
            else:
                worker.join(timeout=5)
                assert not worker.is_alive()
            raise startup_error
        return trace_start

    problem = replace(mock_optimization_problem, evaluation_function=evaluate)
    runner = QueueserverOptimizationRunner(problem, QueueserverClient(mock_re_manager_api))
    previous_trace, previous_thread_trace = sys.gettrace(), threading.gettrace()
    try:
        threading.settrace(trace_worker)
        sys.settrace(trace_start)
        try:
            with pytest.raises(error_type) as exc_info:
                runner.run()
            assert exc_info.value is startup_error
        finally:
            sys.settrace(previous_trace)
            threading.settrace(previous_thread_trace)

        if interruption_point == "during_evaluation":
            _assert_runner_rejects_work(runner)
            release_evaluation.set()
            worker.join(timeout=5)
            assert not worker.is_alive()

        result = runner.run().result(timeout=5)
        assert result.iterations_completed == 1
        assert result.uids == (acquisitions[-1],)
    finally:
        sys.settrace(previous_trace)
        threading.settrace(previous_thread_trace)
        release_worker.set()
        release_evaluation.set()
        if start_returned:
            worker.join(timeout=5)
            assert not worker.is_alive()

    expected_acquisitions = 2 if interruption_point in {"during_evaluation", "after_completion"} else 1
    assert mock_re_manager_api.item_add.call_count == expected_acquisitions
    assert problem.optimizer.ingest.call_count == expected_acquisitions
    assert len(acquisitions) == expected_acquisitions


def test_runner_run_full_cycle(mock_optimization_problem, mock_re_manager_api):
    entered = [threading.Event() for _ in range(3)]
    release = [threading.Event() for _ in range(3)]
    ingested = [threading.Event() for _ in range(3)]
    batches, tokens, ingested_batches = [], [], []
    objective_by_id, outcomes_by_acquisition = {}, {}

    def suggest(num_points):
        assert num_points == 2
        index = len(batches)
        base = max(objective_by_id.values()) + 1 if objective_by_id else 1.0
        batch = [
            {"_id": index * 2, "motor1": base, "motor2": 3.0},
            {"_id": index * 2 + 1, "motor1": base + 1, "motor2": 4.0},
        ]
        batches.append(batch)
        return batch

    def evaluate(uid, suggestions):
        index = len(tokens)
        tokens.append(uid)
        outcomes = [{"_id": point["_id"], "objective": 2 * point["motor1"]} for point in reversed(suggestions)]
        outcomes_by_acquisition[uid] = outcomes
        entered[index].set()
        assert release[index].wait(timeout=5)
        return outcomes

    def ingest(outcomes):
        objective_by_id.update({point["_id"]: point["objective"] for point in outcomes})
        index = len(ingested_batches)
        ingested_batches.append(outcomes)
        ingested[index].set()

    mock_optimization_problem.optimizer.suggest.side_effect = suggest
    mock_optimization_problem.optimizer.ingest.side_effect = ingest
    problem = replace(
        mock_optimization_problem,
        acquisition_plan="my_acquire",
        acquisition_plan_kwargs={"exposure_time": 0.5, "num_frames": 10},
        evaluation_function=evaluate,
    )
    runner = QueueserverOptimizationRunner(problem, QueueserverClient(mock_re_manager_api))
    expected_states = [
        {0: 2.0, 1: 4.0},
        {0: 2.0, 1: 4.0, 2: 10.0, 3: 12.0},
        {0: 2.0, 1: 4.0, 2: 10.0, 3: 12.0, 4: 26.0, 5: 28.0},
    ]
    future = None
    with background_call(runner.run, iterations=3, num_points=2) as call:
        try:
            future = call.result(timeout=5)
            for index in range(3):
                assert entered[index].wait(timeout=5)
                assert not future.done()
                plan = mock_re_manager_api.item_add.call_args_list[index].args[0]
                payload = json.loads(json.dumps(plan.to_dict()))
                token = tokens[index]
                assert token == QueueserverAcquisition(
                    payload["kwargs"]["md"][CORRELATION_UID_KEY], f"item-{index + 1}", "my_acquire"
                )
                assert payload["name"] == "my_acquire"
                assert payload["args"] == [batches[index], ["motor1", "motor2"], ["detector"]]
                assert payload["kwargs"] == {
                    "md": {CORRELATION_UID_KEY: token.correlation_uid, "blop_suggestions": batches[index]},
                    "exposure_time": 0.5,
                    "num_frames": 10,
                }
                release[index].set()
                assert ingested[index].wait(timeout=5)
                assert objective_by_id == expected_states[index]
            result = future.result(timeout=5)
        finally:
            for gate in release:
                gate.set()
            if future is not None:
                future.exception(timeout=5)

    assert result == OptimizationResult(iterations_completed=3, num_points=2, uids=tuple(tokens))
    assert len({token.correlation_uid for token in tokens}) == 3
    assert mock_re_manager_api.item_add.call_count == 3
    assert ingested_batches == [outcomes_by_acquisition[token] for token in result.uids]


@pytest.mark.parametrize("evaluation_fails", [False, True])
def test_runner_stop_waits_for_evaluation(mock_optimization_problem, mock_re_manager_api, evaluation_fails):
    entered, release = threading.Event(), threading.Event()
    tokens = []
    error = ValueError("evaluation failed after stop")
    outcomes = [{"_id": 0, "objective": 8.0}]

    def evaluate(uid, suggestions):
        tokens.append(uid)
        entered.set()
        assert release.wait(timeout=5)
        if evaluation_fails:
            raise error
        return outcomes

    problem = replace(mock_optimization_problem, evaluation_function=evaluate)
    runner = QueueserverOptimizationRunner(problem, QueueserverClient(mock_re_manager_api))
    runner.stop()
    future = None
    with background_call(runner.run, iterations=3) as call:
        try:
            future = call.result(timeout=5)
            assert entered.wait(timeout=5)
            with background_call(runner.stop) as stop_call:
                stop_call.result(timeout=5)
            assert not future.done()
            assert future.cancel() is False
            _assert_runner_rejects_work(runner)
            release.set()
            if evaluation_fails:
                with pytest.raises(ValueError) as exc_info:
                    future.result(timeout=5)
                assert exc_info.value is error
                problem.optimizer.ingest.assert_not_called()
            else:
                result = future.result(timeout=5)
                assert result == OptimizationResult(iterations_completed=1, num_points=1, uids=tuple(tokens))
                problem.optimizer.ingest.assert_called_once_with(outcomes)
                runner.stop()
                assert future.result(timeout=5) is result
        finally:
            release.set()
            if future is not None:
                future.exception(timeout=5)
    assert mock_re_manager_api.item_add.call_count == 1


@pytest.mark.parametrize("notification_fails", [False, True])
def test_runner_stop_during_suggest_prevents_submission(mock_optimization_problem, mock_re_manager_api, notification_fails):
    entered, release = threading.Event(), threading.Event()
    first = [{"_id": 0, "motor1": 1.0}]
    second = [{"_id": 1, "motor1": 2.0}]
    optimizer = MagicMock(spec=FaultAwareOptimizer)
    tokens = []
    notification_error = RuntimeError("failure notification failed")

    def suggest(num_points):
        if optimizer.suggest.call_count == 1:
            return first
        entered.set()
        assert release.wait(timeout=5)
        return second

    def evaluate(uid, suggestions):
        tokens.append(uid)
        return [{"_id": 0, "objective": 4.0}]

    optimizer.suggest.side_effect = suggest
    if notification_fails:
        optimizer.register_failures.side_effect = notification_error
    problem = replace(mock_optimization_problem, optimizer=optimizer, evaluation_function=evaluate)
    runner = QueueserverOptimizationRunner(problem, QueueserverClient(mock_re_manager_api))
    future = None
    with background_call(runner.run, iterations=3) as call:
        try:
            future = call.result(timeout=5)
            assert entered.wait(timeout=5)
            with background_call(runner.stop) as stop_call:
                stop_call.result(timeout=5)
            assert not future.done()
            release.set()
            if notification_fails:
                with pytest.raises(RuntimeError) as exc_info:
                    future.result(timeout=5)
                assert exc_info.value is notification_error
            else:
                assert future.result(timeout=5) == OptimizationResult(1, 1, tuple(tokens))
        finally:
            release.set()
            if future is not None:
                future.exception(timeout=5)
    mock_re_manager_api.item_add.assert_called_once()
    optimizer.ingest.assert_called_once_with([{"_id": 0, "objective": 4.0}])
    optimizer.register_failures.assert_called_once_with(second)


def test_runner_run_twice_fails(mock_optimization_problem, mock_re_manager_api):
    entered, release = threading.Event(), threading.Event()
    suggestions = [{"_id": 0, "motor1": 2.0}]

    def suggest(num_points):
        entered.set()
        assert release.wait(timeout=5)
        return suggestions

    mock_optimization_problem.optimizer.suggest.side_effect = suggest
    runner = QueueserverOptimizationRunner(mock_optimization_problem, QueueserverClient(mock_re_manager_api))
    future = None
    with background_call(runner.run) as call:
        try:
            future = call.result(timeout=5)
            assert entered.wait(timeout=5)
            mock_re_manager_api.item_add.assert_not_called()
            _assert_runner_rejects_work(runner)
            release.set()
            assert future.result(timeout=5).iterations_completed == 1
        finally:
            release.set()
            if future is not None:
                future.exception(timeout=5)
    mock_re_manager_api.item_add.assert_called_once()


def test_runner_submit_suggestions_twice_fails(mock_optimization_problem, mock_re_manager_api):
    entered, release = threading.Event(), threading.Event()
    optimizer = MagicMock(spec=RegisteringOptimizer)
    suggestions = [{"motor1": 2.0}]
    registered = [{"_id": 7, "motor1": 2.0}]
    optimizer.register_suggestions.return_value = registered

    def evaluate(uid, points):
        entered.set()
        assert release.wait(timeout=5)
        return [{"_id": points[0]["_id"], "objective": 4.0}]

    problem = replace(mock_optimization_problem, optimizer=optimizer, evaluation_function=evaluate)
    runner = QueueserverOptimizationRunner(problem, QueueserverClient(mock_re_manager_api))
    future = None
    with background_call(runner.submit_suggestions, suggestions) as call:
        try:
            future = call.result(timeout=5)
            assert entered.wait(timeout=5)
            _assert_runner_rejects_work(runner)
            release.set()
            assert future.result(timeout=5).iterations_completed == 1
        finally:
            release.set()
            if future is not None:
                future.exception(timeout=5)
    optimizer.register_suggestions.assert_called_once_with(suggestions)
    optimizer.suggest.assert_not_called()
    optimizer.ingest.assert_called_once_with([{"_id": 7, "objective": 4.0}])


def test_runner_submit_suggestions_register_fails(mock_optimization_problem, mock_re_manager_api):
    runner = QueueserverOptimizationRunner(mock_optimization_problem, QueueserverClient(mock_re_manager_api))
    with pytest.raises(ValueError, match="'_id'"):
        runner.submit_suggestions([{"motor1": 2.0}])
    mock_re_manager_api.item_add.assert_not_called()
    mock_optimization_problem.optimizer.suggest.assert_not_called()


def test_runner_submit_suggestions_preserves_existing_ids_without_registration(
    mock_optimization_problem, mock_re_manager_api
):
    suggestions = [{"_id": "manual-point", "motor1": 2.0, "motor2": 3.0}]
    outcomes = [{"_id": "manual-point", "objective": 4.0}]
    mock_optimization_problem.evaluation_function.return_value = outcomes
    runner = QueueserverOptimizationRunner(mock_optimization_problem, QueueserverClient(mock_re_manager_api))

    result = runner.submit_suggestions(suggestions).result(timeout=5)

    assert result.iterations_completed == 1
    assert mock_re_manager_api.item_add.call_args.args[0].args[0] == suggestions
    mock_optimization_problem.evaluation_function.assert_called_once_with(result.uids[0], suggestions)
    mock_optimization_problem.optimizer.ingest.assert_called_once_with(outcomes)
    mock_optimization_problem.optimizer.suggest.assert_not_called()


def test_runner_empty_manual_batch_is_submitted(mock_optimization_problem, mock_re_manager_api):
    mock_optimization_problem.evaluation_function.return_value = []
    runner = QueueserverOptimizationRunner(mock_optimization_problem, QueueserverClient(mock_re_manager_api))

    result = runner.submit_suggestions([]).result(timeout=5)

    assert result.iterations_completed == 1 and result.num_points == 0
    assert result.uids[0].item_uid == "item-1"
    assert mock_re_manager_api.item_add.call_args.args[0].args[0] == []
    mock_optimization_problem.optimizer.suggest.assert_not_called()
    mock_optimization_problem.optimizer.ingest.assert_called_once_with([])


@pytest.mark.parametrize("error_type", [ValueError, SystemExit])
def test_runner_error_in_evaluation_function_sets_future_exception(
    mock_optimization_problem, mock_re_manager_api, error_type
):
    error = error_type("evaluation failed")
    mock_optimization_problem.evaluation_function.side_effect = error
    runner = QueueserverOptimizationRunner(mock_optimization_problem, QueueserverClient(mock_re_manager_api))

    future = runner.run(iterations=3)
    with pytest.raises(error_type) as exc_info:
        future.result(timeout=5)

    assert exc_info.value is error
    mock_optimization_problem.optimizer.ingest.assert_not_called()
    mock_re_manager_api.item_add.assert_called_once()


def test_runner_error_in_ingest_sets_future_exception(mock_optimization_problem, mock_re_manager_api):
    error = RuntimeError("ingest failed")
    mock_optimization_problem.optimizer.ingest.side_effect = error
    runner = QueueserverOptimizationRunner(mock_optimization_problem, QueueserverClient(mock_re_manager_api))

    future = runner.run(iterations=3)
    with pytest.raises(RuntimeError) as exc_info:
        future.result(timeout=5)

    assert exc_info.value is error
    mock_optimization_problem.optimizer.ingest.assert_called_once_with([{"_id": 0, "objective": 1.0}])
    mock_re_manager_api.item_add.assert_called_once()


@pytest.mark.parametrize("manual", [False, True])
def test_runner_submission_error_fails_returned_future(mock_optimization_problem, mock_re_manager_api, manual):
    error = RuntimeError("connection refused")
    mock_re_manager_api.item_add.side_effect = error
    runner = QueueserverOptimizationRunner(mock_optimization_problem, QueueserverClient(mock_re_manager_api))

    future = runner.submit_suggestions([{"_id": 1, "motor1": 2.0}]) if manual else runner.run(iterations=3)
    with pytest.raises(RuntimeError) as exc_info:
        future.result(timeout=5)

    assert exc_info.value is error
    mock_optimization_problem.evaluation_function.assert_not_called()
    mock_optimization_problem.optimizer.ingest.assert_not_called()
    mock_re_manager_api.item_add.assert_called_once()


@pytest.mark.parametrize(
    "failure_phase, notification_error_type",
    [("submission", None), ("evaluation", RuntimeError), ("evaluation", SystemExit)],
)
def test_runner_failure_notification_precedes_future_completion(
    mock_optimization_problem, mock_re_manager_api, failure_phase, notification_error_type, caplog
):
    entered, release = threading.Event(), threading.Event()
    primary_error = RuntimeError(f"second {failure_phase} failed")
    secondary_error = notification_error_type("failure notification failed") if notification_error_type else None
    first = [{"_id": 0, "motor1": 2.0}]
    second = [{"_id": 1, "motor1": 3.0}]
    first_outcomes = [{"_id": 0, "objective": 4.0}]
    optimizer = MagicMock(spec=FaultAwareOptimizer)
    optimizer.suggest.side_effect = [first, second]
    completed, failed = [], []
    optimizer.ingest.side_effect = completed.extend
    if failure_phase == "submission":
        mock_re_manager_api.item_add.side_effect = [{"success": True, "item": {"item_uid": "item-1"}}, primary_error]
        mock_optimization_problem.evaluation_function.return_value = first_outcomes
    else:
        mock_optimization_problem.evaluation_function.side_effect = [first_outcomes, primary_error]

    def register_failures(points):
        failed.append(points)
        entered.set()
        assert release.wait(timeout=5)
        if secondary_error is not None:
            raise secondary_error

    optimizer.register_failures.side_effect = register_failures
    problem = replace(mock_optimization_problem, optimizer=optimizer)
    runner = QueueserverOptimizationRunner(problem, QueueserverClient(mock_re_manager_api))
    future = None
    with background_call(runner.run, iterations=3) as call:
        try:
            future = call.result(timeout=5)
            assert entered.wait(timeout=5)
            assert not future.done()
            assert completed == first_outcomes
            assert failed == [second]
            release.set()
            with pytest.raises(RuntimeError) as exc_info:
                future.result(timeout=5)
            assert exc_info.value is primary_error
        finally:
            release.set()
            if future is not None:
                future.exception(timeout=5)
    optimizer.register_failures.assert_called_once_with(second)
    assert mock_re_manager_api.item_add.call_count == 2
    if secondary_error is not None:
        errors = [record for record in caplog.records if record.exc_info and record.exc_info[1] is secondary_error]
        assert len(errors) == 1
        assert errors[0].levelno == logging.ERROR


def test_runner_suggestion_failure_keeps_completed_batch_successful(mock_optimization_problem, mock_re_manager_api):
    error = RuntimeError("next suggestion failed")
    optimizer = MagicMock(spec=FaultAwareOptimizer)
    optimizer.suggest.side_effect = [[{"_id": 0, "motor1": 2.0}], error]
    problem = replace(mock_optimization_problem, optimizer=optimizer)
    runner = QueueserverOptimizationRunner(problem, QueueserverClient(mock_re_manager_api))

    future = runner.run(iterations=3)
    with pytest.raises(RuntimeError) as exc_info:
        future.result(timeout=5)

    assert exc_info.value is error
    optimizer.ingest.assert_called_once_with([{"_id": 0, "objective": 1.0}])
    optimizer.register_failures.assert_not_called()
    mock_re_manager_api.item_add.assert_called_once()


@pytest.mark.parametrize("operation", ["suggest", "register"])
def test_runner_initial_optimizer_error_fails_returned_future(mock_optimization_problem, mock_re_manager_api, operation):
    error = RuntimeError(f"{operation} failed")
    optimizer = MagicMock(spec=RegisteringOptimizer)
    if operation == "suggest":
        optimizer.suggest.side_effect = error
    else:
        optimizer.register_suggestions.side_effect = error
    problem = replace(mock_optimization_problem, optimizer=optimizer)
    runner = QueueserverOptimizationRunner(problem, QueueserverClient(mock_re_manager_api))

    future = runner.run() if operation == "suggest" else runner.submit_suggestions([{"motor1": 2.0}])
    with pytest.raises(RuntimeError) as exc_info:
        future.result(timeout=5)

    assert exc_info.value is error
    mock_re_manager_api.item_add.assert_not_called()
    optimizer.ingest.assert_not_called()


def test_runner_duplicate_metadata_keeps_native_error(mock_optimization_problem, mock_re_manager_api):
    problem = replace(mock_optimization_problem, acquisition_plan_kwargs={"md": {"custom": "metadata"}})
    runner = QueueserverOptimizationRunner(problem, QueueserverClient(mock_re_manager_api))

    with pytest.raises(TypeError, match="multiple values.*md"):
        runner.run().result(timeout=5)
    mock_re_manager_api.item_add.assert_not_called()
    problem.evaluation_function.assert_not_called()


def test_runner_can_restart_from_done_callback(mock_optimization_problem, mock_re_manager_api):
    entered, release, callback_finished = threading.Event(), threading.Event(), threading.Event()
    callback_result = Future()
    tokens = []
    optimizer = mock_optimization_problem.optimizer
    optimizer.suggest.side_effect = [[{"_id": 0, "motor1": 2.0}], [{"_id": 1, "motor1": 3.0}]]

    def evaluate(uid, suggestions):
        tokens.append(uid)
        if len(tokens) == 1:
            entered.set()
            assert release.wait(timeout=5)
        return [{"_id": point["_id"], "objective": point["motor1"] ** 2} for point in suggestions]

    problem = replace(mock_optimization_problem, evaluation_function=evaluate)
    runner = QueueserverOptimizationRunner(problem, QueueserverClient(mock_re_manager_api))

    def restart(finished):
        try:
            callback_result.set_result(runner.run())
        except BaseException as error:
            callback_result.set_exception(error)
        finally:
            callback_finished.set()

    first_future = None
    callback_registered = False
    with background_call(runner.run) as call:
        try:
            first_future = call.result(timeout=5)
            assert entered.wait(timeout=5)
            first_future.add_done_callback(restart)
            callback_registered = True
            release.set()
            first_result = first_future.result(timeout=5)
            assert callback_finished.wait(timeout=5)
            second_future = callback_result.result(timeout=5)
            second_result = second_future.result(timeout=5)
        finally:
            release.set()
            if first_future is not None:
                first_future.exception(timeout=5)
            if callback_registered:
                assert callback_finished.wait(timeout=5)
                callback_result.result(timeout=5).exception(timeout=5)

    assert first_future is not second_future
    assert first_result == OptimizationResult(1, 1, (tokens[0],))
    assert second_result == OptimizationResult(1, 1, (tokens[1],))
    assert tokens[0] != tokens[1]
    assert first_future.result(timeout=5) is first_result
    assert [call.args[0] for call in optimizer.ingest.call_args_list] == [
        [{"_id": 0, "objective": 4.0}],
        [{"_id": 1, "objective": 9.0}],
    ]


def test_runner_can_restart_after_error(mock_optimization_problem, mock_re_manager_api):
    error = RuntimeError("first evaluation failed")
    tokens = []
    mock_optimization_problem.optimizer.suggest.side_effect = [
        [{"_id": 0, "motor1": 2.0}],
        [{"_id": 1, "motor1": 3.0}],
    ]

    def evaluate(uid, suggestions):
        tokens.append(uid)
        if len(tokens) == 1:
            raise error
        return [{"_id": 1, "objective": 9.0}]

    problem = replace(mock_optimization_problem, evaluation_function=evaluate)
    runner = QueueserverOptimizationRunner(problem, QueueserverClient(mock_re_manager_api))
    first_future = runner.run(iterations=3)
    with pytest.raises(RuntimeError) as exc_info:
        first_future.result(timeout=5)
    assert exc_info.value is error

    second_future = runner.run()
    assert second_future.result(timeout=5) == OptimizationResult(1, 1, (tokens[1],))
    assert first_future is not second_future
    assert tokens[0] != tokens[1]
    assert first_future.exception(timeout=5) is error
    problem.optimizer.ingest.assert_called_once_with([{"_id": 1, "objective": 9.0}])


def test_runner_checkpoints(mock_optimization_problem, mock_re_manager_api):
    entered = [threading.Event(), threading.Event()]
    release = [threading.Event(), threading.Event()]
    optimizer = MagicMock(spec=CheckpointableOptimizer)
    optimizer.suggest.side_effect = [[{"_id": index, "motor1": float(index + 1)}] for index in range(4)]
    objective_by_id, snapshots, tokens = {}, [], []

    def evaluate(uid, suggestions):
        tokens.append(uid)
        return [{"_id": point["_id"], "objective": point["motor1"] ** 2} for point in suggestions]

    def ingest(outcomes):
        objective_by_id.update({point["_id"]: point["objective"] for point in outcomes})

    def checkpoint():
        index = len(snapshots)
        snapshots.append(dict(objective_by_id))
        entered[index].set()
        assert release[index].wait(timeout=5)

    optimizer.ingest.side_effect = ingest
    optimizer.checkpoint.side_effect = checkpoint
    problem = replace(mock_optimization_problem, optimizer=optimizer, evaluation_function=evaluate)
    runner = QueueserverOptimizationRunner(problem, QueueserverClient(mock_re_manager_api))
    expected = [{0: 1.0, 1: 4.0}, {0: 1.0, 1: 4.0, 2: 9.0, 3: 16.0}]
    future = None
    with background_call(runner.run, iterations=4, checkpoint_interval=2) as call:
        try:
            future = call.result(timeout=5)
            for index in range(2):
                assert entered[index].wait(timeout=5)
                assert not future.done()
                assert mock_re_manager_api.item_add.call_count == 2 * (index + 1)
                assert snapshots == expected[: index + 1]
                _assert_runner_rejects_work(runner)
                release[index].set()
            assert future.result(timeout=5) == OptimizationResult(4, 1, tuple(tokens))
        finally:
            for gate in release:
                gate.set()
            if future is not None:
                future.exception(timeout=5)
    assert snapshots == expected


def test_runner_skip_checkpoints(mock_optimization_problem, mock_re_manager_api):
    optimizer = MagicMock(spec=CheckpointableOptimizer)
    optimizer.suggest.side_effect = [[{"_id": index, "motor1": float(index)}] for index in range(3)]
    completed = []
    optimizer.ingest.side_effect = completed.extend
    mock_optimization_problem.evaluation_function.side_effect = [
        [{"_id": index, "objective": float(index)}] for index in range(3)
    ]
    problem = replace(mock_optimization_problem, optimizer=optimizer)
    runner = QueueserverOptimizationRunner(problem, QueueserverClient(mock_re_manager_api))

    assert runner.run(iterations=3, checkpoint_interval=None).result(timeout=5).iterations_completed == 3
    assert completed == [{"_id": 0, "objective": 0.0}, {"_id": 1, "objective": 1.0}, {"_id": 2, "objective": 2.0}]
    optimizer.checkpoint.assert_not_called()


def test_runner_raises_not_checkpointable(mock_optimization_problem, mock_re_manager_api):
    runner = QueueserverOptimizationRunner(mock_optimization_problem, QueueserverClient(mock_re_manager_api))

    with pytest.raises(ValueError, match="optimizer is not checkpointable"):
        runner.run(iterations=3, checkpoint_interval=1).result(timeout=5)
    mock_optimization_problem.optimizer.ingest.assert_called_once_with([{"_id": 0, "objective": 1.0}])
    mock_re_manager_api.item_add.assert_called_once()


def test_runner_checkpoint_error_does_not_fail_ingested_points(mock_optimization_problem, mock_re_manager_api):
    optimizer = MagicMock(spec=FaultAwareCheckpointableOptimizer)
    optimizer.suggest.return_value = [{"_id": 0, "motor1": 2.0}]
    error = RuntimeError("checkpoint failed")
    optimizer.checkpoint.side_effect = error
    problem = replace(mock_optimization_problem, optimizer=optimizer)
    runner = QueueserverOptimizationRunner(problem, QueueserverClient(mock_re_manager_api))

    with pytest.raises(RuntimeError) as exc_info:
        runner.run(iterations=3, checkpoint_interval=1).result(timeout=5)

    assert exc_info.value is error
    optimizer.ingest.assert_called_once_with([{"_id": 0, "objective": 1.0}])
    optimizer.register_failures.assert_not_called()
    mock_re_manager_api.item_add.assert_called_once()


@pytest.mark.parametrize("exit_status", ["success", "fail", "abort"])
def test_runner_document_stream_completion_inside_submission(mock_optimization_problem, mock_re_manager_api, exit_status):
    dispatcher = Dispatcher()
    outcomes = [{"_id": 0, "objective": 4.0}]
    evaluate = MagicMock(return_value=outcomes)
    optimizer = MagicMock(spec=FaultAwareOptimizer)
    suggestions = [{"_id": 0, "motor1": 2.0}]
    optimizer.suggest.return_value = suggestions

    def item_add(plan):
        _dispatch_completion(dispatcher, None, "non-blop", exit_status="fail")
        _dispatch_completion(dispatcher, "unrelated", "other-run", exit_status="abort")
        _dispatch_completion(
            dispatcher,
            plan.kwargs["md"][CORRELATION_UID_KEY],
            "actual-run-uid",
            exit_status=exit_status,
            reason="hardware fault" if exit_status != "success" else "",
        )
        return {"success": True, "item": {"item_uid": "early-item"}}

    mock_re_manager_api.item_add.side_effect = item_add
    with closing(DocumentStreamEvaluator(dispatcher, evaluate, timeout=0)) as adapter:
        problem = replace(mock_optimization_problem, optimizer=optimizer, evaluation_function=adapter)
        runner = QueueserverOptimizationRunner(problem, QueueserverClient(mock_re_manager_api))
        future = runner.run()
        if exit_status == "success":
            result = future.result(timeout=5)
            assert result.iterations_completed == 1
            assert result.uids[0].item_uid == "early-item"
            evaluate.assert_called_once_with("actual-run-uid", suggestions)
            optimizer.ingest.assert_called_once_with(outcomes)
            optimizer.register_failures.assert_not_called()
        else:
            with pytest.raises(RuntimeError) as exc_info:
                future.result(timeout=5)
            assert str(exc_info.value) == (
                f"Acquisition run 'actual-run-uid' ended with status {exit_status!r}: hardware fault"
            )
            evaluate.assert_not_called()
            optimizer.ingest.assert_not_called()
            optimizer.register_failures.assert_called_once_with(suggestions)
    mock_re_manager_api.item_add.assert_called_once()
