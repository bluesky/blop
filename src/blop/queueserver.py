"""
Queueserver integration for running optimization through a Bluesky queueserver.

.. warning::

    This module is **experimental**. The API is not yet stable and may change
    in future releases without a deprecation period. It is not recommended for
    production use.

This module provides components for running optimization loops remotely through
a queueserver, rather than directly through a RunEngine.
"""

import logging
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from bluesky.callbacks import CallbackBase
from bluesky.run_engine import Dispatcher

try:
    import bluesky_queueserver_api.http
    import bluesky_queueserver_api.zmq
    from bluesky_queueserver_api import BPlan
except ImportError as e:
    raise ImportError(
        "The queueserver integration requires additional dependencies. Install them with: pip install blop[queueserver]"
    ) from e
from event_model import RunStart, RunStop

from .plans import default_acquire
from .protocols import ID_KEY, CanRegisterSuggestions, EvaluationFunction, QueueserverOptimizationProblem, TrialFaultAware
from .utils import _maybe_checkpoint

logger = logging.getLogger("blop")


DEFAULT_ACQUIRE_PLAN_NAME: str = default_acquire.__name__
CORRELATION_UID_KEY: Literal["blop_correlation_uid"] = "blop_correlation_uid"


@dataclass(frozen=True, slots=True)
class QueueserverAcquisition:
    """
    Client-side identity of a submitted Queue Server acquisition.

    Parameters
    ----------
    correlation_uid : str
        Blop correlation UID attached to the acquisition plan's metadata.
    item_uid : str | None
        Queue Server item UID, when known.
    plan_name : str
        Name of the submitted acquisition plan.

    Notes
    -----
    None of these fields is implicitly a Bluesky run UID. This hashable token
    is passed to the evaluator, never to the server plan or its metadata.
    """

    correlation_uid: str
    item_uid: str | None
    plan_name: str


@dataclass(frozen=True)
class OptimizationResult:
    """
    The result of a completed or stopped optimization run.

    .. warning::

        This class is part of the **experimental** queueserver integration.
        The API may change in future releases without a deprecation period.

    Parameters
    ----------
    iterations_completed : int
        The number of suggest -> acquire -> ingest cycles that finished
        successfully. For a run stopped early via :meth:`QueueserverOptimizationRunner.stop`,
        this includes any in-flight acquisition that finished after the stop request.
    num_points : int
        The number of points suggested per iteration.
    uids : tuple[QueueserverAcquisition, ...]
        Tokens for successfully evaluated and ingested acquisitions, in order.
        These are submission identities, not Bluesky run UIDs.
    """

    iterations_completed: int
    num_points: int
    uids: tuple[QueueserverAcquisition, ...]


class ConsumerCallback(CallbackBase):
    """
    A callback that caches the start document and invokes a callback on stop.

    Parameters
    ----------
    callback : callable
        Function to call when a stop document is received.
        Signature: callback(start_doc, stop_doc)
    """

    def __init__(self, callback: Callable[[RunStart, RunStop], None] | None = None):
        super().__init__()
        self._start_doc_cache: dict[str, RunStart] = {}
        self._callback = callback

    def start(self, doc: RunStart) -> None:
        """
        Process the start document.

        Caches the start document if it has the Blop injected correlation UID.
        """
        if doc.get(CORRELATION_UID_KEY):
            self._start_doc_cache[doc["uid"]] = doc

    def stop(self, doc: RunStop) -> None:
        """
        Process the stop document.

        Calls the callback with the cached start document and stop document pair.
        """
        start_doc = self._start_doc_cache.pop(doc["run_start"], None)
        if self._callback is not None and start_doc is not None:
            self._callback(start_doc, doc)


class DocumentStreamEvaluator:
    """
    Adapt a run-UID evaluator to completed, correlated document streams.

    Parameters
    ----------
    document_dispatcher : Dispatcher
        Application-owned dispatcher. Start its transport and construct this
        adapter before submitting acquisitions; this adapter only subscribes.
    evaluation_function : EvaluationFunction[str]
        Evaluator called with the actual run UID after a successful stop.
    timeout : float | None, optional
        Maximum wait for matching start/stop documents, in seconds. The default
        waits indefinitely; this does not limit the wrapped evaluator's runtime.

    Notes
    -----
    Each acquisition must emit one correlated run. Multi-run aggregation or
    partial-data readiness requires a custom token evaluator. Close the adapter
    when its workflow ends, for example with :func:`contextlib.closing`.
    Closing unsubscribes only this adapter; it never stops the transport or
    interrupts a wrapped evaluator that is already executing.
    """

    def __init__(
        self,
        document_dispatcher: Dispatcher,
        evaluation_function: EvaluationFunction[str],
        *,
        timeout: float | None = None,
    ) -> None:
        self._dispatcher = document_dispatcher
        self._evaluation_function = evaluation_function
        self._timeout = timeout
        self._condition = threading.Condition()
        self._completed: dict[str, tuple[RunStart, RunStop]] = {}
        self._closed = False
        self._consumer: ConsumerCallback | None = ConsumerCallback(self._on_stop)
        self._subscription = document_dispatcher.subscribe(self._consumer)

    def _on_stop(self, start_doc: RunStart, stop_doc: RunStop) -> None:
        with self._condition:
            if not self._closed:
                # RunStart's schema does not describe application-defined metadata.
                metadata = cast(Mapping[str, Any], start_doc)
                self._completed.setdefault(metadata[CORRELATION_UID_KEY], (start_doc, stop_doc))
                self._condition.notify_all()

    def __call__(self, uid: QueueserverAcquisition, suggestions: Sequence[Mapping]) -> Sequence[Mapping]:
        """
        Wait for a successful correlated run and evaluate its data.

        Parameters
        ----------
        uid : QueueserverAcquisition
            Submission token identifying the acquisition to await.
        suggestions : Sequence[Mapping]
            Parameterizations passed unchanged to the wrapped evaluator.

        Returns
        -------
        Sequence[Mapping]
            Outcomes returned unchanged by the wrapped evaluator.

        Raises
        ------
        TimeoutError
            If matching completion does not arrive within the timeout.
        RuntimeError
            If the adapter is closed or the acquisition run was unsuccessful.
        """
        with self._condition:
            if not self._condition.wait_for(
                lambda: self._closed or uid.correlation_uid in self._completed,
                timeout=self._timeout,
            ):
                raise TimeoutError(f"Timed out waiting for acquisition {uid.correlation_uid!r}.")
            if self._closed:
                raise RuntimeError("Document stream evaluator is closed.")
            start_doc, stop_doc = self._completed.pop(uid.correlation_uid)

        exit_status = stop_doc.get("exit_status")
        if exit_status != "success":
            reason = stop_doc.get("reason") or "(no reason given)"
            raise RuntimeError(f"Acquisition run {start_doc['uid']!r} ended with status {exit_status!r}: {reason}")
        return self._evaluation_function(start_doc["uid"], suggestions)

    def close(self) -> None:
        """Release waiters and the owned subscription without stopping transport."""
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._completed.clear()
            self._condition.notify_all()
        self._dispatcher.unsubscribe(self._subscription)
        self._consumer = None


class QueueserverClient:
    """
    Handles communication with a Bluesky queueserver.

    .. warning::

        This class is part of the **experimental** queueserver integration.
        The API may change in future releases without a deprecation period.

    Parameters
    ----------
    re_manager_api : bluesky_queueserver_api.zmq.REManagerAPI | bluesky_queueserver_api.http.REManagerAPI
        Manager instance for communication with Bluesky Queueserver
    autostart : bool, optional
        Whether Queue Server should automatically start processing queued plans.
    """

    def __init__(
        self,
        re_manager_api: bluesky_queueserver_api.zmq.REManagerAPI | bluesky_queueserver_api.http.REManagerAPI,
        *,
        autostart: bool = True,
    ) -> None:
        self._rm = re_manager_api

        response = self._rm.queue_autostart(autostart)
        logger.debug(f"Set queue autostart to {autostart}. Response: {response}")

    def check_environment(self) -> None:
        """
        Verify that the queueserver environment is ready.

        Raises
        ------
        RuntimeError
            If the queueserver environment is not open.
        """
        status = self._rm.status()
        if status is None or not status.get("worker_environment_exists", False):
            raise RuntimeError("The queueserver environment is not open")

    def submit_plan(self, plan: BPlan) -> str:
        """
        Submit a plan to the queueserver queue.

        Parameters
        ----------
        plan : BPlan
            The plan to submit.

        Returns
        -------
        str
            The item UID assigned by Queue Server. Request and transport errors
            propagate unchanged; plan and device permissions are server-owned.
        """
        response = self._rm.item_add(plan)
        logger.debug(f"Submitted plan to queue. Response: {response}")
        return response["item"]["item_uid"]


@dataclass
class _OptimizationState:
    """Internal mutable state for an optimization run."""

    max_iterations: int = 1
    num_points: int = 1
    checkpoint_interval: int | None = None
    current_iteration: int = 0
    uids: list[QueueserverAcquisition] = field(default_factory=list)
    stop_requested: bool = False
    future_claimed: bool = False

    def build_result(self) -> OptimizationResult:
        """Build an :class:`OptimizationResult` from the current state."""
        return OptimizationResult(
            iterations_completed=len(self.uids),
            num_points=self.num_points,
            uids=tuple(self.uids),
        )


class QueueserverOptimizationRunner:
    """
    Run token-driven optimization through a Bluesky Queue Server.

    One worker generates suggestions, submits acquisition plans, evaluates their
    tokens, ingests outcomes, and checkpoints in order. The evaluator owns data
    readiness; no document transport or plan-completion gate is required.

    .. warning::

        This class is part of the **experimental** queueserver integration.
        The API may change in future releases without a deprecation period.

    Parameters
    ----------
    optimization_problem : QueueserverOptimizationProblem[QueueserverAcquisition]
        Optimizer, remote device names, acquisition plan, and token evaluator.
    queueserver_client : QueueserverClient
        Client for communicating with the queueserver.

    Notes
    -----
    Existing run-UID evaluators can opt into :class:`DocumentStreamEvaluator`.
    Construct that adapter before submission and close it after consuming the
    future, for example with :func:`contextlib.closing`. Its document transport
    remains application-owned. The runner never closes the evaluator.

    Interrupting the caller during startup does not cancel optimization work
    already owned by the worker. New runs remain blocked until that work ends;
    use :meth:`stop` to prevent later acquisitions. If startup fails before the
    worker takes ownership, no acquisition is submitted for that run.

    Failure-notification exceptions are logged with their tracebacks. When an
    earlier optimization error exists, it remains the exception raised by
    ``future.result()`` rather than being replaced by the notification failure.
    """

    def __init__(
        self,
        optimization_problem: QueueserverOptimizationProblem[QueueserverAcquisition],
        queueserver_client: QueueserverClient,
    ) -> None:
        self._problem = optimization_problem
        self._client = queueserver_client
        self._plan_name = optimization_problem.acquisition_plan or DEFAULT_ACQUIRE_PLAN_NAME
        self._state: _OptimizationState | None = None
        self._state_lock = threading.RLock()
        self._current_future: Future[OptimizationResult] | None = None

    @property
    def optimization_problem(self) -> QueueserverOptimizationProblem[QueueserverAcquisition]:
        """The optimization problem being solved."""
        return self._problem

    @property
    def current_iteration(self) -> int:
        """The current submission iteration number, or zero before a run."""
        with self._state_lock:
            return self._state.current_iteration if self._state else 0

    def run(
        self, iterations: int = 1, num_points: int = 1, checkpoint_interval: int | None = None
    ) -> Future[OptimizationResult]:
        """
        Start an asynchronous suggest, submit, evaluate, and ingest loop.

        Admission and environment checks are synchronous. After worker launch,
        suggestion, submission, evaluation, ingestion, and checkpoint errors
        propagate through the returned future's ``result()``.

        Parameters
        ----------
        iterations : int
            Number of optimization iterations to run.
        num_points : int
            Number of points to suggest per iteration.
        checkpoint_interval : int | None
            Save an optimizer checkpoint every N completed iterations. None
            disables checkpoints. The optimizer must implement
            :class:`blop.protocols.Checkpointable` when checkpoints are enabled.

        Returns
        -------
        concurrent.futures.Future[OptimizationResult]
            A running, non-cancellable future containing acquisition tokens.
            It completes only after evaluation, ingestion, checkpointing,
            and failure notification have finished, including in-flight work
            after :meth:`stop`.

        Raises
        ------
        RuntimeError
            If a run is still active or the queueserver environment is not ready.
        """
        return self._start_run(
            _OptimizationState(max_iterations=iterations, num_points=num_points, checkpoint_interval=checkpoint_interval)
        )

    def submit_suggestions(self, suggestions: Sequence[Mapping]) -> Future[OptimizationResult]:
        """
        Evaluate one manually supplied batch asynchronously.

        Parameters
        ----------
        suggestions : Sequence[Mapping]
            Parameter combinations to evaluate. Optimizers implementing
            :class:`blop.protocols.CanRegisterSuggestions` register this batch
            even if it already has IDs. Otherwise each point must have "_id".

        Returns
        -------
        concurrent.futures.Future[OptimizationResult]
            A running future whose result contains the acquisition token after
            evaluation and ingestion. Registration, submission, evaluation, and
            ingestion errors propagate through ``result()``. A stop request does
            not complete the future while in-flight work or failure notification
            remains unfinished.

        Raises
        ------
        RuntimeError
            If a run is still active or the queueserver environment is not ready.
        ValueError
            If points lack "_id" and the optimizer cannot register suggestions.
        """
        return self._start_run(
            _OptimizationState(max_iterations=1, num_points=len(suggestions)),
            supplied_suggestions=suggestions,
        )

    def _start_run(
        self,
        state: _OptimizationState,
        supplied_suggestions: Sequence[Mapping] | None = None,
    ) -> Future[OptimizationResult]:
        future: Future[OptimizationResult] | None = None
        try:
            with self._state_lock:
                self._validate()
                if (
                    supplied_suggestions is not None
                    and not isinstance(self._problem.optimizer, CanRegisterSuggestions)
                    and any(ID_KEY not in suggestion for suggestion in supplied_suggestions)
                ):
                    raise ValueError(
                        f"All suggestions must contain an '{ID_KEY}' key to later match with the outcomes or "
                        "your optimizer must implement the `blop.protocols.CanRegisterSuggestions` protocol. "
                        f"Please review your optimizer implementation. Got suggestions: {supplied_suggestions}"
                    )
                future = Future()
                future.set_running_or_notify_cancel()
                self._state = state
                self._current_future = future

            threading.Thread(
                target=self._run_loop,
                args=(state, future, supplied_suggestions),
                name="qserver-optimization",
                daemon=True,
            ).start()
        except BaseException as error:
            if future is not None:
                # Thread.start() can be interrupted after spawning the worker.
                with self._state_lock:
                    if state.future_claimed:
                        raise
                    state.future_claimed = True
                future.set_exception(error)
            raise
        return future

    def _run_loop(
        self,
        state: _OptimizationState,
        future: Future[OptimizationResult],
        supplied_suggestions: Sequence[Mapping] | None,
    ) -> None:
        # A failed launcher can claim settlement before a late worker starts.
        with self._state_lock:
            if state.future_claimed:
                return
            state.future_claimed = True

        pending_suggestions: Sequence[Mapping] = ()
        error: BaseException | None = None
        try:
            for _ in range(state.max_iterations):
                with self._state_lock:
                    if state.stop_requested:
                        break
                pending_suggestions = ()
                if supplied_suggestions is None:
                    suggestions = self._problem.optimizer.suggest(state.num_points)
                elif isinstance(self._problem.optimizer, CanRegisterSuggestions):
                    suggestions = self._problem.optimizer.register_suggestions(supplied_suggestions)
                else:
                    suggestions = supplied_suggestions
                pending_suggestions = suggestions

                # Serialize stop with submission, not with evaluation or ingestion.
                with self._state_lock:
                    if state.stop_requested:
                        break
                    state.current_iteration += 1
                    correlation_uid = str(uuid.uuid4())
                    plan = self._build_plan(suggestions, correlation_uid)
                    logger.info(
                        f"Submitting iteration {state.current_iteration}/{state.max_iterations} "
                        f"with correlation uid: {correlation_uid}"
                    )
                    item_uid = self._client.submit_plan(plan)
                    acquisition = QueueserverAcquisition(correlation_uid, item_uid, self._plan_name)

                outcomes = self._problem.evaluation_function(acquisition, suggestions)
                self._problem.optimizer.ingest(outcomes)
                state.uids.append(acquisition)
                pending_suggestions = ()
                _maybe_checkpoint(self._problem.optimizer, state.checkpoint_interval, iteration=state.current_iteration - 1)
        except BaseException as exc:
            # Transfer worker termination to the caller instead of stranding its future.
            error = exc

        if pending_suggestions:
            try:
                self._try_register_failures(pending_suggestions)
            except BaseException as exc:
                if error is None:
                    error = exc

        # Done callbacks may start another run; this worker only owns this future.
        if error is not None:
            future.set_exception(error)
        else:
            future.set_result(state.build_result())

    def stop(self) -> None:
        """
        Request that no later acquisition be submitted.

        An already-started submission may finish before this method returns.
        Evaluation is never interrupted or awaited here. The running future
        remains pending until in-flight evaluation, ingestion, checkpointing,
        and pending failure notification finish. This does not stop or abort
        Queue Server, close an evaluator, or cancel the future.
        """
        with self._state_lock:
            if self._state is not None:
                self._state.stop_requested = True
        logger.info("Optimization stop requested")

    def _try_register_failures(self, suggestions: Sequence[Mapping]) -> None:
        """Notify a TrialFaultAware optimizer of failed suggestions, if supported."""
        if suggestions and isinstance(self._problem.optimizer, TrialFaultAware):
            try:
                self._problem.optimizer.register_failures(suggestions)
            except BaseException:
                # The worker may preserve an earlier error, so record this failure before re-raising.
                logger.exception("Failed to register trial failures with the optimizer")
                raise

    def _validate(self) -> None:
        """LOCKED: Check run admission and the queueserver environment."""
        if self._current_future is not None and not self._current_future.done():
            raise RuntimeError("Optimization loop is already running.")
        self._client.check_environment()

    def _build_plan(self, suggestions: Sequence[Mapping], correlation_uid: str) -> BPlan:
        md: dict[str, Any] = {
            CORRELATION_UID_KEY: correlation_uid,
            "blop_suggestions": suggestions,
        }
        return BPlan(
            self._plan_name,
            suggestions,
            list(self._problem.actuators),
            list(self._problem.sensors),
            md=md,
            **(self._problem.acquisition_plan_kwargs or {}),
        )
