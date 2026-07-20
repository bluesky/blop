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
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Literal

from bluesky.callbacks import CallbackBase
from bluesky.callbacks.zmq import RemoteDispatcher

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
from .protocols import ID_KEY, CanRegisterSuggestions, QueueserverOptimizationProblem, TrialFaultAware
from .utils import _maybe_checkpoint

logger = logging.getLogger("blop")


DEFAULT_ACQUIRE_PLAN_NAME: str = default_acquire.__name__
CORRELATION_UID_KEY: Literal["blop_correlation_uid"] = "blop_correlation_uid"


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
        this reflects however many iterations completed before the stop.
    num_points : int
        The number of points suggested per iteration.
    uids : tuple[str, ...]
        The Bluesky run UIDs for each completed acquisition, in order.
        These can be used to retrieve raw data from a Tiled or databroker
        catalog for post-hoc analysis.
    """

    iterations_completed: int
    num_points: int
    uids: tuple[str, ...]


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


class QueueserverClient:
    """
    Handles communication with a Bluesky queueserver.

    This class encapsulates queueserver communication, including plan submission
    and event listening.

    .. warning::

        This class is part of the **experimental** queueserver integration.
        The API may change in future releases without a deprecation period.

    Parameters
    ----------
    re_manager_api : bluesky_queueserver_api.zmq.REManagerAPI | bluesky_queueserver_api.http.REManagerAPI
        Manager instance for communication with Bluesky Queueserver
    document_dispatcher : bluesky.callbacks.zmq.RemoteDispatcher
        Dispatcher for the Bluesky document stream.
    """

    def __init__(
        self,
        re_manager_api: bluesky_queueserver_api.zmq.REManagerAPI | bluesky_queueserver_api.http.REManagerAPI,
        document_dispatcher: RemoteDispatcher,
    ):
        self._rm = re_manager_api
        self._dispatcher = document_dispatcher
        self._consumer_callback: ConsumerCallback | None = None
        self._listener_thread: threading.Thread | None = None

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

    def check_devices_available(self, device_names: Sequence[str]) -> None:
        """
        Verify that all specified devices are available in the queueserver.

        Parameters
        ----------
        device_names : Sequence[str]
            Names of devices to check.

        Raises
        ------
        ValueError
            If any device is not available.
        """
        res = self._rm.devices_allowed()
        allowed = res["devices_allowed"]
        for name in device_names:
            if name not in allowed:
                raise ValueError(f"Device '{name}' is not available in the queueserver environment")

    def check_plan_available(self, plan_name: str) -> None:
        """
        Verify that a plan is available in the queueserver.

        Parameters
        ----------
        plan_name : str
            Name of the plan to check.

        Raises
        ------
        ValueError
            If the plan is not available.
        """
        res = self._rm.plans_allowed()
        if plan_name not in res["plans_allowed"]:
            raise ValueError(f"Plan '{plan_name}' is not available in the queueserver environment")

    def submit_plan(self, plan: BPlan, autostart: bool = True, timeout: int = 600) -> None:
        """
        Submit a plan to the queueserver queue.

        Parameters
        ----------
        plan : BPlan
            The plan to submit.
        autostart : bool, optional
            If True, start the queue after adding the plan.
        timeout : float, optional
            Timeout in seconds when waiting for queue to be idle.
        """
        response = self._rm.item_add(plan)
        logger.debug(f"Submitted plan to queue. Response: {response}")

        if autostart:
            logger.debug("Waiting for queue to be idle or paused")
            self._rm.wait_for_idle_or_paused(timeout=timeout)
            response = self._rm.queue_start()
            logger.debug(f"Started queue. Response: {response}")

    def start_listener(
        self,
        on_stop: Callable[[RunStart, RunStop], None],
    ) -> None:
        """
        Start listening for document events from the queueserver.

        Parameters
        ----------
        on_stop : callable
            Callback invoked when a stop document is received.
            Signature: on_stop(start_doc, stop_doc)
        """
        if self._listener_thread is not None:
            logger.warning("Listener already running")
            return

        self._consumer_callback = ConsumerCallback(callback=on_stop)
        self._dispatcher.subscribe(self._consumer_callback)

        logger.info("Starting document listener thread")
        self._listener_thread = threading.Thread(
            target=self._dispatcher.start,
            name="qserver-document-consumer",
            daemon=True,
        )
        self._listener_thread.start()

    def stop_listener(self) -> None:
        """Stop the document listener thread."""
        if self._listener_thread is not None:
            self._dispatcher.stop()
        self._consumer_callback = None
        self._listener_thread = None
        logger.info("Stopped document listener")


@dataclass
class _OptimizationState:
    """Internal mutable state for an optimization run."""

    max_iterations: int = 1
    num_points: int = 1
    checkpoint_interval: int | None = None
    current_iteration: int = 0
    current_suggestions: list[dict] = field(default_factory=list)
    current_uid: str | None = None
    uids: list[str] = field(default_factory=list)

    def build_result(self) -> OptimizationResult:
        """Build an :class:`OptimizationResult` from the current state."""
        return OptimizationResult(
            iterations_completed=len(self.uids),
            num_points=self.num_points,
            uids=tuple(self.uids),
        )


class QueueserverOptimizationRunner:
    """
    Runs optimization loops through a Bluesky queueserver.

    This class coordinates the optimization workflow by getting suggestions from
    the optimizer, submitting acquisition plans to the queueserver, and ingesting
    results when plans complete.

    .. warning::

        This class is part of the **experimental** queueserver integration.
        The API may change in future releases without a deprecation period.

    Parameters
    ----------
    optimization_problem : QueueserverOptimizationProblem
        The optimization problem to solve, containing the optimizer, actuators,
        sensors, and evaluation function.
    queueserver_client : QueueserverClient
        Client for communicating with the queueserver.
        The document listener is started once during runner construction so it
        is ready before any submitted plan can emit documents.
    """

    def __init__(
        self,
        optimization_problem: QueueserverOptimizationProblem,
        queueserver_client: QueueserverClient,
    ):
        self._problem = optimization_problem
        self._client = queueserver_client
        self._plan_name = optimization_problem.acquisition_plan or DEFAULT_ACQUIRE_PLAN_NAME
        self._state: _OptimizationState | None = None
        self._continuous = True
        self._autostart = True
        self._state_lock = threading.RLock()
        self._current_future: Future[OptimizationResult] | None = None
        self._client.start_listener(
            on_stop=self._on_acquisition_complete,
        )

    @property
    def optimization_problem(self) -> QueueserverOptimizationProblem:
        """The optimization problem being solved."""
        return self._problem

    @property
    def current_iteration(self) -> int:
        """The current iteration number (0 if not running)."""
        with self._state_lock:
            return self._state.current_iteration if self._state else 0

    def run(
        self, iterations: int = 1, num_points: int = 1, checkpoint_interval: int | None = None
    ) -> Future[OptimizationResult]:
        """
        Run the optimization loop.

        Validates the queueserver state, then begins the suggest -> acquire -> ingest
        cycle. This method returns immediately; the optimization runs asynchronously
        via callbacks on the Bluesky document stream.

        Parameters
        ----------
        iterations : int
            Number of optimization iterations to run.
        num_points : int
            Number of points to suggest per iteration.
        checkpoint_interval : int | None
            The number of iterations between optimizer checkpoints. If None, checkpoints
            will not be saved. Optimizer must implement the
            :class:`blop.protocols.Checkpointable` protocol.

        Returns
        -------
        concurrent.futures.Future[OptimizationResult]
            A future that resolves to an :class:`OptimizationResult` when all
            iterations complete, or when :meth:`stop` is called. If the
            optimization loop raises an unhandled exception the future will
            hold that exception and re-raise it on ``.result()``.

        Raises
        ------
        RuntimeError
            If the queueserver environment is not ready.
        ValueError
            If required devices or plans are not available.
        """
        with self._state_lock:
            self._validate()
            self._state = _OptimizationState(
                max_iterations=iterations, num_points=num_points, checkpoint_interval=checkpoint_interval
            )
            self._continuous = True
        suggestions = self._problem.optimizer.suggest(num_points)
        with self._state_lock:
            plan = self._build_plan(suggestions)
            logger.info(
                f"Submitting iteration {self._state.current_iteration}/{self._state.max_iterations} "
                f"with correlation uid: {self._state.current_uid}"
            )
            future: Future[OptimizationResult] = Future()
            self._current_future = future
        try:
            self._client.submit_plan(plan, autostart=self._autostart)
        except Exception as exc:
            with self._state_lock:
                self._fail_future(exc)
            raise
        return future

    def submit_suggestions(self, suggestions: list[dict]) -> Future[OptimizationResult]:
        """
        Manually submit suggestions to the queue.

        This method returns immediately; the optimization runs asynchronously via callbacks on the Bluesky document stream.

        Parameters
        ----------
        suggestions : list[dict]
            Parameter combinations to evaluate. Can be:

            - Optimizer suggestions (with "_id" keys from suggest())
            - Manual points (without "_id", requires CanRegisterSuggestions protocol)

        Returns
        -------
        concurrent.futures.Future[OptimizationResult]
            A future that resolves to an :class:`OptimizationResult` when the
            acquisition completes. If an unhandled exception occurs the future
            will hold it and re-raise on ``.result()``.
        """
        with self._state_lock:
            self._validate()
            self._state = _OptimizationState(max_iterations=1, num_points=len(suggestions))
            self._continuous = False
        # Ensure all suggestions have an ID_KEY or register them with the optimizer
        if not isinstance(self.optimization_problem.optimizer, CanRegisterSuggestions) and any(
            ID_KEY not in suggestion for suggestion in suggestions
        ):
            raise ValueError(
                f"All suggestions must contain an '{ID_KEY}' key to later match with the outcomes or your optimizer must "
                "implement the `blop.protocols.CanRegisterSuggestions` protocol. Please review your optimizer "
                f"implementation. Got suggestions: {suggestions}"
            )
        elif isinstance(self.optimization_problem.optimizer, CanRegisterSuggestions):
            suggestions = self.optimization_problem.optimizer.register_suggestions(suggestions)

        with self._state_lock:
            plan = self._build_plan(suggestions)
            logger.info(f"Submitting manually specified suggestion(s) with correlation uid: {self._state.current_uid}")
            future: Future[OptimizationResult] = Future()
            self._current_future = future
        try:
            self._client.submit_plan(plan, autostart=self._autostart)
        except Exception as exc:
            with self._state_lock:
                self._fail_future(exc)
            raise
        return future

    def stop(self) -> None:
        """
        Stop the optimization loop gracefully.

        The future returned by :meth:`run` or :meth:`submit_suggestions` will
        be resolved with a partial :class:`OptimizationResult` containing however
        many iterations completed before the stop.
        """
        with self._state_lock:
            self._continuous = False
            result = (
                self._state.build_result()
                if self._state is not None
                else OptimizationResult(iterations_completed=0, num_points=0, uids=())
            )
            self._resolve_future(result)
        logger.info("Optimization stopped")

    def _resolve_future(self, result: OptimizationResult) -> None:
        """LOCKED: Resolve the current future with a result if it is not already done."""
        if self._current_future is not None and not self._current_future.done():
            self._current_future.set_result(result)

    def _fail_future(self, exc: Exception) -> None:
        """LOCKED: Resolve the current future with an exception if it is not already done."""
        if self._current_future is not None and not self._current_future.done():
            self._current_future.set_exception(exc)

    def _try_register_failures(self, suggestions: list[dict]) -> None:
        """Notify a TrialFaultAware optimizer of failed suggestions, if supported."""
        if suggestions and isinstance(self._problem.optimizer, TrialFaultAware):
            try:
                self._problem.optimizer.register_failures(suggestions)
            except Exception:
                logger.exception("Failed to register trial failures with the optimizer")
                raise

    def _validate(self) -> None:
        """LOCKED: Validate not already running, queueserver environment, devices, and plan availability."""
        if self._current_future is not None and not self._current_future.done():
            raise RuntimeError("Optimization loop is already running.")
        self._client.check_environment()

        # Collect device names from actuators and sensors
        actuator_names = list(self._problem.actuators)
        sensor_names = list(self._problem.sensors)
        self._client.check_devices_available(actuator_names + sensor_names)
        self._client.check_plan_available(self._plan_name)

    def _build_plan(self, suggestions: list[dict]) -> BPlan:
        """LOCKED: Build the plan to submit and update current state."""
        if self._state is None:
            raise RuntimeError("_build_plan() called before run() or submit_suggestions()")

        self._state.current_iteration += 1
        self._state.current_suggestions = suggestions
        self._state.current_uid = str(uuid.uuid4())

        # Build metadata
        md: dict[str, Any] = {
            CORRELATION_UID_KEY: self._state.current_uid,
            "blop_suggestions": self._state.current_suggestions,
        }

        return BPlan(
            self._plan_name,
            self._state.current_suggestions,
            list(self._problem.actuators),
            list(self._problem.sensors),
            md=md,
            **(self._problem.acquisition_plan_kwargs or {}),
        )

    def _on_acquisition_complete(self, start_doc: RunStart, stop_doc: RunStop) -> None:
        """When acquisition finishes, ingest results, and maybe continue."""
        try:
            self._process_acquisition(start_doc, stop_doc)
        except Exception as exc:
            logger.exception(
                "Unhandled exception in optimization background thread. "
                "Optimization has been stopped. Inspect the future's exception for details."
            )
            with self._state_lock:
                suggestions = self._state.current_suggestions if self._state is not None else []
                self._fail_future(exc)
            self._try_register_failures(suggestions)

    def _process_acquisition(self, start_doc: RunStart, stop_doc: RunStop) -> None:
        """Core acquisition-complete logic (called from _on_acquisition_complete)."""
        with self._state_lock:
            if self._state is None:
                raise RuntimeError("_on_acquisition_complete called before run()")
            if self._state.current_uid is None:
                raise RuntimeError("current_uid not set")
            exit_status = stop_doc.get("exit_status")
            if self._state.current_uid != start_doc.get(CORRELATION_UID_KEY):
                raise RuntimeError(
                    "current_uid did not match start document. "
                    f"Got: {start_doc.get(CORRELATION_UID_KEY)}, Expected: {self._state.current_uid}"
                )
            if exit_status != "success":
                reason = stop_doc.get("reason") or "(no reason given)"
                raise RuntimeError(f"Acquisition run {start_doc['uid']!r} ended with status {exit_status!r}: {reason}")
            logger.info(f"Acquisition complete for uid: {self._state.current_uid}")
            suggestions = self._state.current_suggestions
            self._state.uids.append(start_doc["uid"])

        # Evaluate the results
        outcomes = self._problem.evaluation_function(uid=start_doc["uid"], suggestions=suggestions)
        logger.info(f"Evaluated {len(outcomes)} outcomes")

        # Ingest into optimizer
        self._problem.optimizer.ingest(outcomes)

        with self._state_lock:
            # Checkpoint optimizer, if applicable
            _maybe_checkpoint(
                self._problem.optimizer, self._state.checkpoint_interval, iteration=self._state.current_iteration
            )
            # Check if done
            if not self._continuous or self._state.current_iteration >= self._state.max_iterations:
                logger.info(f"Optimization complete after {self._state.current_iteration} iterations")
                result = self._state.build_result()
                self._resolve_future(result)
                return

            # Continue: get next suggestions and submit
            num_points = self._state.num_points

        suggestions = self._problem.optimizer.suggest(num_points)
        with self._state_lock:
            plan = self._build_plan(suggestions)
            logger.info(
                f"Submitting iteration {self._state.current_iteration}/{self._state.max_iterations} "
                f"with correlation uid: {self._state.current_uid}"
            )
        self._client.submit_plan(plan, autostart=self._autostart)
