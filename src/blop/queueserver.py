"""
Queueserver integration for running optimization through a Bluesky queueserver.

This module provides components for running optimization loops remotely through
a queueserver, rather than directly through a RunEngine.
"""

import logging
import threading
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from bluesky.callbacks import CallbackBase
from bluesky.callbacks.zmq import RemoteDispatcher
from bluesky_queueserver_api import BPlan
from bluesky_queueserver_api.zmq import REManagerAPI
from event_model import RunStart, RunStop

from .plans import default_acquire
from .protocols import RemoteOptimizationProblem

logger = logging.getLogger("blop")


DEFAULT_ACQUIRE_PLAN_NAME: str = default_acquire.__name__
CORRELATION_UID_KEY: Literal["blop_correlation_uid"] = "blop_correlation_uid"


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
        self._start_doc_cache: RunStart | None = None
        self._callback = callback

    def start(self, doc: RunStart) -> None:
        """Caches the start document if it came from Blop"""
        if doc.get(CORRELATION_UID_KEY, None):
            self._start_doc_cache = doc
        else:
            self._start_doc_cache = None

    def stop(self, doc: RunStop) -> None:
        """Executes the callback if the cached start and stop document match"""
        if self._callback is not None and self._start_doc_cache is not None and self._start_doc_cache["uid"] == doc["uid"]:
            self._callback(self._start_doc_cache, doc)
        self._start_doc_cache = None


class QueueserverClient:
    """
    Handles communication with a Bluesky queueserver.

    This class encapsulates all ZMQ and HTTP communication with the queueserver,
    including plan submission and event listening.

    Parameters
    ----------
    re_manager_api : bluesky_queueserver_api.zmq.REManagerAPI
        Manager instance for communication with Bluesky Queueserver
    zmq_consumer_addr : str
        Address for ZMQ document consumer (e.g., "localhost:5578").
    """

    def __init__(
        self,
        re_manager_api: REManagerAPI,
        zmq_consumer_addr: str,
    ):
        self._zmq_consumer_addr = zmq_consumer_addr

        self._rm = re_manager_api
        self._dispatcher: RemoteDispatcher | None = None
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

    def start_listener(self, on_stop: Callable[[RunStart, RunStop], None]) -> None:
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

        dispatcher = RemoteDispatcher(self._zmq_consumer_addr)
        self._consumer_callback = ConsumerCallback(callback=on_stop)
        dispatcher.subscribe(self._consumer_callback)

        logger.info("Starting ZMQ listener thread")
        self._listener_thread = threading.Thread(
            target=dispatcher.start,
            name="qserver-zmq-consumer",
            daemon=True,
        )
        self._listener_thread.start()
        self._dispatcher = dispatcher

    def stop_listener(self) -> None:
        """Stop the ZMQ listener thread."""
        if self._dispatcher is not None:
            self._dispatcher.stop()
            self._dispatcher = None
        self._consumer_callback = None
        self._listener_thread = None
        logger.info("Stopped ZMQ listener")


@dataclass
class _OptimizationState:
    """Internal mutable state for an optimization run."""

    max_iterations: int = 1
    num_points: int = 1
    current_iteration: int = 0
    current_suggestions: list[dict] = field(default_factory=list)
    current_uid: str | None = None
    finished: bool = False


class QueueserverOptimizationRunner:
    """
    Runs optimization loops through a Bluesky queueserver.

    This class coordinates the optimization workflow by getting suggestions from
    the optimizer, submitting acquisition plans to the queueserver, and ingesting
    results when plans complete.

    Parameters
    ----------
    optimization_problem : RemoteOptimizationProblem
        The optimization problem to solve, containing the optimizer, actuators,
        sensors, and evaluation function.
    qserver_client : QueueserverClient
        Client for communicating with the queueserver.
    acquisition_plan_name : str
        Name of the acquisition plan registered in the queueserver.
    """

    def __init__(
        self,
        optimization_problem: RemoteOptimizationProblem,
        qserver_client: QueueserverClient,
        acquisition_plan_name: str = DEFAULT_ACQUIRE_PLAN_NAME,
    ):
        self._problem = optimization_problem
        self._client = qserver_client
        self._plan_name = acquisition_plan_name
        self._state: _OptimizationState | None = None
        self._continuous = True
        self._autostart = True

    @property
    def optimization_problem(self) -> RemoteOptimizationProblem:
        """The optimization problem being solved."""
        return self._problem

    @property
    def is_running(self) -> bool:
        """Whether an optimization run is currently in progress."""
        return self._state is not None and not self._state.finished

    @property
    def current_iteration(self) -> int:
        """The current iteration number (0 if not running)."""
        return self._state.current_iteration if self._state else 0

    def run(self, iterations: int = 1, num_points: int = 1) -> None:
        """
        Start the optimization loop.

        Validates the queueserver state, then begins the suggest -> acquire -> ingest
        cycle. This method returns immediately; the optimization runs asynchronously
        via callbacks.

        Parameters
        ----------
        iterations : int
            Number of optimization iterations to run.
        num_points : int
            Number of points to suggest per iteration.

        Raises
        ------
        RuntimeError
            If the queueserver environment is not ready.
        ValueError
            If required devices or plans are not available.
        """
        # TODO: What if there is already a run?
        self._validate()
        self._state = _OptimizationState(max_iterations=iterations, num_points=num_points)
        self._continuous = True
        self._client.start_listener(on_stop=self._on_acquisition_complete)
        self._submit_next()

    def stop(self) -> None:
        """
        Stop the optimization loop gracefully.

        The current acquisition will complete, but no further iterations will run.
        """
        self._continuous = False
        self._client.stop_listener()
        if self._state is not None:
            self._state.finished = True
        logger.info("Optimization stopped")

    def _validate(self) -> None:
        """Validate queueserver environment, devices, and plan availability."""
        self._client.check_environment()

        # Collect device names from actuators and sensors
        actuator_names = list(self._problem.actuators)
        sensor_names = list(self._problem.sensors)
        self._client.check_devices_available(actuator_names + sensor_names)

        self._client.check_plan_available(self._plan_name)

    def _submit_next(self) -> None:
        """Get suggestions from optimizer and submit plan to queueserver."""
        if self._state is None:
            raise RuntimeError("_submit_next called before run()")
        self._state.current_iteration += 1
        self._state.current_suggestions = self._problem.optimizer.suggest(self._state.num_points)
        self._state.current_uid = str(uuid.uuid4())

        logger.info(
            f"Submitting iteration {self._state.current_iteration}/{self._state.max_iterations} "
            f"with suggestion uid: {self._state.current_uid}"
        )

        plan = self._build_plan()
        self._client.submit_plan(plan, autostart=self._autostart)

    def _build_plan(self) -> BPlan:
        """Construct a BPlan from the current suggestions."""
        if self._state is None:
            raise RuntimeError("_build_plan called before run()")
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
        )

    def _on_acquisition_complete(self, start_doc: RunStart, stop_doc: RunStop) -> None:
        """Callback when acquisition finishes. Ingest results and maybe continue."""
        if self._state is None:
            raise RuntimeError("_on_acquisition_complete called before run()")
        if self._state.current_uid is None:
            raise RuntimeError("current_uid not set")
        if self._state.current_uid != start_doc.get("blop_correlation_uid", None):
            raise RuntimeError(
                "current_uid did not match start document. "
                f"Got: {start_doc.get('blop_correlation_uid', None)}, Expected: {self._state.current_uid}"
            )
        logger.info(f"Acquisition complete for uid: {self._state.current_uid}")

        # Evaluate the results
        outcomes = self._problem.evaluation_function(
            uid=start_doc["uid"],
            suggestions=self._state.current_suggestions,
        )

        logger.info(f"Evaluated {len(outcomes)} outcomes")

        # Ingest into optimizer
        self._problem.optimizer.ingest(outcomes)

        # Continue if appropriate
        if self._continuous and self._state.current_iteration < self._state.max_iterations:
            logger.info("Continuing to next iteration")
            self._submit_next()
        else:
            self._state.finished = True
            self._client.stop_listener()
            logger.info(f"Optimization complete after {self._state.current_iteration} iterations")
