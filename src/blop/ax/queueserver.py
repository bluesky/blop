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
from typing import Any

from bluesky.callbacks import CallbackBase
from bluesky.callbacks.zmq import RemoteDispatcher
from bluesky_queueserver_api import BPlan
from bluesky_queueserver_api.zmq import REManagerAPI
from event_model import RunStart, RunStop

from ..protocols import OptimizationProblem

logger = logging.getLogger("blop")


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
        self._start_doc_cache = doc

    def stop(self, doc: RunStop) -> None:
        if self._callback is not None and self._start_doc_cache is not None:
            self._callback(self._start_doc_cache, doc)
        self._start_doc_cache = None


class QServerClient:
    """
    Handles communication with a Bluesky queueserver.

    This class encapsulates all ZMQ and HTTP communication with the queueserver,
    including plan submission and event listening.

    Parameters
    ----------
    control_addr : str
        ZMQ address for queueserver control (e.g., "tcp://localhost:60615").
    info_addr : str
        ZMQ address for queueserver info (e.g., "tcp://localhost:60625").
    zmq_consumer_addr : str
        Address for ZMQ document consumer (e.g., "localhost:5578").
    """

    def __init__(
        self,
        control_addr: str = "tcp://localhost:60615",
        info_addr: str = "tcp://localhost:60625",
        zmq_consumer_addr: str = "localhost:5578",
    ):
        self._control_addr = control_addr
        self._info_addr = info_addr
        self._zmq_consumer_addr = zmq_consumer_addr

        self._rm = REManagerAPI(zmq_control_addr=control_addr, zmq_info_addr=info_addr)
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
        autostart : bool
            If True, start the queue after adding the plan.
        timeout : float
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
    current_trials: list[dict] = field(default_factory=list)
    current_uid: str | None = None
    finished: bool = False


class QServerOptimizationRunner:
    """
    Runs optimization loops through a Bluesky queueserver.

    This class coordinates the optimization workflow by getting suggestions from
    the optimizer, submitting acquisition plans to the queueserver, and ingesting
    results when plans complete.

    Parameters
    ----------
    optimization_problem : OptimizationProblem
        The optimization problem to solve, containing the optimizer, actuators,
        sensors, and evaluation function.
    qserver_client : QServerClient
        Client for communicating with the queueserver.
    acquisition_plan_name : str
        Name of the acquisition plan registered in the queueserver.

    Examples
    --------
    >>> from blop.protocols import OptimizationProblem
    >>> from blop.ax import AxOptimizer
    >>>
    >>> # Create optimization problem
    >>> problem = OptimizationProblem(
    ...     optimizer=optimizer,
    ...     actuators=[motor1, motor2],
    ...     sensors=[detector],
    ...     evaluation_function=my_eval_func,
    ... )
    >>>
    >>> # Create qserver client and runner
    >>> client = QServerClient()
    >>> runner = QServerOptimizationRunner(problem, client, "my_acquire_plan")
    >>>
    >>> # Run optimization
    >>> runner.run(iterations=10, num_points=1)
    """

    def __init__(
        self,
        optimization_problem: OptimizationProblem,
        qserver_client: QServerClient,
        acquisition_plan_name: str = "acquire",
    ):
        self._problem = optimization_problem
        self._client = qserver_client
        self._plan_name = acquisition_plan_name
        self._state: _OptimizationState | None = None
        self._continuous = True
        self._autostart = True

    @property
    def optimization_problem(self) -> OptimizationProblem:
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
        actuator_names = [a.name for a in self._problem.actuators]
        sensor_names = [s.name for s in self._problem.sensors]
        self._client.check_devices_available(actuator_names + sensor_names)

        self._client.check_plan_available(self._plan_name)

    def _submit_next(self) -> None:
        """Get suggestions from optimizer and submit plan to queueserver."""
        if self._state is None:
            raise RuntimeError("_submit_next called before run()")
        self._state.current_iteration += 1
        self._state.current_trials = self._problem.optimizer.suggest(self._state.num_points)
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
            "agent_suggestion_uid": self._state.current_uid,
            "blop_suggestions": self._state.current_trials,
        }

        # Convert trials list to dict format expected by the plan
        # The plan expects {trial_index: parameterization}
        trials_dict = {trial["_id"]: {k: v for k, v in trial.items() if k != "_id"} for trial in self._state.current_trials}

        # Get device names
        actuator_names = [a.name for a in self._problem.actuators]
        sensor_names = [s.name for s in self._problem.sensors]

        return BPlan(
            self._plan_name,
            readables=sensor_names,
            dofs=actuator_names,
            trials=trials_dict,
            md=md,
        )

    def _on_acquisition_complete(self, start_doc: RunStart, stop_doc: RunStop) -> None:
        """Callback when acquisition finishes. Ingest results and maybe continue."""
        if self._state is None:
            raise RuntimeError("_on_acquisition_complete called before run()")
        if self._state.current_uid is None:
            raise RuntimeError("current_uid not set")
        logger.info(f"Acquisition complete for uid: {self._state.current_uid}")

        # Evaluate the results
        outcomes = self._problem.evaluation_function(
            uid=self._state.current_uid,
            suggestions=self._state.current_trials,
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
