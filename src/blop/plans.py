"""Bluesky plans for optimization."""

import logging
import threading
import time
from collections.abc import Mapping, Sequence
from importlib import import_module
from typing import Any, Literal, cast

import bluesky.plan_stubs as bps
import bluesky.plans as bp
import bluesky.preprocessors as bpp
from bluesky.callbacks import CallbackBase
from bluesky.protocols import Readable
from bluesky.utils import MsgGenerator, plan
from event_model import Event

from .plan_stubs import read_step
from .protocols import (
    ID_KEY,
    Actuator,
    CanRegisterSuggestions,
    EvaluationFunction,
    OptimizationProblem,
    Optimizer,
    RunDataReference,
    RunEvaluationFunction,
    Sensor,
    SupportsStoppingCriteria,
    TrialFaultAware,
)
from .utils import InferredReadable, _maybe_checkpoint, collect_optimization_metadata, route_suggestions

logger = logging.getLogger(__name__)

_DEFAULT_ACQUIRE_RUN_KEY: Literal["default_acquire"] = "default_acquire"
SAMPLE_SUGGESTIONS_RUN_KEY: Literal["sample_suggestions"] = "sample_suggestions"
OPTIMIZE_RUN_KEY: Literal["optimize"] = "optimize"
OPTIMIZE_IN_RUN_KEY: Literal["optimize_in_run"] = "optimize_in_run"
OPTIMIZE_TILED_STREAM_KEY: Literal["optimize_tiled_stream"] = "optimize_tiled_stream"
OPTIMIZE_IN_RUN_TRACKING_STREAM: Literal["optimization"] = "optimization"


def _unpack_for_list_scan(suggestions: list[dict], actuators: Sequence[Actuator]) -> list[Any]:
    """Unpack the actuators and inputs into Bluesky list_scan plan arguments."""
    actuators_and_inputs = {actuator: [suggestion[actuator.name] for suggestion in suggestions] for actuator in actuators}
    unpacked_list = []
    for actuator, values in actuators_and_inputs.items():
        unpacked_list.append(actuator)
        unpacked_list.append(values)

    return unpacked_list


@plan
def default_acquire(
    suggestions: list[dict],
    actuators: Sequence[Actuator],
    sensors: Sequence[Sensor] | None = None,
    *,
    per_step: bp.PerStep | None = None,
    **kwargs: Any,
) -> MsgGenerator[str]:
    """
    Acquire data for optimization. Simply a list scan.

    Includes a default metadata key "blop_suggestion_ids" which can be used to identify
    the suggestions that were acquired for each step of the scan.

    Parameters
    ----------
    suggestions: list[dict]
        A list of dictionaries, each containing the parameterization of a point to evaluate.
        The "_id" key is optional and can be used to identify each suggestion. It is suggested
        to add "_id" values to the run metadata for later identification of the acquired data.
    actuators: Sequence[Actuator]
        The actuators to move and the inputs to move them to.
    sensors: Sequence[Sensor]
        The sensors that produce data to evaluate.
    per_step: bp.PerStep | None, optional
        The plan to execute for each step of the scan.
    **kwargs: Any
        Additional keyword arguments to pass to the list_scan plan.

    Returns
    -------
    str
        The UID of the Bluesky run.

    See Also
    --------
    bluesky.plans.list_scan : The Bluesky plan to acquire data.
    """
    if sensors is None:
        sensors = []
    readables = [s for s in sensors if isinstance(s, Readable)]
    if len(readables) != len(sensors):
        logger.warning(f"Some sensors are not readable and will be ignored. Using only the readable sensors: {readables}")

    if len(suggestions) > 1:
        if all(isinstance(actuator, Readable) for actuator in actuators):
            current_position = yield from seq_read(cast(Sequence[Readable], actuators))
        else:
            current_position = None
        suggestions = route_suggestions(suggestions, starting_position=current_position)

    md = {"blop_suggestions": suggestions, "run_key": _DEFAULT_ACQUIRE_RUN_KEY}
    plan_args = _unpack_for_list_scan(suggestions, actuators)
    return (
        # TODO: fix argument type in bluesky.plans.list_scan
        yield from bpp.set_run_key_wrapper(
            bp.list_scan(
                readables,
                *plan_args,  # type: ignore[arg-type]
                per_step=per_step,
                md=md,
                **kwargs,
            ),
            _DEFAULT_ACQUIRE_RUN_KEY,
        )
    )


def _validate_suggestions_have_ids(suggestions: list[dict]) -> None:
    """Ensure every suggestion can be matched to an evaluated outcome."""
    if any(ID_KEY not in suggestion for suggestion in suggestions):
        raise ValueError(
            f"All suggestions must contain an '{ID_KEY}' key to later match with the outcomes. Please review your "
            f"optimizer implementation. Got suggestions: {suggestions}"
        )


def _validate_outcomes_have_ids(outcomes: list[dict], suggestions: list[dict]) -> None:
    """Ensure every outcome can be matched to an optimizer suggestion."""
    if any(ID_KEY not in outcome for outcome in outcomes):
        raise ValueError(
            f"All outcomes must contain an '{ID_KEY}' key that matches with the suggestions. Please review your "
            f"evaluation function. Got suggestions: {suggestions} and outcomes: {outcomes}"
        )


def _make_tiled_writer(tiled_client: Any, batch_size: int) -> Any:
    """Create a TiledWriter without making Tiled a package import dependency."""
    try:
        module = import_module("bluesky_tiled_plugins")
    except ImportError as err:
        raise ImportError(
            "optimize_tiled_stream requires tiled and bluesky-tiled-plugins; run it in the pixi docs environment or "
            "install those packages."
        ) from err
    return module.TiledWriter(tiled_client, batch_size=batch_size)


@plan
def optimize_step(
    optimization_problem: OptimizationProblem,
    n_points: int = 1,
    *args: Any,
    **kwargs: Any,
) -> MsgGenerator[tuple[str, list[dict], list[dict]]]:
    """
    Single step of the optimization loop.

    Parameters
    ----------
    optimization_problem : OptimizationProblem
        The optimization problem to solve.
    n_points : int, optional
        The number of points to suggest.

    Returns
    -------
    tuple[str, list[dict], list[dict]]
        A tuple containing the uid, suggestions, and outcomes of the step.
    """
    if optimization_problem.acquisition_plan is None:
        acquisition_plan = default_acquire
    else:
        acquisition_plan = optimization_problem.acquisition_plan
    optimizer = optimization_problem.optimizer
    actuators = optimization_problem.actuators
    suggestions = optimizer.suggest(n_points)
    _validate_suggestions_have_ids(suggestions)
    try:
        uid = yield from acquisition_plan(suggestions, actuators, optimization_problem.sensors, *args, **kwargs)
    except Exception:
        if isinstance(optimizer, TrialFaultAware):
            optimizer.register_failures(suggestions)
        raise

    evaluation_function: EvaluationFunction = optimization_problem.evaluation_function
    outcomes = evaluation_function(uid, suggestions)
    _validate_outcomes_have_ids(outcomes, suggestions)
    optimizer.ingest(outcomes)

    return uid, suggestions, outcomes


class _InRunEvaluationCallback(CallbackBase):
    """Evaluate in-run acquisition events when an optimization batch is complete."""

    def __init__(
        self, evaluation_function: RunEvaluationFunction, optimizer: Optimizer, n_points: int, primary_stream: str
    ) -> None:
        self._evaluation_function = evaluation_function
        self._optimizer = optimizer
        self._n_points = n_points
        self._primary_stream = primary_stream
        self._start_doc: Mapping[str, Any] | None = None
        self._descriptors: dict[str, Mapping[str, Any]] = {}
        self._active = False
        self._suggestions: list[dict] = []
        self._documents: list[tuple[str, Mapping[str, Any]]] = []
        self._events: list[Mapping[str, Any]] = []
        self._primary_event_count = 0
        self._outcomes: list[dict] | None = None
        self._reference: RunDataReference | None = None
        self._exception: BaseException | None = None

    def begin(self, suggestions: list[dict]) -> None:
        """Start collecting documents for a new in-run evaluation batch."""
        self._active = True
        self._suggestions = suggestions
        self._documents = []
        self._events = []
        self._primary_event_count = 0
        self._outcomes = None
        self._reference = None
        self._exception = None

    def complete(self) -> tuple[RunDataReference, list[dict]]:
        """Return the evaluated batch or raise the stored acquisition/evaluation error."""
        if self._exception is not None:
            raise self._exception
        if self._reference is None or self._outcomes is None:
            self._active = False
            raise RuntimeError(
                f"In-run acquisition produced {self._primary_event_count} {self._primary_stream!r} event documents, "
                f"but n_points={self._n_points} were required for evaluation."
            )
        self._active = False
        return self._reference, self._outcomes

    @property
    def evaluated(self) -> bool:
        """Whether the active batch has already been evaluated."""
        return self._outcomes is not None

    @property
    def failed(self) -> bool:
        """Whether callback evaluation failed and stored an exception."""
        return self._exception is not None

    def start(self, doc: Mapping[str, Any]) -> None:
        """Store the first enclosing optimization start document."""
        if self._start_doc is None:
            self._start_doc = dict(doc)

    def descriptor(self, doc: Mapping[str, Any]) -> None:
        """Store every descriptor and keep active-batch descriptor documents."""
        copied_doc = dict(doc)
        self._descriptors[str(copied_doc["uid"])] = copied_doc
        if self._active:
            self._documents.append(("descriptor", copied_doc))

    def event(self, doc: Event) -> Event:
        """Collect active-batch events and evaluate at the configured event offset."""
        if not self._active or self._outcomes is not None:
            return doc

        copied_doc = dict(doc)
        self._events.append(copied_doc)
        self._documents.append(("event", copied_doc))
        try:
            stream_name = self._stream_name(copied_doc)
        except BaseException as err:
            self._exception = err
            self._active = False
            return doc
        if stream_name != self._primary_stream:
            return doc

        self._primary_event_count += 1
        if self._primary_event_count != self._n_points:
            return doc

        try:
            reference = self._build_reference()
            outcomes = self._evaluation_function(reference, self._suggestions)
            _validate_outcomes_have_ids(outcomes, self._suggestions)
            self._optimizer.ingest(outcomes)
        except BaseException as err:
            self._exception = err
            self._active = False
            return doc

        self._reference = reference
        self._outcomes = outcomes
        self._active = False
        return doc

    def _build_reference(self) -> RunDataReference:
        if self._start_doc is None:
            raise RuntimeError(
                "In-run evaluation cannot build a reference before the optimization run start document is observed."
            )

        events = tuple(self._events)
        stream_bounds: dict[str, tuple[int, int]] = {}
        for event in events:
            descriptor = self._descriptors[str(event["descriptor"])]
            stream_name = str(descriptor["name"])
            start = int(event["seq_num"]) - 1
            stop = int(event["seq_num"])
            if stream_name in stream_bounds:
                previous_start, previous_stop = stream_bounds[stream_name]
                stream_bounds[stream_name] = (min(previous_start, start), max(previous_stop, stop))
            else:
                stream_bounds[stream_name] = (start, stop)

        return RunDataReference(
            run_uid=str(self._start_doc["uid"]),
            start_doc=self._start_doc,
            descriptors=dict(self._descriptors),
            events=events,
            documents=tuple(self._documents),
            stream_slices={stream_name: slice(start, stop) for stream_name, (start, stop) in stream_bounds.items()},
            primary_stream=self._primary_stream,
            source="callback",
            resolver=None,
        )

    def _stream_name(self, event: Mapping[str, Any]) -> str:
        descriptor = self._descriptors[str(event["descriptor"])]
        return str(descriptor["name"])


class _TiledRunResolver:
    """Resolve live run data from a Tiled client."""

    def __init__(self, tiled_client: Any, run_uid: str, timeout: float, poll_interval: float) -> None:
        self._tiled_client = tiled_client
        self.run_uid = run_uid
        self.timeout = timeout
        self.poll_interval = poll_interval

    def wait_for_run(self) -> Any:
        """Wait for the run container to appear in Tiled."""
        deadline = time.monotonic() + self.timeout
        return self._wait_for_child(
            self._tiled_client,
            self.run_uid,
            f"Tiled run {self.run_uid!r} did not appear within {self.timeout}s.",
            deadline,
        )

    def wait_for_stream(self, stream: str) -> Any:
        """Wait for a stream container to appear in the run."""
        deadline = time.monotonic() + self.timeout
        run = self._wait_for_run_until(deadline)
        return self._wait_for_child(
            run,
            stream,
            f"Tiled stream {self.run_uid!r}/{stream!r} did not appear within {self.timeout}s.",
            deadline,
        )

    def wait_for_table_rows(self, stream: str, rows: int) -> Any:
        """Wait for a stream's internal table to reach ``rows`` rows."""
        deadline = time.monotonic() + self.timeout
        timeout_message = (
            f"Tiled stream {self.run_uid!r}/{stream!r}/internal did not reach {rows} rows within {self.timeout}s."
        )
        run = self._wait_for_run_until(deadline)
        table_client = self._wait_for_child(run, f"{stream}/internal", timeout_message, deadline)
        wake_event = threading.Event()
        subscription = self._start_data_subscription(table_client, wake_event)
        try:
            while True:
                table = self._materialize(table_client.read())
                if len(table) >= rows:
                    return table
                self._wait_or_timeout(wake_event, deadline, timeout_message)
        finally:
            self._disconnect(subscription)

    def count_table_rows(self, stream: str) -> int:
        """Return the current number of rows in a stream's internal table."""
        try:
            run = self._tiled_client[self.run_uid]
            table_client = run[f"{stream}/internal"]
        except KeyError:
            return 0
        table = self._materialize(table_client.read())
        return len(table)

    def read(self, path: str, *, stream_slice: slice | None = None) -> Any:
        """Read and materialize a run-relative Tiled path."""
        deadline = time.monotonic() + self.timeout
        run = self._wait_for_run_until(deadline)
        node = self._wait_for_child(
            run,
            path,
            f"Tiled path {self.run_uid!r}/{path!r} did not appear within {self.timeout}s.",
            deadline,
        )
        data = node.read()
        if stream_slice is None:
            return self._materialize(data)
        try:
            return self._materialize(data[stream_slice])
        except (TypeError, ValueError, NotImplementedError, AttributeError):
            materialized = self._materialize(data)
            if hasattr(materialized, "iloc"):
                return materialized.iloc[stream_slice]
            return materialized[stream_slice]

    def _wait_for_run_until(self, deadline: float) -> Any:
        return self._wait_for_child(
            self._tiled_client,
            self.run_uid,
            f"Tiled run {self.run_uid!r} did not appear within {self.timeout}s.",
            deadline,
        )

    def _wait_for_child(self, container: Any, key: str, timeout_message: str, deadline: float) -> Any:
        wake_event = threading.Event()
        callback = self._make_child_callback(key, wake_event)
        subscription = self._start_child_subscription(container, callback)
        try:
            while True:
                try:
                    return container[key]
                except KeyError:
                    self._wait_or_timeout(wake_event, deadline, timeout_message)
                except TypeError as err:
                    if not self._is_unready_tiled_structure(err):
                        raise
                    self._wait_or_timeout(wake_event, deadline, timeout_message)
        finally:
            self._disconnect(subscription)

    def _make_child_callback(self, key: str, wake_event: threading.Event) -> Any:
        def _callback(update: Any) -> None:
            if getattr(update, "key", None) == key:
                wake_event.set()

        return _callback

    def _start_child_subscription(self, container: Any, callback: Any) -> Any | None:
        subscribe = getattr(container, "subscribe", None)
        if subscribe is None:
            return None
        subscription = subscribe()
        child_created = getattr(subscription, "child_created", None)
        if child_created is not None:
            child_created.add_callback(callback)
        subscription.start_in_thread(start=0)
        return subscription

    def _start_data_subscription(self, node: Any, wake_event: threading.Event) -> Any | None:
        subscribe = getattr(node, "subscribe", None)
        if subscribe is None:
            return None
        subscription = subscribe()
        new_data = getattr(subscription, "new_data", None)

        def callback(update: Any) -> None:
            wake_event.set()

        if new_data is not None:
            new_data.add_callback(callback)
        subscription.start_in_thread(start=0)
        return subscription, callback

    def _wait_or_timeout(self, wake_event: threading.Event, deadline: float, timeout_message: str) -> None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(timeout_message)
        wake_event.wait(min(self.poll_interval, remaining))
        wake_event.clear()

    def _disconnect(self, subscription: Any | None) -> None:
        if subscription is not None:
            target = subscription[0] if isinstance(subscription, tuple) else subscription
            target.disconnect()

    def _is_unready_tiled_structure(self, err: TypeError) -> bool:
        return "argument after ** must be a mapping, not NoneType" in str(err)

    def _materialize(self, data: Any) -> Any:
        if hasattr(data, "compute"):
            return data.compute()
        return data


class _TiledStreamingEvaluationBatch:
    """Evaluate one optimizer batch when Tiled shows enough primary-stream rows."""

    def __init__(
        self,
        evaluation_function: RunEvaluationFunction,
        optimizer: Optimizer,
        resolver: _TiledRunResolver,
        suggestions: list[dict],
        primary_stream: str,
        stream_slice: slice,
    ) -> None:
        if stream_slice.stop is None:
            raise ValueError("stream_slice.stop must be set for Tiled streaming evaluation.")
        self._evaluation_function = evaluation_function
        self._optimizer = optimizer
        self._resolver = resolver
        self._suggestions = suggestions
        self._primary_stream = primary_stream
        self._stream_slice = stream_slice
        self._required_rows = stream_slice.stop
        self._done = threading.Event()
        self._cancelled = threading.Event()
        self._thread: threading.Thread | None = None
        self._result: tuple[RunDataReference, list[dict]] | None = None
        self._exception: BaseException | None = None
        self._evaluated = False

    def start(self) -> None:
        """Start background evaluation."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def complete(self) -> tuple[RunDataReference, list[dict]]:
        """Return the evaluated batch or raise the stored error."""
        self._done.wait(self._resolver.timeout)
        if self._exception is not None:
            raise self._exception
        if self._result is None:
            raise TimeoutError(
                f"Tiled evaluation did not complete within {self._resolver.timeout}s for run "
                f"{self._resolver.run_uid!r} and stream slice {self._stream_slice!r}."
            )
        return self._result

    def cancel(self) -> None:
        """Prevent evaluation or ingestion if the batch has not already run."""
        self._cancelled.set()

    @property
    def evaluated(self) -> bool:
        """Whether optimizer ingestion succeeded."""
        return self._evaluated

    @property
    def failed(self) -> bool:
        """Whether background evaluation stored an exception."""
        return self._exception is not None

    def _run(self) -> None:
        try:
            self._resolver.wait_for_table_rows(self._primary_stream, self._required_rows)
            if self._cancelled.is_set():
                return
            run = self._resolver.wait_for_run()
            reference = RunDataReference(
                run_uid=self._resolver.run_uid,
                start_doc=dict(run.metadata["start"]),
                descriptors={},
                events=(),
                documents=(),
                stream_slices={self._primary_stream: self._stream_slice},
                primary_stream=self._primary_stream,
                source="tiled",
                resolver=self._resolver,
            )
            if self._cancelled.is_set():
                return
            outcomes = self._evaluation_function(reference, self._suggestions)
            _validate_outcomes_have_ids(outcomes, self._suggestions)
            if self._cancelled.is_set():
                return
            self._optimizer.ingest(outcomes)
        except BaseException as err:
            self._exception = err
        else:
            self._result = (reference, outcomes)
            self._evaluated = True
        finally:
            self._done.set()


@plan
def optimize(
    optimization_problem: OptimizationProblem,
    iterations: int = 1,
    n_points: int = 1,
    checkpoint_interval: int | None = None,
    readable_cache: dict[str, InferredReadable] | None = None,
    **kwargs: Any,
) -> MsgGenerator[None]:
    """
    Solve the optimization problem.

    Parameters
    ----------
    optimization_problem : OptimizationProblem
        The optimization problem to solve.
    iterations : int, optional
        The number of optimization iterations to run.
    n_points : int, optional
        The number of points to suggest per iteration.
    checkpoint_interval : int | None, optional
        The number of iterations between optimizer checkpoints. If None, checkpoints
        will not be saved. Optimizer must implement the
        :class:`blop.protocols.Checkpointable` protocol.
    readable_cache: dict[str, InferredReadable] | None = None
        Cache of readable objects to store the suggestions and outcomes as events.
        If None, a new cache will be created.
    **kwargs : Any
        Additional keyword arguments to pass to the :func:`optimize_step` plan.

    See Also
    --------
    blop.protocols.OptimizationProblem : The problem to solve.
    blop.protocols.Checkpointable : The protocol for checkpointable objects.
    optimize_step : The plan to execute a single step of the optimization.
    """
    # Cache to track readables created from suggestions and outcomes
    readable_cache = readable_cache or {}

    _md = collect_optimization_metadata(optimization_problem)
    _md.update(
        {
            "plan_name": "optimize",
            "iterations": iterations,
            "n_points": n_points,
            "checkpoint_interval": checkpoint_interval,
            "run_key": OPTIMIZE_RUN_KEY,
        }
    )

    # Encapsulate the optimization plan in a run decorator
    @bpp.set_run_key_decorator(OPTIMIZE_RUN_KEY)
    @bpp.run_decorator(md=_md)
    def _optimize() -> MsgGenerator[None]:
        for i in range(iterations):
            # Perform a single step of the optimization
            uid, suggestions, outcomes = yield from optimize_step(optimization_problem, n_points, **kwargs)

            if isinstance(optimization_problem.optimizer, SupportsStoppingCriteria):
                stop_now, stop_reason = optimization_problem.optimizer.should_stop()
                if stop_now:
                    reason = stop_reason if stop_reason is not None else "No reason provided"
                    logger.info(f"Global stopping triggered at iteration {i + 1}: {reason}")
                    return

            # Read the optimization step into the Bluesky and emit events for each suggestion and outcome
            yield from read_step(uid, suggestions, outcomes, n_points, readable_cache)

            # Possibly take a checkpoint of the optimizer state
            _maybe_checkpoint(optimization_problem.optimizer, checkpoint_interval, i)

    # Start the optimization run
    return (yield from _optimize())


@plan
def optimize_in_run(
    optimization_problem: OptimizationProblem,
    evaluation_function: RunEvaluationFunction,
    iterations: int = 1,
    n_points: int = 1,
    checkpoint_interval: int | None = None,
    primary_stream: str = "primary",
    readable_cache: dict[str, InferredReadable] | None = None,
    **kwargs: Any,
) -> MsgGenerator[None]:
    """Solve an optimization problem by evaluating acquisition documents inside one run.

    Parameters
    ----------
    optimization_problem : OptimizationProblem
        The optimization problem to solve.
    evaluation_function : RunEvaluationFunction
        Callable that transforms run data and suggestions into outcomes.
    iterations : int, optional
        The number of optimization iterations to run.
    n_points : int, optional
        The number of primary-stream acquisition events required before evaluating each iteration.
    checkpoint_interval : int | None, optional
        The number of iterations between optimizer checkpoints. If None, checkpoints
        will not be saved. Optimizer must implement the
        :class:`blop.protocols.Checkpointable` protocol.
    primary_stream : str, optional
        Acquisition stream name whose events count toward ``n_points``.
    readable_cache: dict[str, InferredReadable] | None = None
        Cache of readable objects to store the suggestions and outcomes as optimization events.
        If None, a new cache will be created.
    **kwargs : Any
        Additional keyword arguments to pass to the acquisition plan.
    """
    _md = collect_optimization_metadata(optimization_problem)
    _md.update(
        {
            "plan_name": "optimize_in_run",
            "iterations": iterations,
            "n_points": n_points,
            "checkpoint_interval": checkpoint_interval,
            "run_key": OPTIMIZE_IN_RUN_KEY,
            "primary_stream": primary_stream,
            "optimization_stream": OPTIMIZE_IN_RUN_TRACKING_STREAM,
        }
    )
    readable_cache = readable_cache or {}

    optimizer = optimization_problem.optimizer
    actuators = optimization_problem.actuators
    acquisition_plan = optimization_problem.acquisition_plan or default_acquire
    callback = _InRunEvaluationCallback(evaluation_function, optimizer, n_points, primary_stream)

    def _use_optimize_in_run_key(msg: Any) -> Any:
        if msg.run is None:
            return msg
        return msg._replace(run=OPTIMIZE_IN_RUN_KEY)

    @bpp.set_run_key_decorator(OPTIMIZE_IN_RUN_KEY)
    @bpp.run_decorator(md=_md)
    def _optimize_in_run() -> MsgGenerator[None]:
        for i in range(iterations):
            suggestions = optimizer.suggest(n_points)
            _validate_suggestions_have_ids(suggestions)
            callback.begin(suggestions)
            try:
                yield from bpp.msg_mutator(
                    bpp.stub_wrapper(acquisition_plan(suggestions, actuators, optimization_problem.sensors, **kwargs)),
                    _use_optimize_in_run_key,
                )
            except Exception:
                if callback.failed:
                    callback.complete()
                if callback.evaluated:
                    raise
                if isinstance(optimizer, TrialFaultAware):
                    optimizer.register_failures(suggestions)
                raise

            reference, outcomes = callback.complete()

            yield from read_step(
                reference.run_uid,
                suggestions,
                outcomes,
                n_points,
                readable_cache,
                stream_name=OPTIMIZE_IN_RUN_TRACKING_STREAM,
            )

            if isinstance(optimizer, SupportsStoppingCriteria):
                stop_now, stop_reason = optimizer.should_stop()
                if stop_now:
                    reason = stop_reason if stop_reason is not None else "No reason provided"
                    logger.info(f"Global stopping triggered at iteration {i + 1}: {reason}")
                    return

            _maybe_checkpoint(optimizer, checkpoint_interval, i)

    return (yield from bpp.subs_wrapper(_optimize_in_run(), callback))


@plan
def optimize_tiled_stream(
    optimization_problem: OptimizationProblem,
    evaluation_function: RunEvaluationFunction,
    tiled_client: Any,
    iterations: int = 1,
    n_points: int = 1,
    checkpoint_interval: int | None = None,
    primary_stream: str = "primary",
    optimization_stream: str = "optimization",
    readable_cache: dict[str, InferredReadable] | None = None,
    tiled_writer: Any | None = None,
    timeout: float = 30.0,
    poll_interval: float = 0.05,
    **kwargs: Any,
) -> MsgGenerator[None]:
    """Solve an optimization problem by evaluating live data through Tiled."""
    if primary_stream == optimization_stream:
        raise ValueError(
            "primary_stream and optimization_stream must be different. Got "
            f"primary_stream={primary_stream!r}, optimization_stream={optimization_stream!r}."
        )

    _md = collect_optimization_metadata(optimization_problem)
    _md.update(
        {
            "plan_name": "optimize_tiled_stream",
            "run_key": OPTIMIZE_TILED_STREAM_KEY,
            "iterations": iterations,
            "n_points": n_points,
            "checkpoint_interval": checkpoint_interval,
            "primary_stream": primary_stream,
            "optimization_stream": optimization_stream,
            "evaluation_source": "tiled",
        }
    )
    readable_cache = readable_cache or {}
    writer = tiled_writer if tiled_writer is not None else _make_tiled_writer(tiled_client, batch_size=1)

    optimizer = optimization_problem.optimizer
    actuators = optimization_problem.actuators
    acquisition_plan = optimization_problem.acquisition_plan or default_acquire

    def _use_optimize_tiled_stream_key(msg: Any) -> Any:
        if msg.run is None:
            return msg
        return msg._replace(run=OPTIMIZE_TILED_STREAM_KEY)

    @bpp.set_run_key_decorator(OPTIMIZE_TILED_STREAM_KEY)
    def _optimize_tiled_stream() -> MsgGenerator[None]:
        run_uid = yield from bps.open_run(md=_md)
        resolver = _TiledRunResolver(tiled_client, run_uid, timeout, poll_interval)
        try:
            for i in range(iterations):
                stream_start = resolver.count_table_rows(primary_stream)
                suggestions = optimizer.suggest(n_points)
                _validate_suggestions_have_ids(suggestions)
                stream_slice = slice(stream_start, stream_start + n_points)
                batch = _TiledStreamingEvaluationBatch(
                    evaluation_function,
                    optimizer,
                    resolver,
                    suggestions,
                    primary_stream,
                    stream_slice,
                )
                batch.start()
                try:
                    yield from bpp.msg_mutator(
                        bpp.stub_wrapper(acquisition_plan(suggestions, actuators, optimization_problem.sensors, **kwargs)),
                        _use_optimize_tiled_stream_key,
                    )
                except BaseException:
                    if batch.failed:
                        batch.complete()
                    if batch.evaluated:
                        raise
                    batch.cancel()
                    if isinstance(optimizer, TrialFaultAware):
                        optimizer.register_failures(suggestions)
                    raise

                reference, outcomes = batch.complete()

                yield from read_step(
                    reference.run_uid,
                    suggestions,
                    outcomes,
                    n_points,
                    readable_cache,
                    stream_name=optimization_stream,
                )

                if isinstance(optimizer, SupportsStoppingCriteria):
                    stop_now, stop_reason = optimizer.should_stop()
                    if stop_now:
                        reason = stop_reason if stop_reason is not None else "No reason provided"
                        logger.info(f"Global stopping triggered at iteration {i + 1}: {reason}")
                        break

                _maybe_checkpoint(optimizer, checkpoint_interval, i)
        except BaseException as err:
            yield from bps.close_run(exit_status="fail", reason=str(err))
            raise
        else:
            yield from bps.close_run()

    return (yield from bpp.subs_wrapper(_optimize_tiled_stream(), writer))


@plan
def sample_suggestions(
    optimization_problem: OptimizationProblem,
    suggestions: list[dict],
    readable_cache: dict[str, InferredReadable] | None = None,
    **kwargs: Any,
) -> MsgGenerator[tuple[str, list[dict], list[dict]]]:
    """
    Evaluate specific parameter combinations.

    This plan acquires data for given suggestions and ingests results into the optimizer.
    Supports both optimizer-generated suggestions (with "_id") and manual points
    (without "_id", if optimizer implements CanRegisterSuggestions).

    Parameters
    ----------
    optimization_problem : OptimizationProblem
        The optimization problem.
    suggestions : list[dict]
        Parameter combinations to evaluate. Can be:

        - Optimizer suggestions (with "_id" keys from suggest())
        - Manual points (without "_id", requires CanRegisterSuggestions protocol)

    readable_cache : dict[str, InferredReadable] | None
        Cache for storing suggestions/outcomes as events.
    **kwargs : Any
        Additional arguments for acquisition plan.

    Returns
    -------
    uid : str
        Bluesky run UID.
    suggestions : list[dict]
        Suggestions with "_id" keys.
    outcomes : list[dict]
        Evaluated outcomes.

    Raises
    ------
    ValueError
        If suggestions lack "_id" and optimizer doesn't implement CanRegisterSuggestions.

    See Also
    --------
    optimize_step : Standard optimizer-driven step.
    blop.protocols.CanRegisterSuggestions : Protocol for manual suggestions.
    """
    # Ensure the suggestions have an ID_KEY or register them with the optimizer
    if not isinstance(optimization_problem.optimizer, CanRegisterSuggestions) and any(
        ID_KEY not in suggestion for suggestion in suggestions
    ):
        raise ValueError(
            f"All suggestions must contain an '{ID_KEY}' key to later match with the outcomes or your optimizer must "
            "implement the `blop.protocols.CanRegisterSuggestions` protocol. Please review your optimizer "
            f"implementation. Got suggestions: {suggestions}"
        )
    elif isinstance(optimization_problem.optimizer, CanRegisterSuggestions):
        suggestions = optimization_problem.optimizer.register_suggestions(suggestions)

    # Collect the metadata for the run
    _md = collect_optimization_metadata(optimization_problem)
    _md.update(
        {
            "plan_name": "sample_suggestions",
            "suggestions": suggestions,
            "run_key": SAMPLE_SUGGESTIONS_RUN_KEY,
        }
    )

    @bpp.set_run_key_decorator(SAMPLE_SUGGESTIONS_RUN_KEY)
    @bpp.run_decorator(md=_md)
    def _inner_sample_suggestions() -> MsgGenerator[tuple[str, list[dict], list[dict]]]:

        # Acquire data, evaluate, and ingest outcomes
        if optimization_problem.acquisition_plan is None:
            acquisition_plan = default_acquire
        else:
            acquisition_plan = optimization_problem.acquisition_plan
        uid = yield from acquisition_plan(
            suggestions, optimization_problem.actuators, optimization_problem.sensors, **kwargs
        )
        outcomes = optimization_problem.evaluation_function(uid, suggestions)
        optimization_problem.optimizer.ingest(outcomes)

        # Emit a Bluesky event
        yield from read_step(uid, suggestions, outcomes, len(suggestions), readable_cache or {})

        return uid, suggestions, outcomes

    return (yield from _inner_sample_suggestions())


@plan
def seq_read(readables: Sequence[Readable], **kwargs: Any) -> MsgGenerator[dict[str, Any]]:
    """
    Read the current values of the given readables.

    Parameters
    ----------
    readables : Sequence[Readable]
        The readables to read.

    Returns
    -------
    dict[str, Any]
        A dictionary of the readable names and their current values.
    """
    results = {}
    for readable in readables:
        results[readable.name] = yield from bps.rd(readable, **kwargs)
    return results


def acquire_baseline(
    optimization_problem: OptimizationProblem,
    parameterization: dict[str, Any] | None = None,
    **kwargs: Any,
) -> MsgGenerator[None]:
    """
    Acquire a baseline reading. Useful for relative outcome constraints.

    Parameters
    ----------
    optimization_problem : OptimizationProblem
        The optimization problem to solve.
    parameterization : dict[str, Any] | None = None
        Move the DOFs to the given parameterization, if provided.

    See Also
    --------
    default_acquire : The default plan to acquire data.
    """
    actuators = optimization_problem.actuators
    if parameterization is None:
        if all(isinstance(actuator, Readable) for actuator in actuators):
            parameterization = yield from seq_read(cast(Sequence[Readable], actuators))
        else:
            raise ValueError(
                "All actuators must also implement the Readable protocol to acquire a baseline from current positions."
            )
    if ID_KEY not in parameterization:
        parameterization[ID_KEY] = "baseline"
    optimizer = optimization_problem.optimizer
    if optimization_problem.acquisition_plan is None:
        acquisition_plan = default_acquire
    else:
        acquisition_plan = optimization_problem.acquisition_plan
    uid = yield from acquisition_plan([parameterization], actuators, optimization_problem.sensors, **kwargs)
    outcome = optimization_problem.evaluation_function(uid, [parameterization])[0]
    data = {**outcome, **parameterization}
    optimizer.ingest([data])
