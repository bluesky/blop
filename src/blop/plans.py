"""Bluesky plans for optimization."""

import logging
from collections.abc import Hashable, Mapping, Sequence
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
    InRunDataReference,
    InRunEvaluationFunction,
    OptimizationProblem,
    Optimizer,
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
OPTIMIZE_IN_RUN_TRACKING_STREAM: Literal["optimization"] = "optimization"


def _unpack_for_list_scan(suggestions: Sequence[Mapping], actuators: Sequence[Actuator]) -> list[Any]:
    """Unpack the actuators and inputs into Bluesky list_scan plan arguments."""
    actuators_and_inputs = {actuator: [suggestion[actuator.name] for suggestion in suggestions] for actuator in actuators}
    unpacked_list = []
    for actuator, values in actuators_and_inputs.items():
        unpacked_list.append(actuator)
        unpacked_list.append(values)

    return unpacked_list


@plan
def default_acquire(
    suggestions: Sequence[Mapping],
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
    suggestions: Sequence[Mapping]
        A sequence of mappings, each containing the parameterization of a point to evaluate.
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


@plan
def optimize_step(
    optimization_problem: OptimizationProblem,
    n_points: int = 1,
    *args: Any,
    **kwargs: Any,
) -> MsgGenerator[tuple[Hashable, Sequence[Mapping], Sequence[Mapping]]]:
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
    tuple[Hashable, Sequence[Mapping], Sequence[Mapping]]
        The acquisition identifier, suggestions, and outcomes of the step.
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
        self, evaluation_function: InRunEvaluationFunction, optimizer: Optimizer, n_points: int, primary_stream: str
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
        self._reference: InRunDataReference | None = None
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

    def complete(self) -> tuple[InRunDataReference, list[dict]]:
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

    def _build_reference(self) -> InRunDataReference:
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

        return InRunDataReference(
            run_uid=str(self._start_doc["uid"]),
            start_doc=self._start_doc,
            descriptors=dict(self._descriptors),
            events=events,
            documents=tuple(self._documents),
            stream_slices={stream_name: slice(start, stop) for stream_name, (start, stop) in stream_bounds.items()},
        )

    def _stream_name(self, event: Mapping[str, Any]) -> str:
        descriptor = self._descriptors[str(event["descriptor"])]
        return str(descriptor["name"])


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
    evaluation_function: InRunEvaluationFunction,
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
    evaluation_function : InRunEvaluationFunction
        Callable that transforms in-run Bluesky documents and suggestions into outcomes.
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
def sample_suggestions(
    optimization_problem: OptimizationProblem,
    suggestions: Sequence[Mapping],
    readable_cache: dict[str, InferredReadable] | None = None,
    **kwargs: Any,
) -> MsgGenerator[tuple[Hashable, Sequence[Mapping], Sequence[Mapping]]]:
    """
    Evaluate specific parameter combinations.

    This plan acquires data for given suggestions and ingests results into the optimizer.
    Supports both optimizer-generated suggestions (with "_id") and manual points
    (without "_id", if optimizer implements CanRegisterSuggestions).

    Parameters
    ----------
    optimization_problem : OptimizationProblem
        The optimization problem.
    suggestions : Sequence[Mapping]
        Parameter combinations to evaluate. Can be:

        - Optimizer suggestions (with "_id" keys from suggest())
        - Manual points (without "_id", requires CanRegisterSuggestions protocol)

    readable_cache : dict[str, InferredReadable] | None
        Cache for storing suggestions/outcomes as events.
    **kwargs : Any
        Additional arguments for acquisition plan.

    Returns
    -------
    uid : Hashable
        The acquisition identifier returned by the acquisition plan.
    suggestions : Sequence[Mapping]
        Suggestions with "_id" keys.
    outcomes : Sequence[Mapping]
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
    def _inner_sample_suggestions() -> MsgGenerator[tuple[Hashable, Sequence[Mapping], Sequence[Mapping]]]:

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
    parameterization: Mapping[str, Any] | None = None,
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
    parameterization_copy = dict(parameterization)
    if ID_KEY not in parameterization:
        parameterization_copy[ID_KEY] = "baseline"
    optimizer = optimization_problem.optimizer
    if optimization_problem.acquisition_plan is None:
        acquisition_plan = default_acquire
    else:
        acquisition_plan = optimization_problem.acquisition_plan
    uid = yield from acquisition_plan([parameterization_copy], actuators, optimization_problem.sensors, **kwargs)
    outcome = optimization_problem.evaluation_function(uid, [parameterization_copy])[0]
    data = {**outcome, **parameterization_copy}
    optimizer.ingest([data])
