"""Bluesky plans for optimization."""

import logging
from collections.abc import Hashable, Mapping, Sequence
from typing import Any, Literal, cast

import bluesky.plan_stubs as bps
import bluesky.plans as bp
import bluesky.preprocessors as bpp
from bluesky.protocols import Readable
from bluesky.utils import MsgGenerator, plan

from .plan_stubs import read_step
from .protocols import (
    ID_KEY,
    Actuator,
    CanRegisterSuggestions,
    EvaluationFunction,
    OptimizationProblem,
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


def _validate_ids_are_unique_and_hashable(records: Sequence[Mapping], label: str) -> None:
    """Ensure record IDs can be used as opaque identity values."""
    record_ids = []
    for record in records:
        record_id = record[ID_KEY]
        try:
            hash(record_id)
        except TypeError as err:
            raise TypeError(f"All {label} must contain hashable '{ID_KEY}' values. Got {record_id!r}.") from err
        record_ids.append(record_id)
    if len(record_ids) != len(set(record_ids)):
        raise ValueError(f"All {label} must contain unique '{ID_KEY}' values. Got {record_ids!r}.")


def _validate_suggestions(suggestions: Sequence[Mapping]) -> None:
    """Ensure every suggestion can be matched to an evaluated outcome."""
    if any(ID_KEY not in suggestion for suggestion in suggestions):
        raise ValueError(
            f"All suggestions must contain an '{ID_KEY}' key to later match with the outcomes. Please review your "
            f"optimizer implementation. Got suggestions: {suggestions}"
        )
    _validate_ids_are_unique_and_hashable(suggestions, "suggestions")


def _validate_outcomes(outcomes: Sequence[Mapping], suggestions: Sequence[Mapping]) -> None:
    """Ensure every outcome can be matched to an optimizer suggestion."""
    if any(ID_KEY not in outcome for outcome in outcomes):
        raise ValueError(
            f"All outcomes must contain an '{ID_KEY}' key that matches with the suggestions. Please review your "
            f"evaluation function. Got suggestions: {suggestions} and outcomes: {outcomes}"
        )
    _validate_ids_are_unique_and_hashable(outcomes, "outcomes")
    suggestion_ids = {suggestion[ID_KEY] for suggestion in suggestions}
    outcome_ids = {outcome[ID_KEY] for outcome in outcomes}
    if suggestion_ids != outcome_ids:
        raise ValueError(
            "The suggestions and outcomes must contain the same IDs. Got suggestions: "
            f"{suggestion_ids} and outcomes: {outcome_ids}"
        )


def _suggestion_ids(suggestions: Sequence[Mapping]) -> tuple[Hashable, ...]:
    """Return suggestion IDs as a hashable acquisition identifier."""
    return tuple(cast(Hashable, suggestion[ID_KEY]) for suggestion in suggestions)


def _drop_run_control_messages(plan: MsgGenerator[Hashable]) -> MsgGenerator[Hashable]:
    """Drop child-run messages while preserving hardware lifecycle messages."""

    def _drop(msg: Any) -> Any | None:
        if msg.command in {"open_run", "close_run"}:
            return None
        return msg

    return (yield from bpp.msg_mutator(plan, _drop))


def _reject_child_run_messages(plan: MsgGenerator[Hashable]) -> MsgGenerator[Hashable]:
    """Reject child-run messages inside the optimize_in_run acquisition step."""

    def _reject(msg: Any) -> Any:
        if msg.command in {"open_run", "close_run"}:
            raise ValueError(
                "Custom optimize_in_run acquisition plans must not issue "
                f"{msg.command!r}; they run inside Blop's enclosing optimization run."
            )
        return msg

    return (yield from bpp.msg_mutator(plan, _reject))


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
    _validate_suggestions(suggestions)
    try:
        uid = yield from acquisition_plan(suggestions, actuators, optimization_problem.sensors, *args, **kwargs)
    except Exception:
        if isinstance(optimizer, TrialFaultAware):
            optimizer.register_failures(suggestions)
        raise

    evaluation_function: EvaluationFunction = optimization_problem.evaluation_function
    outcomes = evaluation_function(uid, suggestions)
    _validate_outcomes(outcomes, suggestions)
    optimizer.ingest(outcomes)

    return uid, suggestions, outcomes


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
def default_in_run_acquire(
    suggestions: Sequence[Mapping],
    actuators: Sequence[Actuator],
    sensors: Sequence[Sensor] | None = None,
    *,
    per_step: bp.PerStep | None = None,
    **kwargs: Any,
) -> MsgGenerator[tuple[Hashable, ...]]:
    """Acquire suggestions inside an already-open Bluesky run.

    This plan moves through the suggestions, optionally reordering them for efficient motion,
    and executes a Bluesky list scan without opening a child run. The list scan's stage and
    unstage messages are preserved.

    Parameters
    ----------
    suggestions : Sequence[Mapping]
        Suggested parameterizations to execute. Each suggestion must contain a hashable ``_id``.
    actuators : Sequence[Actuator]
        Actuators to move to the suggested positions.
    sensors : Sequence[Sensor] | None, optional
        Sensors that produce data to evaluate. Non-readable sensors are ignored.
    per_step : bp.PerStep | None, optional
        Bluesky list-scan step hook. Custom hooks may emit any number of events into any streams.
    **kwargs : Any
        Additional keyword arguments to pass to :func:`bluesky.plans.list_scan`.

    Returns
    -------
    tuple[Hashable, ...]
        Suggestion IDs in the order the suggestions were executed.

        This identifier intentionally does not encode stream names, event UIDs, event counts, or
        per-stream offsets. Custom ``per_step`` hooks may emit any number of events into any number
        of streams. The matching evaluation function is responsible for interpreting those documents
        and correlating them with these ordered suggestion IDs.

    Yields
    ------
    Msg
        Bluesky messages.
    """
    _validate_suggestions(suggestions)
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

    suggestion_ids = _suggestion_ids(suggestions)
    plan_args = _unpack_for_list_scan(suggestions, actuators)
    # TODO: fix argument type in bluesky.plans.list_scan
    yield from _drop_run_control_messages(
        bp.list_scan(
            readables,
            *plan_args,  # type: ignore[arg-type]
            per_step=per_step,
            **kwargs,
        )
    )

    return suggestion_ids


@plan
def optimize_in_run(
    optimization_problem: OptimizationProblem,
    iterations: int = 1,
    n_points: int = 1,
    checkpoint_interval: int | None = None,
    readable_cache: dict[str, InferredReadable] | None = None,
    **kwargs: Any,
) -> MsgGenerator[None]:
    """Solve an optimization problem by evaluating acquisition documents inside one run.

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
            "optimization_stream": OPTIMIZE_IN_RUN_TRACKING_STREAM,
        }
    )
    readable_cache = readable_cache or {}

    optimizer = optimization_problem.optimizer
    actuators = optimization_problem.actuators
    acquisition_plan = optimization_problem.acquisition_plan or default_in_run_acquire

    @bpp.set_run_key_decorator(OPTIMIZE_IN_RUN_KEY)
    @bpp.run_decorator(md=_md)
    def _optimize_in_run() -> MsgGenerator[None]:
        for i in range(iterations):
            suggestions = optimizer.suggest(n_points)
            _validate_suggestions(suggestions)
            try:
                uid = yield from _reject_child_run_messages(
                    acquisition_plan(suggestions, actuators, optimization_problem.sensors, **kwargs)
                )
                outcomes = optimization_problem.evaluation_function(uid, suggestions)
                _validate_outcomes(outcomes, suggestions)
            except Exception:
                if isinstance(optimizer, TrialFaultAware):
                    # TODO: Is it possible to be more fine-grained than this?
                    # Some suggestions may have been acquired/evaluated without issue.
                    optimizer.register_failures(suggestions)
                raise

            optimizer.ingest(outcomes)
            yield from read_step(
                uid,
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

    return (yield from _optimize_in_run())


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
