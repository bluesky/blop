import functools
import logging
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, cast

import bluesky.plan_stubs as bps
import bluesky.plans as bp
import bluesky.preprocessors as bpp
import numpy as np
from bluesky.protocols import Readable, Reading
from bluesky.utils import MsgGenerator, plan

from ..protocols import ID_KEY, Actuator, CanRegisterSuggestions, Checkpointable, OptimizationProblem, Optimizer, Sensor
from .utils import InferredReadable, collect_optimization_metadata, route_suggestions

logger = logging.getLogger(__name__)

_BLUESKY_UID_KEY: Literal["bluesky_uid"] = "bluesky_uid"
_SUGGESTION_IDS_KEY: Literal["suggestion_ids"] = "suggestion_ids"
_DEFAULT_ACQUIRE_RUN_KEY: Literal["default_acquire"] = "default_acquire"
_SAMPLE_SUGGESTIONS_RUN_KEY: Literal["sample_suggestions"] = "sample_suggestions"
_OPTIMIZE_RUN_KEY: Literal["optimize"] = "optimize"


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
    A default plan to acquire data for optimization. Simply a list scan.

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
            current_position = yield from read(cast(Sequence[Readable], actuators))
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


@plan
def optimize_step(
    optimization_problem: OptimizationProblem,
    n_points: int = 1,
    *args: Any,
    **kwargs: Any,
) -> MsgGenerator[tuple[str, list[dict], list[dict]]]:
    """
    A single step of the optimization loop.

    Parameters
    ----------
    optimization_problem : OptimizationProblem
        The optimization problem to solve.
    n_points : int, optional
        The number of points to suggest.

    Returns
    -------
    tuple[list[dict], list[dict]]
        A tuple containing the suggestions and outcomes of the step.
    """
    if optimization_problem.acquisition_plan is None:
        acquisition_plan = default_acquire
    else:
        acquisition_plan = optimization_problem.acquisition_plan
    optimizer = optimization_problem.optimizer
    actuators = optimization_problem.actuators
    suggestions = optimizer.suggest(n_points)
    if any(ID_KEY not in suggestion for suggestion in suggestions):
        raise ValueError(
            f"All suggestions must contain an '{ID_KEY}' key to later match with the outcomes. Please review your "
            f"optimizer implementation. Got suggestions: {suggestions}"
        )

    uid = yield from acquisition_plan(suggestions, actuators, optimization_problem.sensors, *args, **kwargs)
    outcomes = optimization_problem.evaluation_function(uid, suggestions)
    if any(ID_KEY not in outcome for outcome in outcomes):
        raise ValueError(
            f"All outcomes must contain an '{ID_KEY}' key that matches with the suggestions. Please review your "
            f"evaluation function. Got suggestions: {suggestions} and outcomes: {outcomes}"
        )
    optimizer.ingest(outcomes)

    return uid, suggestions, outcomes


def _maybe_checkpoint(optimizer: Optimizer, checkpoint_interval: int | None, iteration: int) -> None:
    """Helper function to maybe create a checkpoint of the optimizer state at a given interval and iteration."""
    if checkpoint_interval and (iteration + 1) % checkpoint_interval == 0:
        if not isinstance(optimizer, Checkpointable):
            raise ValueError(
                "The optimizer is not checkpointable. Please review your optimizer configuration or implementation."
            )
        optimizer.checkpoint()


@plan
def _read_step(
    uid: str, suggestions: list[dict], outcomes: list[dict], n_points: int, readable_cache: dict[str, InferredReadable]
) -> MsgGenerator[None]:
    """Helper plan to read the suggestions and outcomes of a single optimization step.

    If fewer suggestions are returned than n_points arrays are padded to n_points length
    with np.nan to ensure consistent shapes for event-model specification.

    Parameters
    ----------
    uid : str
        The Bluesky run UID from the acquisition plan.
    suggestions : list[dict]
        List of suggestion dictionaries, each containing an ID_KEY.
    outcomes : list[dict]
        List of outcome dictionaries, each containing an ID_KEY matching suggestions.
    n_points : int
        Expected number of suggestions. Arrays will be padded to this length if needed.
    readable_cache : dict[str, InferredReadable]
        Cache of InferredReadable objects to reuse across iterations.
    """
    # Group by ID_KEY to get proper suggestion/outcome order
    suggestion_by_id = {}
    outcome_by_id = {}
    for suggestion in suggestions:
        suggestion_copy = suggestion.copy()
        key = str(suggestion_copy.pop(ID_KEY))
        suggestion_by_id[key] = suggestion_copy
    for outcome in outcomes:
        outcome_copy = outcome.copy()
        key = str(outcome_copy.pop(ID_KEY))
        outcome_by_id[key] = outcome_copy
    sids = {str(sid) for sid in suggestion_by_id.keys()}
    if sids != set(outcome_by_id.keys()):
        raise ValueError(
            "The suggestions and outcomes must contain the same IDs. Got suggestions: "
            f"{set(suggestion_by_id.keys())} and outcomes: {set(outcome_by_id.keys())}"
        )

    # Flatten the suggestions and outcomes into a single dictionary of lists
    suggestions_flat: dict[str, list[Any]] = defaultdict(list)
    outcomes_flat: dict[str, list[Any]] = defaultdict(list)
    # Sort for deterministic ordering, not strictly necessary
    sorted_sids = sorted(sids)
    for key in sorted_sids:
        for name, value in suggestion_by_id[key].items():
            suggestions_flat[name].append(value)
        for name, value in outcome_by_id[key].items():
            outcomes_flat[name].append(value)

    # Pad arrays to n_points if suggestions had fewer trials than expected
    # TODO: Use awkward-array to handle this in the future
    actual_n = len(sorted_sids)
    if actual_n < n_points:
        # Pad suggestion arrays with NaN
        for name in suggestions_flat:
            suggestions_flat[name].extend([np.nan] * (n_points - actual_n))
        # Pad outcome arrays with NaN
        for name in outcomes_flat:
            outcomes_flat[name].extend([np.nan] * (n_points - actual_n))
        # Pad suggestion IDs with empty string to maintain string dtype
        sorted_sids.extend([""] * (n_points - actual_n))

    # Create or update the InferredReadables for the suggestion_ids, step uid, suggestions, and outcomes
    if _SUGGESTION_IDS_KEY not in readable_cache:
        readable_cache[_SUGGESTION_IDS_KEY] = InferredReadable(_SUGGESTION_IDS_KEY, initial_value=sorted_sids)
    else:
        readable_cache[_SUGGESTION_IDS_KEY].update(sorted_sids)
    if _BLUESKY_UID_KEY not in readable_cache:
        readable_cache[_BLUESKY_UID_KEY] = InferredReadable(_BLUESKY_UID_KEY, initial_value=uid)
    else:
        readable_cache[_BLUESKY_UID_KEY].update(uid)
    for name, value in suggestions_flat.items():
        if name not in readable_cache:
            readable_cache[name] = InferredReadable(name, initial_value=value)
        else:
            readable_cache[name].update(value)
    for name, value in outcomes_flat.items():
        if name not in readable_cache:
            readable_cache[name] = InferredReadable(name, initial_value=value)
        else:
            readable_cache[name].update(value)

    # Read and save to produce a single event
    yield from bps.trigger_and_read(list(readable_cache.values()))


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
    A plan to solve the optimization problem.

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
            "run_key": _OPTIMIZE_RUN_KEY,
        }
    )

    # Encapsulate the optimization plan in a run decorator
    @bpp.set_run_key_decorator(_OPTIMIZE_RUN_KEY)
    @bpp.run_decorator(md=_md)
    def _optimize() -> MsgGenerator[None]:
        for i in range(iterations):
            # Perform a single step of the optimization
            uid, suggestions, outcomes = yield from optimize_step(optimization_problem, n_points, **kwargs)

            # Read the optimization step into the Bluesky and emit events for each suggestion and outcome
            yield from _read_step(uid, suggestions, outcomes, n_points, readable_cache)

            # Possibly take a checkpoint of the optimizer state
            _maybe_checkpoint(optimization_problem.optimizer, checkpoint_interval, i)

    # Start the optimization run
    return (yield from _optimize())


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
            "run_key": _SAMPLE_SUGGESTIONS_RUN_KEY,
        }
    )

    @bpp.set_run_key_decorator(_SAMPLE_SUGGESTIONS_RUN_KEY)
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
        yield from _read_step(uid, suggestions, outcomes, len(suggestions), readable_cache or {})

        return uid, suggestions, outcomes

    return (yield from _inner_sample_suggestions())


@plan
def read(readables: Sequence[Readable], **kwargs: Any) -> MsgGenerator[dict[str, Any]]:
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
            parameterization = yield from read(cast(Sequence[Readable], actuators))
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


@plan
def take_reading_with_background(
    readables: Sequence[Readable],
    name: str = "primary",
    block_beam: Callable[[], MsgGenerator[None]] | None = None,
    unblock_beam: Callable[[], MsgGenerator[None]] | None = None,
) -> MsgGenerator[Mapping[str, Reading]]:
    """
    Takes a reading of the readables while the beam is blocked and then again while the beam is unblocked.

    Parameters
    ----------
    readables: Sequence[Readable]
        The readables to read.
    name: str = "primary"
        The name of the reading.
    block_beam: Callable[[], MsgGenerator[None]] | None = None
        A callable that blocks the beam (e.g. by closing a shutter).
    unblock_beam: Callable[[], MsgGenerator[None]] | None = None
        A callable that unblocks the beam (e.g. by opening a shutter).

    Returns
    -------
    Mapping[str, Reading]
        The readings from the final trigger_and_read operation.
    """
    if block_beam is None or unblock_beam is None:
        raise ValueError("block_beam and unblock_beam plans must be provided.")
    yield from block_beam()
    yield from bps.trigger_and_read(readables, name=f"{name}_background")
    yield from unblock_beam()
    reading = yield from bps.trigger_and_read(readables, name=name)
    return reading


def per_step_background_read(
    block_beam: Callable[[], MsgGenerator[None]], unblock_beam: Callable[[], MsgGenerator[None]]
) -> bp.PerStep:
    """
    Returns a per-step plan function that takes a reading of the readables while the beam is blocked and then
    again while the beam is unblocked.

    Useful for downstream analysis that requires per-step background readings (e.g. background subtraction).

    Parameters
    ----------
    block_beam: Callable[[], MsgGenerator[None]]
        A callable that blocks the beam (e.g. by closing a shutter).
    unblock_beam: Callable[[], MsgGenerator[None]]
        A callable that unblocks the beam (e.g. by opening a shutter).

    See Also
    --------
    bluesky.plans.one_nd_step : The Bluesky plan to execute for each step of the scan.
    """
    take_reading = functools.partial(take_reading_with_background, block_beam=block_beam, unblock_beam=unblock_beam)
    return functools.partial(bps.one_nd_step, take_reading=take_reading)


@plan
def acquire_with_background(
    suggestions: list[dict],
    actuators: Sequence[Actuator],
    sensors: Sequence[Sensor] | None = None,
    *,
    block_beam: Callable[[], MsgGenerator[None]],
    unblock_beam: Callable[[], MsgGenerator[None]],
    **kwargs: Any,
) -> MsgGenerator[str]:
    """
    A plan to acquire data for optimization with background readings.

    Parameters
    ----------
    suggestions: list[dict]
        A list of dictionaries, each containing the parameterization of a point to evaluate.
        The "_id" key is optional and can be used to identify each suggestion. It is suggested
        to add "_id" values to the run metadata for later identification of the acquired data.
    actuators: Sequence[Actuator]
        The actuators to move to their suggested positions.
    sensors: Sequence[Sensor] | None = None
        The sensors that produce data to evaluate.
    block_beam: Callable[[], MsgGenerator[None]]
        A Bluesky plan that blocks the beam (e.g. by closing a shutter).
    unblock_beam: Callable[[], MsgGenerator[None]]
        A Bluesky plan that unblocks the beam (e.g. by opening a shutter).
    **kwargs: Any
        Additional keyword arguments to pass to the acquisition plan.

    Returns
    -------
    str
        The UID of the Bluesky run.

    See Also
    --------
    acquire : The base plan to acquire data.
    per_step_background_read : The per-step plan to take background readings.
    """
    if sensors is None:
        sensors = []
    per_step = per_step_background_read(block_beam, unblock_beam)
    return (yield from default_acquire(suggestions, actuators, sensors, per_step=per_step, **kwargs))
