"""Bluesky plans for optimization."""

import logging
from collections.abc import Hashable, Mapping, Sequence
from itertools import count
from typing import Any, Literal, cast

import bluesky.plans as bp
import bluesky.preprocessors as bpp
from bluesky.protocols import Readable
from bluesky.utils import MsgGenerator, plan

from .plan_stubs import list_scan_in_run, read_step, seq_read
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
from .utils import (
    InferredReadable,
    _maybe_checkpoint,
    _reject_child_run_messages,
    _unpack_for_list_scan,
    _validate_outcomes,
    _validate_suggestions,
    collect_optimization_metadata,
    route_suggestions,
)

logger = logging.getLogger(__name__)

_DEFAULT_ACQUIRE_RUN_KEY: Literal["default_acquire"] = "default_acquire"
SAMPLE_SUGGESTIONS_RUN_KEY: Literal["sample_suggestions"] = "sample_suggestions"
OPTIMIZE_RUN_KEY: Literal["optimize"] = "optimize"
OPTIMIZE_IN_RUN_KEY: Literal["optimize_in_run"] = "optimize_in_run"
OPTIMIZE_IN_RUN_TRACKING_STREAM: Literal["optimization"] = "optimization"


@plan
def default_acquire(
    suggestions: Sequence[Mapping],
    actuators: Sequence[Actuator],
    sensors: Sequence[Sensor] | None = None,
    md: Mapping[str, Any] | None = None,
    *,
    per_step: bp.PerStep | None = None,
    **kwargs: Any,
) -> MsgGenerator[str]:
    """
    Acquire data for optimization. Simply a list scan.

    Includes ``"blop_suggestions"`` metadata containing the routed suggestions for
    backwards compatibility and ``"blop_acquisition_order"`` containing IDs in actual scan order.
    Use those IDs, rather than positions in ``suggestions``, to associate acquired rows.

    Parameters
    ----------
    suggestions: Sequence[Mapping]
        A sequence of mappings, each containing the parameterization of a point to evaluate.
        Each mapping must contain a unique ``"_id"`` key used to associate the acquired
        data with its suggestion.
    actuators: Sequence[Actuator]
        The actuators to move and the inputs to move them to.
    sensors: Sequence[Sensor]
        The sensors that produce data to evaluate.
    md : Mapping[str, Any] | None, optional
        Metadata to attach to the start document.
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

    run_md = dict(md or {})
    run_md.update(
        {
            "blop_suggestions": suggestions,
            "blop_acquisition_order": [suggestion[ID_KEY] for suggestion in suggestions],
            "run_key": _DEFAULT_ACQUIRE_RUN_KEY,
        }
    )
    plan_args = _unpack_for_list_scan(suggestions, actuators)
    return (
        # TODO: fix argument type in bluesky.plans.list_scan
        yield from bpp.set_run_key_wrapper(
            bp.list_scan(
                readables,
                *plan_args,  # type: ignore[arg-type]
                per_step=per_step,
                md=run_md,
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
        evaluation_function: EvaluationFunction = optimization_problem.evaluation_function
        outcomes = evaluation_function(uid, suggestions)
    except Exception:
        if isinstance(optimizer, TrialFaultAware):
            optimizer.register_failures(suggestions)
        raise

    _validate_outcomes(outcomes, suggestions)
    optimizer.ingest(outcomes)

    return uid, suggestions, outcomes


@plan
def optimize(
    optimization_problem: OptimizationProblem,
    iterations: int | None = 1,
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
    iterations : int | None, optional
        The maximum number of optimization iterations to run. If None, run until
        the optimizer's stopping criterion is met. An optimizer implementing
        :class:`blop.protocols.SupportsStoppingCriteria` is required when None.
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

    Raises
    ------
    ValueError
        If ``iterations`` is None and the optimizer does not implement
        :class:`blop.protocols.SupportsStoppingCriteria`.

    See Also
    --------
    blop.protocols.OptimizationProblem : The problem to solve.
    blop.protocols.Checkpointable : The protocol for checkpointable objects.
    optimize_step : The plan to execute a single step of the optimization.
    """
    optimizer = optimization_problem.optimizer
    if iterations is None and not isinstance(optimizer, SupportsStoppingCriteria):
        raise ValueError("iterations=None requires an optimizer that implements SupportsStoppingCriteria")

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
        iteration_indices = count() if iterations is None else range(iterations)
        for i in iteration_indices:
            # Perform a single step of the optimization
            uid, suggestions, outcomes = yield from optimize_step(optimization_problem, n_points, **kwargs)

            if isinstance(optimizer, SupportsStoppingCriteria):
                stop_now, stop_reason = optimizer.should_stop()
                if stop_now:
                    reason = stop_reason if stop_reason is not None else "No reason provided"
                    logger.info(f"Global stopping triggered at iteration {i + 1}: {reason}")
                    return

            # Read the optimization step into the Bluesky and emit events for each suggestion and outcome
            yield from read_step(uid, suggestions, outcomes, n_points, readable_cache)

            # Possibly take a checkpoint of the optimizer state
            _maybe_checkpoint(optimizer, checkpoint_interval, i)

    # Start the optimization run
    return (yield from _optimize())


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

    .. warning::

        This plan is **experimental**. Its API is not yet stable and may change in
        future releases without a deprecation period.

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
    acquisition_plan = optimization_problem.acquisition_plan or list_scan_in_run

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
