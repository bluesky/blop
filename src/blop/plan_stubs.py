"""Bluesky plan stubs for optimization."""

import logging
from collections import defaultdict
from collections.abc import Hashable, Mapping, MutableMapping, Sequence
from typing import Any, Literal, cast

import bluesky.plan_stubs as bps
import bluesky.plans as bp
import numpy as np
from bluesky.protocols import Readable
from bluesky.utils import MsgGenerator, plan
from numpy.typing import ArrayLike

from .protocols import ID_KEY, Actuator, Optimizer, Sensor
from .utils import (
    InferredReadable,
    Source,
    _drop_run_control_messages,
    _suggestion_ids,
    _unpack_for_list_scan,
    _validate_suggestions,
    route_suggestions,
)

logger = logging.getLogger(__name__)

_ACQUISITION_UID_KEY: Literal["acquisition_uid"] = "acquisition_uid"
_SUGGESTION_IDS_KEY: Literal["suggestion_ids"] = "suggestion_ids"


def _is_array_like_identifier(uid: Hashable) -> bool:
    try:
        numpy_array = np.array(uid)
    except (TypeError, ValueError):
        return False
    return numpy_array.dtype != object


def _acquisition_identifier_value(uid: Hashable) -> ArrayLike:
    """Convert a hashable acquisition identifier to an event-readable value."""
    if _is_array_like_identifier(uid):
        return cast(ArrayLike, uid)
    return repr(uid)


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


@plan
def read_step(
    uid: Hashable,
    suggestions: Sequence[Mapping],
    outcomes: Sequence[Mapping],
    n_points: int,
    readable_cache: MutableMapping[str, InferredReadable],
    stream_name: str = "primary",
) -> MsgGenerator[None]:
    """Plan stub to read the suggestions and outcomes of a single optimization step.

    If fewer suggestions are returned than n_points arrays are padded to n_points length
    with np.nan to ensure consistent shapes for event-model specification.

    The emitted ``acquisition_uid`` field retains native array-like identifiers.
    Other hashable identifiers are represented by ``repr(uid)``.

    Parameters
    ----------
    uid : Hashable
        The acquisition identifier returned by the acquisition plan.
    suggestions : Sequence[Mapping]
        Sequence of suggestion mappings, each containing an ID_KEY.
    outcomes : Sequence[Mapping]
        Sequence of outcome mappings, each containing an ID_KEY matching suggestions.
    n_points : int
        Expected number of suggestions. Arrays will be padded to this length if needed.
    readable_cache : dict[str, InferredReadable]
        Cache of InferredReadable objects to reuse across iterations.
    stream_name : str, optional
        Event stream name for the optimization tracking event.
    """
    # Group by ID_KEY to get proper suggestion/outcome order
    suggestion_by_id = {}
    outcome_by_id = {}
    for suggestion in suggestions:
        suggestion_copy = dict(suggestion)
        key = str(suggestion_copy.pop(ID_KEY))
        suggestion_by_id[key] = suggestion_copy
    for outcome in outcomes:
        outcome_copy = dict(outcome)
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
        readable_cache[_SUGGESTION_IDS_KEY] = InferredReadable(
            _SUGGESTION_IDS_KEY, source=Source.SUGGESTION_ID, initial_value=sorted_sids
        )
    else:
        readable_cache[_SUGGESTION_IDS_KEY].update(sorted_sids)
    # Need to normalize the value here since `Hashable` is very broad
    normalized_uid = _acquisition_identifier_value(uid)
    if _ACQUISITION_UID_KEY not in readable_cache:
        readable_cache[_ACQUISITION_UID_KEY] = InferredReadable(
            _ACQUISITION_UID_KEY, source=Source.ACQUISITION_UID, initial_value=normalized_uid
        )
    else:
        readable_cache[_ACQUISITION_UID_KEY].update(normalized_uid)
    for name, value in suggestions_flat.items():
        if name not in readable_cache:
            readable_cache[name] = InferredReadable(name, source=Source.PARAMETER, initial_value=value)
        else:
            readable_cache[name].update(value)
    for name, value in outcomes_flat.items():
        if name not in readable_cache:
            readable_cache[name] = InferredReadable(name, source=Source.OUTCOME, initial_value=value)
        else:
            readable_cache[name].update(value)

    # Read and save to produce a single event
    yield from bps.trigger_and_read(list(readable_cache.values()), name=stream_name)


@plan
def navigate_to_best(
    actuators: Sequence[Actuator],
    optimizer: Optimizer | None = None,
    parameterization: Mapping | None = None,
) -> MsgGenerator[None]:
    """
    Move actuators to the best point found during optimization.

    If no explicit parameterization is provided, queries the optimizer for its
    best point(s). For multi-objective optimizers that return multiple Pareto-optimal
    points, an explicit parameterization must be provided.

    Parameters
    ----------
    actuators : Sequence[Actuator]
        The actuators to move to the best parameterization.
    optimizer : Optimizer | None, optional
        The optimizer to query for the best point.
    parameterization : Mapping | None, optional
        Explicit parameterization to navigate to. If None, queries the optimizer's
        best point. For multi-objective problems, call ``optimizer.get_best_points()``
        to inspect the Pareto set and select one.

    Raises
    ------
    TypeError
        If both ``parameterization`` and ``optimizer`` arguments are ``None``.
    ValueError
        If the optimizer returns multiple Pareto-optimal points and no
        explicit ``parameterization`` is provided.
    """
    if parameterization is None:
        if optimizer is None:
            raise TypeError("Either pass an explicit parameterization or use an optimizer.")
        best_points = optimizer.get_best_points()
        if len(best_points) > 1:
            raise ValueError(
                f"The optimizer returned {len(best_points)} Pareto-optimal points. "
                "Please call optimizer.get_best_points() to inspect them and pass your "
                "chosen parameterization explicitly via the 'parameterization' argument."
            )
        _, parameterization, _ = best_points[0]

    actuator_by_name = {actuator.name: actuator for actuator in actuators}
    moves = []
    for name, value in parameterization.items():
        if name in actuator_by_name:
            moves.append(actuator_by_name[name])
            moves.append(value)

    if moves:
        yield from bps.mv(*moves)


@plan
def list_scan_in_run(
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

    .. warning::

        The single-run optimization API is **experimental**. This plan may change in
        future releases without a deprecation period.

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
