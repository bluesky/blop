"""Bluesky plan stubs for optimization."""

from collections import defaultdict
from collections.abc import Hashable, Mapping, MutableMapping, Sequence
from typing import Any, Literal, cast

import bluesky.plan_stubs as bps
import numpy as np
from bluesky.utils import MsgGenerator, plan
from numpy.typing import ArrayLike

from .protocols import ID_KEY, Actuator, Optimizer
from .utils import InferredReadable, Source

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
def read_step(
    uid: Hashable,
    suggestions: Sequence[Mapping],
    outcomes: Sequence[Mapping],
    n_points: int,
    readable_cache: MutableMapping[str, InferredReadable],
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
    yield from bps.trigger_and_read(list(readable_cache.values()))


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
