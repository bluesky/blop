from collections import defaultdict
from typing import Any, Literal

import bluesky.plan_stubs as bps
import numpy as np
from bluesky.utils import MsgGenerator, plan

from .protocols import ID_KEY
from .utils import InferredReadable

_BLUESKY_UID_KEY: Literal["bluesky_uid"] = "bluesky_uid"
_SUGGESTION_IDS_KEY: Literal["suggestion_ids"] = "suggestion_ids"


@plan
def read_step(
    uid: str, suggestions: list[dict], outcomes: list[dict], n_points: int, readable_cache: dict[str, InferredReadable]
) -> MsgGenerator[None]:
    """Plan stub to read the suggestions and outcomes of a single optimization step.

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
