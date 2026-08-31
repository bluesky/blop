"""A set of useful helper utilities."""

import time
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any

import networkx as nx
import numpy as np
from bluesky.protocols import HasHints, HasParent, Hints, Readable, Reading
from event_model import DataKey
from numpy.typing import ArrayLike

from .protocols import ID_KEY, Checkpointable, OptimizationProblem, Optimizer


class Source(StrEnum):
    """An enum that helps describe where the data key comes from."""

    OUTCOME = "optimization-outcome"
    PARAMETER = "optimization-parameter"
    SUGGESTION_ID = "optimization-suggestion-id"
    ACQUISITION_UID = "optimization-acquisition-uid"
    OTHER = "optimization-other"


def _maybe_checkpoint(optimizer: Optimizer, checkpoint_interval: int | None, iteration: int) -> None:
    """Maybe create a checkpoint of the optimizer state at a given interval and iteration."""
    if checkpoint_interval and (iteration + 1) % checkpoint_interval == 0:
        if not isinstance(optimizer, Checkpointable):
            raise ValueError(
                "The optimizer is not checkpointable. Please review your optimizer configuration or implementation."
            )
        optimizer.checkpoint()


def _infer_data_key(source: Source, value: ArrayLike) -> DataKey:
    """Infer the data key from the provided value."""
    numpy_array = np.array(value)
    # Descriptions are cached across updates, so reserve enough space for UUIDs.
    dtype_numpy = (
        np.promote_types(numpy_array.dtype, np.dtype("<U36")).str if numpy_array.dtype.kind == "U" else numpy_array.dtype.str
    )
    if len(numpy_array.shape) > 1 or (len(numpy_array.shape) == 1 and numpy_array.shape[0] > 1):
        dtype = "array"
        shape = list(numpy_array.shape)
    else:
        shape = []
        item = numpy_array[0] if len(numpy_array.shape) == 1 else numpy_array.item()
        if isinstance(item, (int, float)):
            dtype = "number"
        else:
            dtype = "string"
    return DataKey(source=source.value, dtype=dtype, shape=shape, dtype_numpy=dtype_numpy)


class InferredReadable(Readable, HasHints, HasParent):
    """
    An inferred readable object that can be used in Bluesky plans.

    It performs inference on the initial value to describe the data key.

    Parameters
    ----------
    name : str
        The name of the readable instance.
    source : str

    initial_value : numpy.typing.ArrayLike
        The initial value of the readable instance.
    """

    def __init__(self, name: str, source: Source, initial_value: ArrayLike) -> None:
        self._name = name
        self._source = source
        self._data_key = None

        if isinstance(initial_value, np.ndarray):
            self._dtype = initial_value.dtype
            initial_value = initial_value.tolist()
        else:
            self._dtype = None

        if isinstance(initial_value, Sequence) and len(initial_value) == 1:
            initial_value = initial_value[0]
        self._value = initial_value

    @property
    def parent(self) -> Any | None:
        """Parent of the readable, always ``None``."""
        return None

    @property
    def name(self) -> str:
        """Name of the readable."""
        return self._name

    @property
    def hints(self) -> Hints:
        """Hints for callbacks (such as plotting)."""
        return {
            "fields": [self.name],
            "dimensions": [],
            "gridding": "rectilinear",
        }

    def describe(self) -> dict[str, DataKey]:
        """Describe the properties of this readable."""
        if not self._data_key:
            # Use stored dtype if available, otherwise infer
            if self._dtype is not None:
                numpy_array = np.array(self._value, dtype=self._dtype)
            else:
                numpy_array = np.array(self._value)
            self._data_key = _infer_data_key(self._source, numpy_array)
        return {self.name: self._data_key}

    def update(self, value: ArrayLike) -> None:
        """Update the stored value of this readable."""
        if isinstance(value, np.ndarray):
            self._dtype = value.dtype
            value = value.tolist()
        else:
            self._dtype = None

        if isinstance(value, Sequence) and len(value) == 1:
            value = value[0]
        self._value = value

    def read(self) -> dict[str, Reading]:
        """Emit a reading for this readable."""
        return {
            self.name: {
                "value": self._value,
                "timestamp": time.time(),
            }
        }


def _validate_route_index(index: Sequence[int], num_points: int) -> list[int]:
    """Validate that a route visits every suggestion once."""
    route = list(index)
    counts: dict[int, int] = {}
    for point_index in route:
        counts[point_index] = counts.get(point_index, 0) + 1

    expected = set(range(num_points))
    seen = set(counts)
    if seen != expected or len(route) != num_points:
        missing = sorted(expected - seen)
        duplicates = sorted(point_index for point_index, count in counts.items() if count > 1)
        raise RuntimeError(
            "TSP solver returned an invalid route. "
            f"Missing point indices {missing!r}; duplicate point indices {duplicates!r}; route {route!r}."
        )

    return route


def _get_route_index(points: np.ndarray, starting_point: np.ndarray | None = None) -> list[int]:
    num_suggestions = len(points)
    if num_suggestions < 2:
        return list(range(num_suggestions))

    graph = nx.DiGraph()
    for i, i_point in enumerate(points):
        for j, j_point in enumerate(points):
            if i == j:
                continue
            d = np.sqrt(np.sum(np.square(i_point - j_point)))
            graph.add_edge(i, j, weight=d)

    anchor_node = num_suggestions
    for i, point in enumerate(points):
        d = 0.0 if starting_point is None else np.sqrt(np.sum(np.square(starting_point - point)))
        graph.add_edge(anchor_node, i, weight=d)
        graph.add_edge(i, anchor_node, weight=0.0)

    cycle = nx.approximation.simulated_annealing_tsp(graph, init_cycle="greedy", source=anchor_node, seed=0)
    index = list(cycle[1:-1])

    return _validate_route_index(index, num_suggestions)


def route_suggestions(suggestions: Sequence[Mapping], starting_position: dict | None = None):
    """Route suggestions through a shortest open path over routed dimensions."""
    if len(suggestions) == 1:
        return suggestions

    dims_to_route = [dim for dim, value in suggestions[0].items() if (dim != ID_KEY) and isinstance(value, float)]
    points = np.array([[s[dim] for dim in dims_to_route] for s in suggestions])
    starting_point = np.array([starting_position[dim] for dim in dims_to_route]) if starting_position else None

    return [suggestions[i] for i in _get_route_index(points=points, starting_point=starting_point)]


def collect_optimization_metadata(optimization_problem: OptimizationProblem) -> dict[str, Any]:
    """Collect the metadata for the optimization problem."""
    if hasattr(optimization_problem.evaluation_function, "__name__"):
        evaluation_function_name = optimization_problem.evaluation_function.__name__  # type: ignore[attr-defined]
    else:
        evaluation_function_name = optimization_problem.evaluation_function.__class__.__name__
    if hasattr(optimization_problem.acquisition_plan, "__name__"):
        acquisition_plan_name = optimization_problem.acquisition_plan.__name__  # type: ignore[attr-defined]
    else:
        acquisition_plan_name = optimization_problem.acquisition_plan.__class__.__name__
    return {
        "evaluation_function": evaluation_function_name,
        "acquisition_plan": acquisition_plan_name,
        "optimizer": optimization_problem.optimizer.__class__.__name__,
        "sensors": [sensor.name for sensor in optimization_problem.sensors],
        "actuators": [actuator.name for actuator in optimization_problem.actuators],
    }
