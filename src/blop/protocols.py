from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Generic, Literal, Protocol, TypeVar, runtime_checkable

from bluesky.protocols import EventCollectable, EventPageCollectable, Flyable, HasName, Movable, Readable
from bluesky.utils import MsgGenerator, plan


@runtime_checkable
class MovableHasName(Movable, HasName, Protocol):
    """
    A movable that has a name.

    We use this instead of `bluesky.protocols.NamedMovable` since
    we do not want to require `HasHints` on the movable.

    A `Movable` and `HasName` is sufficient. `HasHints` should be optional.
    """

    ...


ID_KEY: Literal["_id"] = "_id"
Actuator = MovableHasName | Flyable
Sensor = Readable | EventCollectable | EventPageCollectable

TActuator = TypeVar("TActuator")
"""Actuator generic type"""
TSensor = TypeVar("TSensor")
"""Sensor generic type"""
TPlan = TypeVar("TPlan")
"""Plan generic type"""


@runtime_checkable
class CanRegisterSuggestions(Protocol):
    """
    A protocol for optimizers that can register suggestions. This
    allows them to add an "_id" key to the suggestions dynamically and ensure
    that the suggestions are unique.
    """

    def register_suggestions(self, suggestions: list[dict]) -> list[dict]:
        """
        Register the suggestions with the optimizer.

        Parameters
        ----------
        suggestions: list[dict]
            The suggestions to register. The "_id" key is optional and will be overwritten if present.

        Returns
        -------
        list[dict]
            The original suggestions with an "_id" key added.
        """
        ...


@runtime_checkable
class Checkpointable(Protocol):
    """
    A protocol for objects that can can write state to persistent storage.

    Implementers configure storage at construction time (e.g., a file path, databse URI).
    The checkpoint method then saves or updates to that pre-configured location.
    """

    def checkpoint(self) -> None:
        """
        Write the object's state to persistent storage.
        """
        ...


@runtime_checkable
class Optimizer(Protocol):
    """
    A minimal optimizer interface for optimization.

    This protocol defines the interface for optimizers in Blop. Most users will use
    the built-in :class:`blop.ax.optimizer.AxOptimizer`, which provides Bayesian optimization
    via Ax. Custom implementations are only needed for specialized optimization algorithms.

    See Also
    --------
    blop.ax.optimizer.AxOptimizer : Built-in Ax-based optimizer implementation.
    blop.ax.Agent : High-level interface that uses AxOptimizer internally.
    """

    def suggest(self, num_points: int | None = None) -> list[dict]:
        """
        Returns a set of points in the input space, to be evaulated next.

        The "_id" key is optional and can be used to identify suggested trials for later evaluation
        and ingestion.

        Parameters
        ----------
        num_points : int | None, optional
            The number of points to suggest. If not provided, will default to 1.

        Returns
        -------
        list[dict]
            A list of dictionaries, each containing a parameterization of a point to evaluate next.
            Each dictionary must contain a unique "_id" key to identify each parameterization.
        """
        ...

    def ingest(self, points: list[dict]) -> None:
        """
        Ingest a set of points into the experiment. Either from previously suggested points or from an external source.

        The "_id" key is optional and can be used to identify points from previously suggested trials or to identify
        the point as a "baseline" trial.

        Parameters
        ----------
        points : list[dict]
            A list of dictionaries, each containing the outcomes of each suggested parameterization.
        """
        ...


@runtime_checkable
class EvaluationFunction(Protocol):
    """
    A protocol for transforming acquired data into measurable outcomes.

    This protocol defines how to extract and compute optimization objectives from
    acquired data. Custom implementations are needed to define how your beamline
    data translates into the outcomes you want to optimize.

    Notes
    -----
    The evaluation function is called after data acquisition to compute outcomes
    from the acquired data. It should extract relevant data from the Bluesky run
    and compute objective values and metrics for each suggestion.

    Examples
    --------
    See the tutorial documentation for complete examples of evaluation functions:
    :doc:`/tutorials/simple-experiment`

    See Also
    --------
    blop.ax.Agent : Accepts an evaluation function during initialization.
    """

    def __call__(self, uid: str, suggestions: list[dict]) -> list[dict]:
        """
        Evaluate the data from a Bluesky run and produce outcomes.

        Parameters
        ----------
        uid: str
            The unique identifier of the Bluesky run to evaluate.
        suggestions: list[dict]
            A list of dictionaries, each containing the parameterization of a point to evaluate.
            The "_id" key is optional and can be used to identify each suggestion.

        Returns
        -------
        list[dict]
            A list of dictionaries containing the outcomes of the run, one for each suggested parameterization.
            The "_id" key is optional and can be used to identify each outcome.
        """
        ...


@runtime_checkable
class AcquisitionPlan(Protocol):
    """
    A protocol for custom data acquisition plans.

    This protocol defines how to acquire data from the beamline. Most users will use
    the default :func:`blop.plans.default_acquire` plan, which performs a list scan
    over the suggested points. Custom implementations are only needed for specialized
    acquisition strategies (e.g., fly scans, complex detector configurations).

    Notes
    -----
    The acquisition plan is a Bluesky plan that should move the actuators to each
    suggested position and acquire data from the sensors. It must return the UID
    of the Bluesky run so that the evaluation function can retrieve the data.

    See Also
    --------
    blop.plans.default_acquire : Default acquisition plan implementation.
    blop.ax.Agent : Accepts an optional acquisition plan during initialization.
    """

    @plan
    def __call__(
        self,
        suggestions: list[dict],
        actuators: Sequence[Actuator],
        sensors: Sequence[Sensor] | None = None,
        md: dict[str, Any] | None = None,
    ) -> MsgGenerator[str]:
        """
        Acquire data for optimization.

        This should be a Bluesky plan that moves the actuators to each of their suggested positions
        and acquires data from the sensors.

        Parameters
        ----------
        suggestions: list[dict]
            A list of dictionaries, each containing the parameterization of a point to evaluate.
            The "_id" key is optional and can be used to identify each suggestion. It is suggested
            to add "_id" values to the run metadata for later identification of the acquired data.
        actuators: Sequence[Actuator]
            The actuators to move to their suggested positions.
        sensors: Sequence[Sensor], optional
            The sensors that produce data to evaluate.
        md : dict[str, Any] | None, optional
            Metadata to attach to the start document

        Returns
        -------
        str
            The unique identifier of the Bluesky run.
        """
        ...


@dataclass(frozen=True)
class BaseOptimizationProblem(Generic[TActuator, TSensor, TPlan]):
    optimizer: Optimizer
    actuators: Sequence[TActuator]
    sensors: Sequence[TSensor]
    evaluation_function: EvaluationFunction
    acquisition_plan: TPlan | None = None


class OptimizationProblem(BaseOptimizationProblem[Actuator, Sensor, AcquisitionPlan]):
    """
    An optimization problem to solve. Immutable once initialized.

    This dataclass encapsulates all components needed for optimization into a single
    immutable structure. It is typically created via :meth:`blop.ax.Agent.to_optimization_problem`
    and used with optimization plans like :func:`blop.plans.optimize`.

    Attributes
    ----------
    optimizer: Optimizer
        Suggests points to evaluate and ingests outcomes to inform the optimization.
    actuators: Sequence[Actuator]
        Objects that can be moved to control the beamline using the Bluesky RunEngine.
        A subset of the actuators' names must match the names of suggested parameterizations.
    sensors: Sequence[Sensor]
        Objects that can produce data to acquire data from the beamline using the Bluesky RunEngine.
    evaluation_function: EvaluationFunction
        A callable to evaluate data from a Bluesky run and produce outcomes.
    acquisition_plan: AcquisitionPlan, optional
        A Bluesky plan to acquire data from the beamline. If not provided, a default plan will be used.

    See Also
    --------
    blop.ax.Agent.to_optimization_problem : Creates an OptimizationProblem from an Agent.
    blop.plans.optimize : Bluesky plan that uses an OptimizationProblem.
    """

    ...


class QueueserverOptimizationProblem(BaseOptimizationProblem[str, str, str]):
    """
    An optimization problem to solve. Immutable once initialized.

    This dataclass encapsulates all components needed for optimization into a single
    immutable structure. It is typically created via :meth:`blop.ax.QueueserverAgent.to_optimization_problem`
    and used with bluesky-queueserver-api. Actuators, sensors, and the acquisition plan are referenced
    by their names, since their instances live on a remote server.

    Attributes
    ----------
    optimizer: Optimizer
        Suggests points to evaluate and ingests outcomes to inform the optimization.
    actuators: Sequence[str]
        Names of objects that can be moved to control the beamline using the Bluesky RunEngine.
        A subset of the actuators' names must match the names of suggested parameterizations.
    sensors: Sequence[str]
        Names of objects that can produce data to acquire data from the beamline using the Bluesky RunEngine.
    evaluation_function: EvaluationFunction
        A callable to evaluate data from a Bluesky run and produce outcomes.
    acquisition_plan: str, optional
        The name of a Bluesky plan to acquire data from the beamline. If not provided, a default plan name will be used.
        The plan must match the arguments of :ref:`AcquisitionPlan`.

    See Also
    --------
    blop.ax.QueueserverAgent.to_optimization_problem : Creates a QueueserverOptimizationProblem from an agent.
    blop.queueserver.QueueserverOptimizationRunner : Runs the optimization loop using the bluesky-queueserver-api.
    """

    ...
