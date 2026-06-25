from collections.abc import Mapping, Sequence
from typing import Any, cast

import bluesky.preprocessors as bpp
from bluesky.callbacks import CallbackBase

from blop.ax.dof import RangeDOF
from blop.ax.objective import Objective
from blop.callbacks.logger import OptimizationLogger
from blop.callbacks.router import OptimizationCallbackRouter
from blop.plans import optimize
from blop.protocols import (
    AcquisitionPlan,
    Actuator,
    EvaluationFunction,
    OptimizationProblem,
    Sensor,
)
from blop.utils import InferredReadable

from .optimizer import SCP, ScipyCFG, ScipyOptimizer


class Scipy:
    """
    A convenience interface associated with running optimizations with Scipy, providing similar syntax to the Ax Agent
    (allowing drop in swapping as much as possible). Useful as a cover in for all the QOL provided by the Agent object.
    """

    def __init__(
        self,
        sensors: Sequence[Sensor],
        config: ScipyCFG,
        evaluation_function: EvaluationFunction,
        acquisition_plan: AcquisitionPlan | None = None,
        **kwargs: Any,
    ):

        self.config = config
        self._sensors = sensors
        self._actuators = [cast(Actuator, dof.actuator) for dof in config.dofs if dof.actuator is not None]
        self._evaluation_function = evaluation_function
        self._acquisition_plan = acquisition_plan
        self.timeout = kwargs.pop("timeout", 200)
        self._optimizer = ScipyOptimizer(self.config, timeout=self.timeout)
        self._readable_cache: dict[str, InferredReadable] = {}
        self._callbacks: list[CallbackBase] = [OptimizationLogger()]
        self._callback_router = OptimizationCallbackRouter(self._callbacks)
        self.sessioning = kwargs.pop("sessioning", True)

    @classmethod
    def Agent(
        cls,
        sensors: Sequence[Sensor],
        dofs: Sequence[RangeDOF],
        objectives: Sequence[Objective],
        evaluation_function: EvaluationFunction,
        acquisition_plan: AcquisitionPlan | None = None,
        optimizer: SCP = SCP.Default,
        # dof_constraints: Sequence[DOFConstraint] | None = None,  #implemented in future iterations? make to match ax?
        # outcome_constraints: Sequence[OutcomeConstraint] | None = None,
        **kwargs: Any,
    ):
        """
        A nearly emcompassing interface to provide strong interoperability with Ax agent formalism.

        Parameters
        ----------
        sensors : Sequence[Sensor]
            The sensors to use for acquisition. These should be the minimal set
            of sensors that are needed to compute the objectives.
        dofs : Sequence[DOF]
            The degrees of freedom that the agent can control, which determine the search space.
        objectives : Sequence[Objective]
            The objectives which the agent will try to optimize.
        evaluation_function : EvaluationFunction
            The function to evaluate acquired data and produce outcomes.
        acquisition_plan : AcquisitionPlan | None, optional
            The acquisition plan to use for acquiring data from the beamline. If not provided,
            :func:`blop.plans.default_acquire` will be used.
        **kwargs : Any
            Additional keyword arguments to configure the Ax experiment.

        Notes
        -----
        This is a nearly drop in replacement for Ax agent sans dof + outcome constraints and checkpointing

        See Also
        --------
        blop.ax.Agent

        """

        if optimizer not in SCP:
            raise ValueError(f"optimizer {optimizer} not in supported optimizers:{list(SCP)}")
        if len(objectives) > 1:
            raise ValueError("Multiple Objectives are not supported for gradient optimizers")
        config = ScipyCFG(
            dofs=dofs,
            objective=objectives[0],
            optimizer=optimizer,
            max_iter=kwargs.get("max_iter", None),
            eps=kwargs.get("eps", None),
            rescale=kwargs.get("scale", None),
        )
        return cls(sensors, config, evaluation_function, acquisition_plan, **kwargs)

    @property
    def sensors(self) -> Sequence[Sensor]:
        """The sensors used for data acquisition."""
        return self._sensors

    @property
    def actuators(self) -> Sequence[Actuator]:
        """The actuators that control the degrees of freedom."""
        return self._actuators

    @property
    def evaluation_function(self) -> EvaluationFunction:
        """The function used to evaluate acquired data and produce outcomes."""
        return self._evaluation_function

    @property
    def acquisition_plan(self) -> AcquisitionPlan | None:
        """The acquisition plan for acquiring data, or ``None`` if using the default."""
        return self._acquisition_plan

    @property
    def callbacks(self) -> list[CallbackBase]:
        """The list of active optimization callbacks.

        Callbacks in this list receive documents from ``"optimize"`` and
        ``"sample_suggestions"`` runs. The default list contains an
        :class:`~blop.callbacks.logger.OptimizationLogger`.

        The list can be mutated directly, or use :meth:`subscribe` /
        :meth:`unsubscribe` for convenience.
        """
        return self._callbacks

    def subscribe(self, callback: CallbackBase) -> None:
        """Subscribe a callback to receive optimization run documents.

        Parameters
        ----------
        callback : CallbackBase
            A Bluesky callback instance.

        Raises
        ------
        ValueError
            If *callback* is already subscribed.
        """
        if callback in self._callbacks:
            raise ValueError(f"Callback {callback!r} is already subscribed.")
        self._callbacks.append(callback)

    def unsubscribe(self, callback: CallbackBase) -> None:
        """Unsubscribe a previously subscribed callback.

        Parameters
        ----------
        callback : CallbackBase
            The callback instance to remove.

        Raises
        ------
        ValueError
            If *callback* is not subscribed.
        """
        self._callbacks.remove(callback)

    def to_optimization_problem(self) -> OptimizationProblem:
        """
        Construct an optimization problem from the Scipy Base class

        Creates an immutable :class:`blop.protocols.OptimizationProblem` that
        encapsulates all components needed for optimization. This is typically
        used internally by optimization plans.

        Returns
        -------
        OptimizationProblem
            An immutable optimization problem that can be deployed via Bluesky.

        See Also
        --------
        blop.protocols.OptimizationProblem : The optimization problem dataclass.
        blop.plans.optimize : Uses the optimization problem to run optimization.
        """
        return OptimizationProblem(
            optimizer=self._optimizer,
            actuators=self._actuators,
            sensors=self._sensors,
            evaluation_function=self._evaluation_function,
            acquisition_plan=self._acquisition_plan,
        )

    def suggest(self, num_points: int = 1) -> list[dict]:
        """
        Get the next point(s) to evaluate in the search space.

        Uses the Bayesian optimization algorithm to suggest promising points based
        on all previously acquired data. Each suggestion includes an "_id" key for
        tracking.

        Parameters
        ----------
        num_points : int, optional
            The number of points to suggest. Default is 1. Higher values enable
            batch optimization but may reduce optimization efficiency per iteration.

        Returns
        -------
        list[dict]
            A list of dictionaries, each containing a parameterization of a point to
            evaluate next. Each dictionary includes an "_id" key for identification.
        """
        return self._optimizer.suggest(num_points)

    def ingest(self, points: list[dict]) -> None:
        """
        Ingest evaluation results into the optimizer.

        Updates the optimizer's model with new data. Can ingest both suggested points
        (with "_id" key) and external data (without "_id" key).

        Parameters
        ----------
        points : list[dict]
            A list of dictionaries, each containing outcomes for a trial. For suggested
            points, include the "_id" key. For external data, include DOF names and
            objective values, and omit "_id".

        Notes
        -----
        This method is typically called automatically by :meth:`optimize`. Manual usage
        is only needed for custom workflows or when ingesting external data.

        For complete examples, see :doc:`/how-to-guides/attach-data-to-experiments`.
        """
        self._optimizer.ingest(points)

    def optimize(self, iterations=10, n_points=1):
        if self._optimizer.final is not None:
            self.config.initial = self._optimizer.final.x
            self._optimizer = ScipyOptimizer(self.config, timeout=self.timeout)
        optimize_plan = optimize(
            self.to_optimization_problem(),
            iterations=iterations,
            n_points=n_points,
            readable_cache=self._readable_cache,
        )

        if self._callbacks:
            optimize_plan = bpp.subs_wrapper(
                optimize_plan,
                self._callback_router,
            )
        if self.sessioning:
            with self._optimizer:
                yield from optimize_plan
        else:
            yield from optimize_plan

    def get_best_points(self) -> list[tuple[Any, Mapping, Mapping]]:
        """
        Get a list of the optimal points found during optimization.

        For single-objective optimization, returns a single best point.
        For multi-objective optimization, returns the Pareto-optimal set.

        Returns
        -------
        list[tuple[int, TParameterization, TOutcome]]
            Each element in the list is a tuple of:
              - trial index (int)
              - parameter values (dict)
              - metric values (dict, where values may be (value, sem) tuples)

        See Also
        --------
        navigate_to_best : Plan stub to move actuators to a best point.
        """
        return self._optimizer.get_best_points()
