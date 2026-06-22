from collections import OrderedDict
from collections.abc import Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from enum import StrEnum
from threading import Thread
from typing import Any, cast

import bluesky.preprocessors as bpp
import numpy as np
from bluesky.callbacks import CallbackBase
from scipy.optimize import OptimizeResult, dual_annealing, minimize

from blop.ax.dof import RangeDOF
from blop.ax.objective import Objective
from blop.callbacks.logger import OptimizationLogger
from blop.callbacks.router import OptimizationCallbackRouter
from blop.plans import optimize
from blop.utils import InferredReadable

from ..protocols import (
    ID_KEY,
    AcquisitionPlan,
    Actuator,
    EvaluationFunction,
    OptimizationProblem,
    Optimizer,
    Sensor,
)


class SCP(StrEnum):
    Default = "Default"
    Dual_Annealing = "dual annealing"


@dataclass
class ScipyCFG:
    dofs: Sequence[RangeDOF]
    objective: Objective
    # dof_constraints: Sequence[DOFConstraint] | None = None
    # outcome_constraints: Sequence[OutcomeConstraint] | None = None
    optimizer: SCP = SCP.Default
    initial: Sequence[float] | None = None
    rescale: Sequence[float] | float | None = None
    max_iter: int | None = 100
    eps: float | None = None


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

        self._config = config
        self._sensors = sensors
        self._actuators = [cast(Actuator, dof.actuator) for dof in config.dofs if dof.actuator is not None]
        self._evaluation_function = evaluation_function
        self._acquisition_plan = acquisition_plan
        self._optimizer = ScipyOptimizer(self._config)
        self._readable_cache: dict[str, InferredReadable] = {}
        self._callbacks: list[CallbackBase] = [OptimizationLogger()]
        self._callback_router = OptimizationCallbackRouter(self._callbacks)

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
        return cls(sensors, config, evaluation_function, acquisition_plan)

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

    def optimize(self, iterations=10):
        if self._optimizer.final is not None:
            self._optimizer = ScipyOptimizer(self._config)
        optimize_plan = optimize(
            self.to_optimization_problem(),
            iterations=iterations,
            readable_cache=self._readable_cache,
        )

        if self._callbacks:
            optimize_plan = bpp.subs_wrapper(
                optimize_plan,
                self._callback_router,
            )

        yield from optimize_plan


class ScipyOptimizer(Optimizer):
    """
    An optimizer object to supply an interactive interface for the scipy optimizers, with some caveats.
    """

    @dataclass
    class Request:
        args: tuple
        future: Future

    @dataclass
    class Result:
        x: list
        fun: float
        nit: int
        status: int = 2

    def __init__(self, config: ScipyCFG):
        self._params: list[str] = []
        self._bounds: list[tuple[Any, Any]] = []
        self._increment: int = 0
        self._objective: Objective = config.objective
        self.force_resiliance = False  # kinda hidden for now
        self._scale = np.ones(len(config.dofs))
        self._active: dict[int, ScipyOptimizer.Request] = OrderedDict()
        self.intermediate: OptimizeResult | ScipyOptimizer.Result | None = None
        self.final: OptimizeResult | ScipyOptimizer.Result | None = None

        if config.rescale is not None:
            if isinstance(config.rescale, list):
                self._scale = config.rescale
            else:
                self._scale *= config.rescale

        for ind, dof in enumerate(config.dofs):
            self._params.append(dof.parameter_name)
            self._bounds.append(tuple(np.array(dof.bounds) / self._scale[ind]))

        _x = np.mean(self._bounds, axis=1)
        if config.initial is not None:
            _x = np.array(config.initial) / self._scale

        def cost(x):
            """
            simple cooperative thread that defers evaluation of cost call by scipy to the run engine
            """
            req = self.Request(args=x, future=Future())
            self._active[self._increment] = req
            self._increment += 1
            res = req.future.result()
            if res is None:
                raise ValueError("return value is not present")
            return res

        kw = {}

        if config.optimizer in (SCP.Default):
            if config.max_iter is not None:
                kw["max_iter"] = config.max_iter
            if config.eps is not None:
                kw["eps"] = config.eps

            def default_callback(intermediate_result: OptimizeResult):
                self.intermediate = intermediate_result

            def mini_worker():
                self.final = minimize(
                    fun=cost,
                    x0=_x,
                    bounds=self._bounds,
                    callback=default_callback,
                    options=kw,
                )
        elif config.optimizer in (SCP.Dual_Annealing):

            def dual_callback(x, f, context):
                self.intermediate = self.Result(x, f, self._increment, context)

            def mini_worker():
                self.final = dual_annealing(
                    func=cost,
                    x0=_x,
                    bounds=self._bounds,
                    callback=dual_callback,
                    minimizer_kwargs=kw,
                )
        else:
            raise NotImplementedError("")

        self._t = Thread(target=mini_worker, name="optimizer")
        self._t.start()

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
        if self.final is not None:
            vector = [x_n * s for s, x_n in zip(self._scale, self.final.x, strict=True)]
            suggestion = dict(zip(self._params, vector, strict=True))
            suggestion[ID_KEY] = self.final.nit
            return [suggestion]

        suggestions = []
        for id in list(self._active.keys())[: num_points if num_points is not None else 1]:
            x = self._active[id].args
            vector = [x_n * s for s, x_n in zip(self._scale, x, strict=True)]

            suggestion = dict(zip(self._params, vector, strict=True))
            suggestion[ID_KEY] = id
            suggestions.append(suggestion)
        return suggestions

    def ingest(self, points: list[dict]) -> None:
        """
        Ingest a set of points into the experiment. Either from previously suggested points or from an external source.

        The "_id" key is optional.

        Parameters
        ----------
        points : list[dict]
            A list of dictionaries, each containing the outcomes of each suggested parameterization.
        """
        for res in points:
            if self._objective is None:
                self._objective = [param for param in res if param not in (*self._params, ID_KEY)][0]
            y = res[self._objective]
            if res[ID_KEY] not in self._active:
                if not self.force_resiliance:
                    raise ValueError("optimizer did not expect to receive an update")
                continue
            self._active.pop(res[ID_KEY]).future.set_result(y)

    def get_best_points(self) -> list[tuple[Any, Mapping, Mapping]]:
        """
        Get a list of the optimal point found during optimization.

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
        result = self.intermediate
        if self.final is not None:
            result = self.final
        if (result is None) or (self._objective is None):
            raise ValueError("no optimization epoch has been recorded")

        vector = [x_n * s for s, x_n in zip(self._scale, result.x, strict=True)]
        cart = [
            result.nit - 1,
            cast(Mapping, dict(zip(self._params, vector, strict=True))),
            cast(Mapping, {self._objective: result.fun}),
        ]
        return cart
