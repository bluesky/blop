from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from threading import Event, Thread
from typing import Any, cast

import bluesky.preprocessors as bpp
import numpy as np
from bluesky.callbacks import CallbackBase
from scipy.optimize import OptimizeResult, dual_annealing, minimize

from blop.ax.dof import RangeDOF
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


class SCP(str, Enum):
    Default = "Default"
    Dual_Annealing = "dual annealing"


@dataclass
class ScipyCFG:
    dofs: Sequence[RangeDOF]
    # dof_constraints: Sequence[DOFConstraint] | None = None
    # outcome_constraints: Sequence[OutcomeConstraint] | None = None
    optimizer: str = "Default"
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
        evaluation_function: EvaluationFunction,
        acquisition_plan: AcquisitionPlan | None = None,
        optimizer: SCP | str = SCP.Default,
        # dof_constraints: Sequence[DOFConstraint] | None = None,  #implemented in future iterations? make to match ax?
        # outcome_constraints: Sequence[OutcomeConstraint] | None = None,
        **kwargs: Any,
    ):

        if optimizer not in SCP:
            raise ValueError(f"optimizer {optimizer} not in supported optimizers:{list(SCP)}")

        config = ScipyCFG(
            dofs=dofs,
            optimizer=optimizer,
            max_iter=kwargs.get("max_iter", None),
            eps=kwargs.get("eps", None),
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

    def __init__(self, config: ScipyCFG):
        self._semaphore = Event()
        self._params: list[str] = []
        self._bounds: list[tuple[Any, Any]] = []
        self._increment: int = 0
        self._objective = None
        self._y = None
        self.intermediate: OptimizeResult | None = None
        self.final: OptimizeResult | None = None
        self.force_resiliance = False
        self._scale = np.ones(len(config.dofs))
        if config.rescale is not None:
            if isinstance(config.rescale, list):
                self._scale = config.rescale
            else:
                self._scale *= config.rescale

        for ind, dof in enumerate(config.dofs):
            self._params.append(dof.parameter_name)
            self._bounds.append(tuple(np.array(dof.bounds) / self._scale[ind]))

        self._x = np.mean(self._bounds, axis=1)
        if config.initial is not None:
            self._x = np.array(config.initial) / self._scale

        def cost(x):
            self._x = x
            self._semaphore.clear()
            self._semaphore.wait()
            if self._y is None:
                raise ValueError("return value is not present")
            return self._y

        def optim_callback(intermediate_result: OptimizeResult):
            self.intermediate = intermediate_result

        kw = {}
        if config.eps is not None:
            kw["eps"] = config.eps
        if config.max_iter is not None:
            kw["max_iter"] = config.max_iter

        if config.optimizer in (SCP.Default):

            def mini_worker():
                self.final = minimize(
                    fun=cost,
                    x0=self._x,
                    bounds=self._bounds,
                    callback=optim_callback,
                    options=kw,
                )
        elif config.optimizer in (SCP.Dual_Annealing):

            def mini_worker():
                self.final = dual_annealing(
                    func=cost,
                    x0=self._x,
                    bounds=self._bounds,
                    callback=optim_callback,
                    minimizer_kwargs=kw,
                )
        else:
            raise NotImplementedError("")

        # self.t = Thread(target=minimize, args=(cost, self._x), kwargs=kw, name="optimizer")
        self.t = Thread(target=mini_worker, name="optimizer")
        self.t.start()

    def get_best_points(self) -> list[tuple[Any, Mapping, Mapping]]:
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
        vector = [x_n * s for s, x_n in zip(self._scale, self._x, strict=True)]
        if self.final is not None:
            vector = [x_n * s for s, x_n in zip(self._scale, self.final.x, strict=True)]

        print("sample:", self._x, " rescaled to:", vector)

        suggestion = dict(zip(self._params, vector, strict=True))
        suggestion[ID_KEY] = self._increment
        self._increment += 1
        return [suggestion]

    def ingest(self, points: list[dict]) -> None:
        """
        Ingest a set of points into the experiment. Either from previously suggested points or from an external source.

        The "_id" key is optional.

        Parameters
        ----------
        points : list[dict]
            A list of dictionaries, each containing the outcomes of each suggested parameterization.
        """
        if self._semaphore.is_set() and not self.force_resiliance:
            raise ValueError("optimizer did not expect to receive an update")
        res = points[0]
        if self._objective is None:
            self._objective = [param for param in res if param not in (*self._params, ID_KEY)][0]
        self._y = res[self._objective]
        self._semaphore.set()
