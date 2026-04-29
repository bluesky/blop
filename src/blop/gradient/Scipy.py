from collections.abc import Sequence
from dataclasses import dataclass
from threading import Event, thread
from typing import Any, cast

from scipy.optimize import Bounds, dual_annealing, minimize

from blop.ax.dof import RangeDOF, DOFConstraint
from blop.ax.objective import OutcomeConstraint

from ..protocols import AcquisitionPlan, Actuator, EvaluationFunction, OptimizationProblem, Optimizer, Sensor


@dataclass
class ScpCFG:
    dof: Sequence[RangeDOF]
    # dof_constraints: Sequence[DOFConstraint] | None = None
    outcome_constraints: Sequence[OutcomeConstraint] | None = None
    Optimizer: str = "Default"
    max_iter: int | None = None
    eps: float | None = None


class Scipy:
    def __init__(
        self,
        sensors: Sequence[Sensor],
        dofs: Sequence[RangeDOF],
        evaluation_function: EvaluationFunction,
        acquisition_plan: AcquisitionPlan | None = None,
        # dof_constraints: Sequence[DOFConstraint] | None = None,
        outcome_constraints: Sequence[OutcomeConstraint] | None = None,
        checkpoint_path: str | None = None,
        **kwargs: Any,
    ):
        self._sensors = sensors
        self._actuators: Sequence[Actuator] = [cast(Actuator, dof.actuator) for dof in dofs if dof.actuator is not None]
        self._evaluation_function = evaluation_function
        self._acquisition_plan = acquisition_plan
        self._params = []
        self._bounds = 
        for dof in dofs:
            self._params.append(dof.parameter_name)
            self.bounds.append(Bounds(lb=param.bounds[0], ub=param.bounds[1]))

    @classmethod
    def configure(cls, config: ScipyCFG):
        self = cls()
        self.cfg = config
        return cls()

    def to_optimization_problem(self) -> OptimizationProblem:
        ...

    def optimize():
        ...


class ScipyOptimizer(Optimizer):

    def __init__(self, ScpCFG):
        ...

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
