import logging
from collections.abc import Mapping, Sequence
from typing import Any, cast

import bluesky.preprocessors as bpp
from bluesky.callbacks import CallbackBase
from bluesky.utils import MsgGenerator

from ..callbacks.logger import OptimizationLogger
from ..callbacks.router import OptimizationCallbackRouter
from ..plan_stubs import navigate_to_best
from ..plans import acquire_baseline, optimize, sample_suggestions
from ..protocols import AcquisitionPlan, Actuator, EvaluationFunction, OptimizationProblem, Sensor
from ..utils import InferredReadable
from ..ax.dof import DOF, DOFConstraint
from ..ax.objective import Objective, OutcomeConstraint, ScalarizedObjective
from .mapping import build_vocs
from .optimizer import XoptOptimizer

logger = logging.getLogger(__name__)


class XoptAgent:
    """Synchronous blop agent that wraps an arbitrary Xopt generator."""

    def __init__(
        self,
        sensors: Sequence[Sensor],
        dofs: Sequence[DOF],
        objectives: Sequence[Objective] | ScalarizedObjective,
        evaluation_function: EvaluationFunction,
        *,
        generator: Any,
        generator_kwargs: dict[str, Any] | None = None,
        acquisition_plan: AcquisitionPlan | None = None,
        dof_constraints: Sequence[DOFConstraint] | None = None,
        outcome_constraints: Sequence[OutcomeConstraint] | None = None,
        checkpoint_path: str | None = None,
    ):
        if any(isinstance(dof.actuator, str) for dof in dofs):
            dof_actuator_strs = [dof.actuator for dof in dofs if isinstance(dof.actuator, str)]
            raise ValueError(
                f"DOFs with actuators must be `Actuator` instances, not strings. Got strings for: {dof_actuator_strs}"
            )

        vocs = build_vocs(
            dofs=dofs,
            objectives=objectives,
            sensors=sensors,
            dof_constraints=dof_constraints,
            outcome_constraints=outcome_constraints,
        )

        self._sensors = sensors
        self._actuators: Sequence[Actuator] = [cast(Actuator, dof.actuator) for dof in dofs if dof.actuator is not None]
        self._evaluation_function = evaluation_function
        self._acquisition_plan = acquisition_plan
        self._optimizer = XoptOptimizer(
            generator=generator,
            vocs=vocs,
            generator_kwargs=generator_kwargs,
            checkpoint_path=checkpoint_path,
        )
        self._readable_cache: dict[str, InferredReadable] = {}
        self._callbacks: list[CallbackBase] = [OptimizationLogger()]
        self._callback_router = OptimizationCallbackRouter(self._callbacks)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        actuators: Sequence[Actuator],
        sensors: Sequence[Sensor],
        evaluation_function: EvaluationFunction,
        acquisition_plan: AcquisitionPlan | None = None,
    ) -> "XoptAgent":
        instance = object.__new__(cls)
        instance._optimizer = XoptOptimizer.from_checkpoint(checkpoint_path)
        instance._actuators = actuators
        instance._sensors = sensors
        instance._evaluation_function = evaluation_function
        instance._acquisition_plan = acquisition_plan
        instance._readable_cache = {}
        instance._callbacks = [OptimizationLogger()]
        instance._callback_router = OptimizationCallbackRouter(instance._callbacks)
        return instance

    @property
    def checkpoint_path(self) -> str | None:
        return self._optimizer.checkpoint_path

    @property
    def optimizer(self) -> XoptOptimizer:
        """Return the underlying Xopt-backed optimizer adapter."""
        return self._optimizer

    @property
    def fixed_dofs(self) -> dict[str, Any] | None:
        return self._optimizer.fixed_parameters

    @fixed_dofs.setter
    def fixed_dofs(self, fixed_dofs: dict[DOF, Any] | None) -> None:
        if not fixed_dofs:
            self._optimizer.fixed_parameters = None
            return

        self._optimizer.fixed_parameters = {dof.parameter_name: value for dof, value in fixed_dofs.items()}

    def suggest(self, num_points: int = 1) -> list[dict]:
        return self._optimizer.suggest(num_points)

    def ingest(self, points: list[dict]) -> None:
        self._optimizer.ingest(points)

    def get_best_points(self):
        return self._optimizer.get_best_points()

    def checkpoint(self) -> None:
        self._optimizer.checkpoint()

    @property
    def callbacks(self) -> list[CallbackBase]:
        return self._callbacks

    def subscribe(self, callback: CallbackBase) -> None:
        if callback in self._callbacks:
            raise ValueError(f"Callback {callback!r} is already subscribed.")
        self._callbacks.append(callback)

    def unsubscribe(self, callback: CallbackBase) -> None:
        self._callbacks.remove(callback)

    @property
    def sensors(self) -> Sequence[Sensor]:
        return self._sensors

    @property
    def actuators(self) -> Sequence[Actuator]:
        return self._actuators

    @property
    def evaluation_function(self) -> EvaluationFunction:
        return self._evaluation_function

    @property
    def acquisition_plan(self) -> AcquisitionPlan | None:
        return self._acquisition_plan

    def to_optimization_problem(self) -> OptimizationProblem:
        return OptimizationProblem(
            optimizer=self._optimizer,
            actuators=self.actuators,
            sensors=self.sensors,
            evaluation_function=self.evaluation_function,
            acquisition_plan=self.acquisition_plan,
        )

    def acquire_baseline(self, parameterization: dict[str, Any] | None = None) -> MsgGenerator[None]:
        yield from acquire_baseline(self.to_optimization_problem(), parameterization=parameterization)

    def optimize(self, iterations: int = 1, n_points: int = 1) -> MsgGenerator[None]:
        optimize_plan = optimize(
            self.to_optimization_problem(), iterations=iterations, n_points=n_points, readable_cache=self._readable_cache
        )
        if self._callbacks:
            optimize_plan = bpp.subs_wrapper(optimize_plan, self._callback_router)

        yield from optimize_plan

    def sample_suggestions(self, suggestions: list[dict]) -> MsgGenerator[tuple[str, list[dict], list[dict]]]:
        sample_suggestions_plan = sample_suggestions(
            self.to_optimization_problem(), suggestions=suggestions, readable_cache=self._readable_cache
        )
        if self._callbacks:
            sample_suggestions_plan = bpp.subs_wrapper(sample_suggestions_plan, self._callback_router)

        return (yield from sample_suggestions_plan)

    def navigate_to_best(self, parameterization: Mapping | None = None) -> MsgGenerator[None]:
        optimization_problem = self.to_optimization_problem()
        return (
            yield from navigate_to_best(
                optimization_problem.actuators,
                optimization_problem.optimizer,
                parameterization,
            )
        )
