from collections.abc import Mapping, Sequence
from concurrent.futures import Future
from typing import Any

from bluesky.callbacks.zmq import RemoteDispatcher
from bluesky_queueserver_api.zmq import REManagerAPI

from ..protocols import (
    Actuator,
    EvaluationFunction,
    QueueserverOptimizationProblem,
)
from ..queueserver import OptimizationResult, QueueserverClient, QueueserverOptimizationRunner
from .agent import _AxAgentMixin
from .dof import DOF, DOFConstraint
from .objective import Objective, OutcomeConstraint, to_ax_objective_str
from .optimizer import AxOptimizer


class QueueserverAgent(_AxAgentMixin):
    """
    An asynchronous interface that uses Ax as the backend for optimization and experiment tracking
    and the bluesky-queueserver-api for scheduling plan execution.

    .. warning::

        This class is **experimental**. The API is not yet stable and may change
        in future releases without a deprecation period. It is not recommended for
        production use.

    Parameters
    ----------
    re_manager_api : REManagerAPI
        The manager API for interaction with Bluesky queueserver.
    document_dispatcher : RemoteDispatcher
        Dispatcher for consuming Bluesky documents from the remote server.
    sensors : Sequence[str]
        The sensors to use for acquisition. These should be the minimal set
        of sensors that are needed to compute the objectives.
    dofs : Sequence[DOF]
        The degrees of freedom that the agent can control, which determine the search space.
    objectives : Sequence[Objective]
        The objectives which the agent will try to optimize.
    evaluation_function : EvaluationFunction
        The function to evaluate acquired data and produce outcomes.
    acquisition_plan : str | None, optional
        The acquisition plan to use for acquiring data from the beamline. If not provided,
        :func:`blop.plans.default_acquire` will be assumed.
    dof_constraints : Sequence[DOFConstraint] | None, optional
        Constraints on DOFs to refine the search space.
    outcome_constraints : Sequence[OutcomeConstraint] | None, optional
        Constraints on outcomes to be satisfied during optimization.
    checkpoint_path : str | None, optional
        The path to the checkpoint file to save the optimizer's state to.
    **kwargs : Any
        Additional keyword arguments to configure the Ax experiment.

    See Also
    --------
    blop.protocols.Sensor : The protocol for sensors.
    blop.ax.dof.RangeDOF : For continuous parameters.
    blop.ax.dof.ChoiceDOF : For discrete parameters.
    blop.ax.objective.Objective : For defining objectives.
    blop.ax.optimizer.AxOptimizer : The optimizer used internally.
    blop.queueserver.QueueserverOptimizatonRunner : Runner that handles interaction with bluesky-queueserver.
    """

    def __init__(
        self,
        re_manager_api: REManagerAPI,
        document_dispatcher: RemoteDispatcher,
        sensors: Sequence[str],
        dofs: Sequence[DOF],
        objectives: Sequence[Objective],
        evaluation_function: EvaluationFunction,
        acquisition_plan: str | None = None,
        dof_constraints: Sequence[DOFConstraint] | None = None,
        outcome_constraints: Sequence[OutcomeConstraint] | None = None,
        checkpoint_path: str | None = None,
        acquisition_plan_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ):
        self._sensors = sensors
        self._actuators: Sequence[str] = []
        for dof in dofs:
            if dof.actuator is not None:
                if isinstance(dof.actuator, Actuator):
                    self._actuators.append(dof.actuator.name)
                else:
                    self._actuators.append(dof.actuator)
        self._evaluation_function = evaluation_function
        self._acquisition_plan = acquisition_plan
        self._acquisition_plan_kwargs = acquisition_plan_kwargs or {}
        self._optimizer = AxOptimizer(
            parameters=[dof.to_ax_parameter_config() for dof in dofs],
            objective=to_ax_objective_str(objectives),
            parameter_constraints=[constraint.ax_constraint for constraint in dof_constraints] if dof_constraints else None,
            outcome_constraints=[constraint.ax_constraint for constraint in outcome_constraints]
            if outcome_constraints
            else None,
            checkpoint_path=checkpoint_path,
            **kwargs,
        )
        self._runner = QueueserverOptimizationRunner(
            self.to_optimization_problem(),
            QueueserverClient(re_manager_api, document_dispatcher),
        )

    @property
    def evaluation_function(self) -> EvaluationFunction:
        return self._evaluation_function

    @property
    def actuators(self) -> Sequence[str]:
        return self._actuators

    @property
    def sensors(self) -> Sequence[str]:
        return self._sensors

    @property
    def acquisition_plan(self) -> str | None:
        return self._acquisition_plan

    def stop(self) -> None:
        self._runner.stop()

    @property
    def current_iteration(self) -> int:
        return self._runner.current_iteration

    def to_optimization_problem(self) -> QueueserverOptimizationProblem:
        return QueueserverOptimizationProblem(
            optimizer=self._optimizer,
            actuators=self._actuators,
            sensors=self._sensors,
            evaluation_function=self._evaluation_function,
            acquisition_plan=self._acquisition_plan,
            acquisition_plan_kwargs=self._acquisition_plan_kwargs,
        )

    def run(self, iterations: int = 1, n_points: int = 1) -> Future[OptimizationResult]:
        """
        Start the optimization loop.

        Validates the queueserver state, then begins the suggest -> acquire -> ingest
        cycle. This method returns immediately; the optimization runs asynchronously
        via callbacks.

        Parameters
        ----------
        iterations : int
            Number of optimization iterations to run.
        n_points : int
            Number of points to suggest per iteration.

        Returns
        -------
        concurrent.futures.Future[OptimizationResult]
            A future that resolves to an :class:`~blop.queueserver.OptimizationResult`
            when all iterations complete or when :meth:`stop` is called. If an
            unhandled exception occurs the future will hold it and re-raise on
            ``.result()``.

        Raises
        ------
        RuntimeError
            If the queueserver environment is not ready.
        ValueError
            If required devices or plans are not available.
        """
        return self._runner.run(iterations, n_points)

    def submit_suggestions(self, suggestions: list[dict]) -> Future[OptimizationResult]:
        """
        Evaluate specific parameter combinations.

        Acquires data for given suggestions and ingests results. Supports both
        optimizer suggestions and manual points.

        Parameters
        ----------
        suggestions : list[dict]
            Either optimizer suggestions (with "_id") or manual points (without "_id").

        Returns
        -------
        concurrent.futures.Future[OptimizationResult]
            A future that resolves to an :class:`~blop.queueserver.OptimizationResult`
            when the acquisition completes.

        See Also
        --------
        run : Run the full optimization loop.
        """
        return self._runner.submit_suggestions(suggestions)
