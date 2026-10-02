"""Agent interface for asynchronous optimization using Bluesky Queueserver and Ax."""

from collections.abc import Mapping, Sequence
from concurrent.futures import Future
from typing import Any

try:
    import bluesky_queueserver_api.http
    import bluesky_queueserver_api.zmq
except ImportError as e:
    raise ImportError(
        "The queueserver integration requires additional dependencies. Install them with: pip install blop[queueserver]"
    ) from e

from ..protocols import (
    Actuator,
    EvaluationFunction,
    QueueserverOptimizationProblem,
)
from ..queueserver import OptimizationResult, QueueserverAcquisition, QueueserverClient, QueueserverOptimizationRunner
from .agent import _AxAgentMixin
from .dof import DOF, DOFConstraint
from .objective import Objective, OutcomeConstraint, to_ax_objective_str
from .optimizer import AxOptimizer


class QueueserverAgent(_AxAgentMixin):
    """
    An asynchronous interface that uses Ax as the backend for optimization and experiment tracking.

    Uses the bluesky-queueserver-api for scheduling plan execution. The evaluator
    receives an acquisition token immediately after submission and owns waiting
    for the data it needs; no document dispatcher is required.

    .. warning::

        This class is **experimental**. The API is not yet stable and may change
        in future releases without a deprecation period. It is not recommended for
        production use.

    Parameters
    ----------
    re_manager_api : bluesky_queueserver_api.zmq.REManagerAPI | bluesky_queueserver_api.http.REManagerAPI
        The manager API for interaction with Bluesky queueserver.
    sensors : Sequence[str]
        The sensors to use for acquisition. These should be the minimal set
        of sensors that are needed to compute the objectives.
    dofs : Sequence[DOF]
        The degrees of freedom that the agent can control, which determine the search space.
    objectives : Sequence[Objective]
        The objectives which the agent will try to optimize.
    evaluation_function : EvaluationFunction[QueueserverAcquisition]
        Evaluates a submission token and its suggestions, waiting for data readiness
        before returning outcomes. The token is not a Bluesky run UID.
    acquisition_plan : str | None, optional
        The acquisition plan to use for acquiring data from the beamline. If not provided,
        :func:`blop.plans.default_acquire` will be assumed.
    dof_constraints : Sequence[DOFConstraint] | None, optional
        Constraints on DOFs to refine the search space.
    outcome_constraints : Sequence[OutcomeConstraint] | None, optional
        Constraints on outcomes to be satisfied during optimization.
    checkpoint_path : str | None, optional
        The path to the checkpoint file to save the optimizer's state to.
    acquisition_plan_kwargs : Mapping[str, Any] | None, optional
        Additional keyword arguments passed to the acquisition plan.
    **kwargs : Any
        Additional keyword arguments to configure the Ax experiment.

    See Also
    --------
    blop.protocols.Sensor : The protocol for sensors.
    blop.ax.dof.RangeDOF : For continuous parameters.
    blop.ax.dof.ChoiceDOF : For discrete parameters.
    blop.ax.objective.Objective : For defining objectives.
    blop.ax.optimizer.AxOptimizer : The optimizer used internally.
    blop.queueserver.QueueserverOptimizationRunner : Runner that handles interaction with bluesky-queueserver.

    Notes
    -----
    Existing run-UID evaluators can be wrapped with
    :class:`blop.queueserver.DocumentStreamEvaluator`. Construct the adapter before
    submission with an application-managed dispatcher, use ``contextlib.closing``
    to own its subscription, and consume ``future.result()`` inside that scope.
    The agent does not start or stop the dispatcher or close the evaluator.
    A completed future reflects evaluator readiness and ingestion, not necessarily
    completion of the Queue Server plan's cleanup.
    """

    def __init__(
        self,
        re_manager_api: bluesky_queueserver_api.zmq.REManagerAPI | bluesky_queueserver_api.http.REManagerAPI,
        sensors: Sequence[str],
        dofs: Sequence[DOF],
        objectives: Sequence[Objective],
        evaluation_function: EvaluationFunction[QueueserverAcquisition],
        acquisition_plan: str | None = None,
        dof_constraints: Sequence[DOFConstraint] | None = None,
        outcome_constraints: Sequence[OutcomeConstraint] | None = None,
        checkpoint_path: str | None = None,
        acquisition_plan_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
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
            QueueserverClient(re_manager_api),
        )

    @property
    def evaluation_function(self) -> EvaluationFunction[QueueserverAcquisition]:
        """Evaluate a submission token and suggestions once the needed data is ready."""
        return self._evaluation_function

    @property
    def actuators(self) -> Sequence[str]:
        """Set of actuator names used during acquisition."""
        return self._actuators

    @property
    def sensors(self) -> Sequence[str]:
        """Set of sensor names used during acquisition."""
        return self._sensors

    @property
    def acquisition_plan(self) -> str | None:
        """Acquisition plan name used during acquisition."""
        return self._acquisition_plan

    def stop(self) -> None:
        """
        Prevent later acquisitions without interrupting in-flight evaluation.

        This may wait for an already-started submission, but not for evaluation.
        The current future remains running through evaluation, ingestion,
        checkpointing, and any pending failure notification. New work is rejected
        until that future completes. This does not stop or abort a Queue Server
        plan, close the evaluator, or stop a document transport.
        """
        self._runner.stop()

    @property
    def current_iteration(self) -> int:
        """The current iteration of the optimization."""
        return self._runner.current_iteration

    def to_optimization_problem(self) -> QueueserverOptimizationProblem[QueueserverAcquisition]:
        """Convert the agent state to an optimization problem."""
        return QueueserverOptimizationProblem(
            optimizer=self._optimizer,
            actuators=self._actuators,
            sensors=self._sensors,
            evaluation_function=self._evaluation_function,
            acquisition_plan=self._acquisition_plan,
            acquisition_plan_kwargs=self._acquisition_plan_kwargs,
        )

    def run(
        self, iterations: int = 1, n_points: int = 1, checkpoint_interval: int | None = None
    ) -> Future[OptimizationResult]:
        """
        Start the optimization loop.

        Checks the queueserver environment synchronously, then runs the
        suggest -> submit -> evaluate -> ingest cycle in a background worker.
        The evaluator receives a submission token immediately after submission
        and is responsible for data readiness.

        Parameters
        ----------
        iterations : int
            Number of optimization iterations to run.
        n_points : int
            Number of points to suggest per iteration.
        checkpoint_interval : int | None
            The number of iterations between optimizer checkpoints. If None, checkpoints
            will not be saved. Optimizer must implement the
            :class:`blop.protocols.Checkpointable` protocol.

        Returns
        -------
        concurrent.futures.Future[OptimizationResult]
            A future resolving after all iterations, or after a stop request and
            completion of in-flight evaluation, ingestion, checkpointing, and
            pending failure notification. The result's ``uids`` are acquisition
            tokens for successfully evaluated and ingested batches, not run UIDs.
            Suggestion, submission, evaluation, ingestion, and checkpoint errors
            are raised by ``future.result()``, not by this method.

        Raises
        ------
        RuntimeError
            Synchronously if the queueserver environment is not ready or an
            optimization future is still running, including after a stop request.
        """
        return self._runner.run(iterations=iterations, num_points=n_points, checkpoint_interval=checkpoint_interval)

    def submit_suggestions(self, suggestions: Sequence[Mapping]) -> Future[OptimizationResult]:
        """
        Evaluate specific parameter combinations.

        Registers the points with Ax, submits an acquisition, evaluates its token,
        and ingests outcomes in a background worker. The evaluator owns waiting
        for data readiness. Supports both optimizer suggestions and manual points.

        Parameters
        ----------
        suggestions : Sequence[Mapping]
            Optimizer suggestions (with "_id") or manual points (without "_id").
            Ax registers each point and assigns a new ID, replacing any supplied ID.

        Returns
        -------
        concurrent.futures.Future[OptimizationResult]
            A future resolving after evaluation, ingestion, and pending failure
            notification, even if :meth:`stop` is called meanwhile. The result's
            ``uids`` contain acquisition tokens, not run UIDs. Registration,
            submission, evaluation, and ingestion errors are raised by
            ``future.result()``.

        Raises
        ------
        RuntimeError
            Synchronously if the queueserver environment is not ready or an
            optimization future is still running, including after a stop request.

        See Also
        --------
        run : Run the full optimization loop.
        """
        return self._runner.submit_suggestions(suggestions)
