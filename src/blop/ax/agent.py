import importlib.util
import logging
from collections.abc import Sequence
from typing import Any, cast

from ax import Client
from ax.analysis.plotly.surface.contour import ContourPlot
from ax.core.types import TParamValue

# ===============================
# TODO: Remove when Python 3.10 is no longer supported
if importlib.util.find_spec("ax.core.analysis_card") is not None:
    from ax.core.analysis_card import AnalysisCardBase
else:
    from ax.analysis.analysis_card import AnalysisCardBase  # type: ignore[import-untyped]
# ===============================
import bluesky.preprocessors as bpp
from bluesky.callbacks import CallbackBase
from bluesky.utils import MsgGenerator
from bluesky_queueserver_api.zmq import REManagerAPI

from ..callbacks.logger import OptimizationLogger
from ..callbacks.router import OptimizationCallbackRouter
from ..plans import acquire_baseline, optimize, sample_suggestions
from ..protocols import (
    AcquisitionPlan,
    Actuator,
    EvaluationFunction,
    OptimizationProblem,
    QueueserverOptimizationProblem,
    Sensor,
)
from ..queueserver import QueueserverClient, QueueserverOptimizationRunner
from ..utils import InferredReadable
from .dof import DOF, DOFConstraint
from .objective import Objective, OutcomeConstraint, ScalarizedObjective, to_ax_objective_str
from .optimizer import AxOptimizer

logger = logging.getLogger(__name__)


class _AxAgentMixin:
    """
    Mixin providing Ax-related functionality shared by agents.
    Expects subclasses to define `self._optimizer` as an `AxOptimizer`.
    """

    _optimizer: AxOptimizer

    @property
    def ax_client(self) -> Client:
        return self._optimizer.ax_client

    @property
    def checkpoint_path(self) -> str | None:
        return self._optimizer.checkpoint_path

    @property
    def fixed_dofs(self) -> dict[str, Any] | None:
        return self._optimizer.fixed_parameters

    @fixed_dofs.setter
    def fixed_dofs(self, fixed_dofs: dict[DOF, Any] | None) -> None:
        """
        Fix degrees of freedom to a certain value for future optimizations.

        Parameters
        ----------
        fixed_dofs : dict[DOF, Any] | dict[str, Any] | None
            A mapping of DOFs or DOF names to the values they should be fixed to.

        """
        if not fixed_dofs:
            self._optimizer.fixed_parameters = None
            return

        self._optimizer.fixed_parameters = {dof.parameter_name: value for dof, value in fixed_dofs.items()}

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

    def plot_objective(
        self, x_dof_name: str, y_dof_name: str, objective_name: str, *args: Any, **kwargs: Any
    ) -> list[AnalysisCardBase]:
        """
        Plot the predicted objective as a function of two DOFs.

        Creates a contour plot showing the model's prediction of an objective across
        the space defined by two DOFs. Useful for visualizing the optimization landscape.

        Parameters
        ----------
        x_dof_name : str
            The name of the DOF to plot on the x-axis.
        y_dof_name : str
            The name of the DOF to plot on the y-axis.
        objective_name : str
            The name of the objective to plot.
        *args : Any
            Additional positional arguments passed to Ax's compute_analyses.
        **kwargs : Any
            Additional keyword arguments passed to Ax's compute_analyses.

        Returns
        -------
        list[AnalysisCard]
            The computed analysis cards containing the plot data.

        See Also
        --------
        ax.analysis.ContourPlot : Pre-built analysis for plotting objectives.
        ax.analysis.AnalysisCard : Contains the raw and computed data.
        """
        return self.ax_client.compute_analyses(
            [
                ContourPlot(
                    x_parameter_name=x_dof_name,
                    y_parameter_name=y_dof_name,
                    metric_name=objective_name,
                ),
            ],
            *args,
            **kwargs,
        )

    def checkpoint(self) -> None:
        """
        Save the agent's state to a JSON file.
        """
        self._optimizer.checkpoint()

    def reconfigure_search_space(self, dof_mappings: dict[DOF, tuple[float, float] | list[TParamValue]]) -> None:
        """
        Update bounds or values of existing DOFs for future optimizations.

        Parameters
        ----------
        dof_mappings : dict[DOF, tuple[float, float] | list[float] | list[int] | list[str] | list[bool]]
            Mapping of DOFs to their new search space.
        """

        self._optimizer.reconfigure_search_space({dof.parameter_name: update for dof, update in dof_mappings.items()})


class Agent(_AxAgentMixin):
    """
    An interface that uses Ax as the backend for optimization and experiment tracking.

    The Agent is the main entry point for setting up and running Bayesian optimization
    using Blop. It coordinates the DOFs, objectives, evaluation function, and optimizer
    to perform intelligent exploration of the parameter space.

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
    dof_constraints : Sequence[DOFConstraint] | None, optional
        Constraints on DOFs to refine the search space.
    outcome_constraints : Sequence[OutcomeConstraint] | None, optional
        Constraints on outcomes to be satisfied during optimization.
    checkpoint_path : str | None, optional
        The path to the checkpoint file to save the optimizer's state to.
    **kwargs : Any
        Additional keyword arguments to configure the Ax experiment.

    Notes
    -----
    For more complex setups, you can configure the Ax client directly via ``self.ax_client``.

    For complete working examples of creating and using an Agent, see the tutorial
    documentation, particularly :doc:`/tutorials/simple-experiment`.

    See Also
    --------
    blop.protocols.Sensor : The protocol for sensors.
    blop.ax.dof.RangeDOF : For continuous parameters.
    blop.ax.dof.ChoiceDOF : For discrete parameters.
    blop.ax.objective.Objective : For defining objectives.
    blop.ax.optimizer.AxOptimizer : The optimizer used internally.
    blop.plans.optimize : Bluesky plan for running optimization.
    """

    def __init__(
        self,
        sensors: Sequence[Sensor],
        dofs: Sequence[DOF],
        objectives: Sequence[Objective] | ScalarizedObjective,
        evaluation_function: EvaluationFunction,
        acquisition_plan: AcquisitionPlan | None = None,
        dof_constraints: Sequence[DOFConstraint] | None = None,
        outcome_constraints: Sequence[OutcomeConstraint] | None = None,
        checkpoint_path: str | None = None,
        **kwargs: Any,
    ):
        if any(isinstance(dof.actuator, str) for dof in dofs):
            dof_actuator_strs = [dof.actuator for dof in dofs if isinstance(dof.actuator, str)]
            raise ValueError(
                f"DOFs with actuators must be `Actuator` instances, not strings. Got strings for: {dof_actuator_strs}"
            )
        self._sensors = sensors
        self._actuators: Sequence[Actuator] = [cast(Actuator, dof.actuator) for dof in dofs if dof.actuator is not None]
        self._evaluation_function = evaluation_function
        self._acquisition_plan = acquisition_plan
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
    ) -> "Agent":
        """
        Load an agent from the optimizer's checkpoint file.

        .. note::

            Only the optimizer state is saved during a checkpoint, so we cannot reliably validate
            the remaining state against the optimizer configuration.

        Parameters
        ----------
        checkpoint_path : str
            The checkpoint path to load the agent from.
        actuators: Sequence[Actuator]
            Objects that can be moved to control the beamline using the Bluesky RunEngine.
            A subset of the actuators' names must match the names of suggested parameterizations.
        sensors: Sequence[Sensor]
            Objects that can produce data to acquire data from the beamline using the Bluesky RunEngine.
        evaluation_function: EvaluationFunction
            A callable to evaluate data from a Bluesky run and produce outcomes.
        acquisition_plan: AcquisitionPlan, optional
            A Bluesky plan to acquire data from the beamline. If not provided, a default plan will be used.
        """
        instance = object.__new__(cls)
        instance._optimizer = AxOptimizer.from_checkpoint(checkpoint_path)
        instance._actuators = actuators
        instance._sensors = sensors
        instance._evaluation_function = evaluation_function
        instance._acquisition_plan = acquisition_plan
        instance._readable_cache = {}
        instance._callbacks = [OptimizationLogger()]
        instance._callback_router = OptimizationCallbackRouter(instance._callbacks)

        return instance

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
        Construct an optimization problem from the agent.

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
            actuators=self.actuators,
            sensors=self.sensors,
            evaluation_function=self.evaluation_function,
            acquisition_plan=self.acquisition_plan,
        )

    def acquire_baseline(self, parameterization: dict[str, Any] | None = None) -> MsgGenerator[None]:
        """
        Acquire a baseline reading for reference.

        Acquires data at a specific parameterization (or current positions) to establish
        a baseline for comparison. Useful for relative outcome constraints.

        Parameters
        ----------
        parameterization : dict[str, Any] | None, optional
            The DOF values to move to before acquiring baseline. If None, acquires
            at current positions.

        Yields
        ------
        Msg
            Bluesky messages for the run engine.

        See Also
        --------
        blop.plans.acquire_baseline : The underlying Bluesky plan.
        """
        yield from acquire_baseline(self.to_optimization_problem(), parameterization=parameterization)

    def optimize(self, iterations: int = 1, n_points: int = 1) -> MsgGenerator[None]:
        """
        Run Bayesian optimization.

        Performs iterative optimization by suggesting points, acquiring data, evaluating
        outcomes, and updating the model. This is the main method for running optimization
        with an agent.

        Parameters
        ----------
        iterations : int, optional
            The number of optimization iterations to run. Default is 1. Each iteration
            suggests, evaluates, and learns from n_points.
        n_points : int, optional
            The number of points to evaluate per iteration. Default is 1. Higher values
            enable batch optimization but may reduce optimization efficiency per iteration.

        Yields
        ------
        Msg
            Bluesky messages for the run engine.

        Notes
        -----
        This is the primary method for running optimization. It handles the full loop
        of suggesting points, acquiring data, evaluating outcomes, and updating the model.

        For complete examples, see :doc:`/tutorials/simple-experiment`.

        See Also
        --------
        blop.plans.optimize : The underlying Bluesky optimization plan.
        suggest : Get point suggestions without running acquisition.
        ingest : Manually ingest evaluation results.
        """
        optimize_plan = optimize(
            self.to_optimization_problem(), iterations=iterations, n_points=n_points, readable_cache=self._readable_cache
        )
        if self._callbacks:
            optimize_plan = bpp.subs_wrapper(
                optimize_plan,
                self._callback_router,
            )

        yield from optimize_plan

    def sample_suggestions(self, suggestions: list[dict]) -> MsgGenerator[tuple[str, list[dict], list[dict]]]:
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
        tuple[str, list[dict], list[dict]]
            Bluesky run UID, suggestions with "_id", and outcomes.

        See Also
        --------
        suggest : Get optimizer suggestions.
        optimize : Run full optimization loop.
        """
        sample_suggestions_plan = sample_suggestions(
            self.to_optimization_problem(), suggestions=suggestions, readable_cache=self._readable_cache
        )
        if self._callbacks:
            sample_suggestions_plan = bpp.subs_wrapper(
                sample_suggestions_plan,
                self._callback_router,
            )

        return (yield from sample_suggestions_plan)


class QueueserverAgent(_AxAgentMixin):
    """
    An asynchronous interface that uses Ax as the backend for optimization and experiment tracking
    and the bluesky-queueserver-api for scheduling plan execution.

    Parameters
    ----------
    re_manager_api : REManagerAPI
        The manager API for interaction with Bluesky queueserver.
    zmq_consumer_addr : str
        A ZMQ address to consume Bluesky messages from, to react to plan execution on the
        remote server.
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
    blop.queueserver.QueueservverOptimizatonRunner : Runner that handles interaction with bluesky-queueserver.
    """

    def __init__(
        self,
        re_manager_api: REManagerAPI,
        zmq_consumer_addr: str,
        sensors: Sequence[str],
        dofs: Sequence[DOF],
        objectives: Sequence[Objective],
        evaluation_function: EvaluationFunction,
        acquisition_plan: str | None = None,
        dof_constraints: Sequence[DOFConstraint] | None = None,
        outcome_constraints: Sequence[OutcomeConstraint] | None = None,
        checkpoint_path: str | None = None,
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
            QueueserverClient(re_manager_api, zmq_consumer_addr),
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

    def to_optimization_problem(self) -> QueueserverOptimizationProblem:
        return QueueserverOptimizationProblem(
            optimizer=self._optimizer,
            actuators=self._actuators,
            sensors=self._sensors,
            evaluation_function=self._evaluation_function,
            acquisition_plan=self._acquisition_plan,
        )

    def run(self, iterations=1, n_points=1) -> None:
        """
        Start the optimization loop.

        Validates the queueserver state, then begins the suggest -> acquire -> ingest
        cycle. This method returns immediately; the optimization runs asynchronously
        via callbacks.

        Parameters
        ----------
        iterations : int
            Number of optimization iterations to run.
        num_points : int
            Number of points to suggest per iteration.

        Raises
        ------
        RuntimeError
            If the queueserver environment is not ready.
        ValueError
            If required devices or plans are not available.
        """

        self._runner.run(iterations, n_points)

    def submit_suggestions(self, suggestions: list[dict]) -> None:
        """
        Evaluate specific parameter combinations.

        Acquires data for given suggestions and ingests results. Supports both
        optimizer suggestions and manual points.

        Parameters
        ----------
        suggestions : list[dict]
            Either optimizer suggestions (with "_id") or manual points (without "_id").

        See Also
        --------
        suggest : Get optimizer suggestions.
        """
        self._runner.submit_suggestions(suggestions)
