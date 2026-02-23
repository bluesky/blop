import importlib.util
import logging
from collections.abc import Sequence
from typing import Any

from ax import Client
from ax.analysis import ContourPlot

# ===============================
# TODO: Remove when Python 3.10 is no longer supported
if importlib.util.find_spec("ax.core.analysis_card") is not None:
    from ax.core.analysis_card import AnalysisCardBase
else:
    from ax.analysis.analysis_card import AnalysisCardBase  # type: ignore[import-untyped]
# ===============================
from bluesky.utils import MsgGenerator

from ..plans import acquire_baseline, optimize, sample_suggestions
from ..plans.utils import InferredReadable
from ..protocols import AcquisitionPlan, Actuator, EvaluationFunction, OptimizationProblem, Sensor
from .dof import DOF, DOFConstraint
from .objective import Objective, OutcomeConstraint, to_ax_objective_str
from .optimizer import AxOptimizer

logger = logging.getLogger(__name__)


class Agent:
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
        objectives: Sequence[Objective],
        evaluation_function: EvaluationFunction,
        acquisition_plan: AcquisitionPlan | None = None,
        dof_constraints: Sequence[DOFConstraint] | None = None,
        outcome_constraints: Sequence[OutcomeConstraint] | None = None,
        checkpoint_path: str | None = None,
        **kwargs: Any,
    ):
        self._sensors = sensors
        self._actuators = [dof.actuator for dof in dofs if dof.actuator is not None]
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

        return instance

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

    @property
    def ax_client(self) -> Client:
        return self._optimizer.ax_client

    @property
    def checkpoint_path(self) -> str | None:
        return self._optimizer.checkpoint_path

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
        yield from optimize(
            self.to_optimization_problem(), iterations=iterations, n_points=n_points, readable_cache=self._readable_cache
        )

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
        return (
            yield from sample_suggestions(
                self.to_optimization_problem(), suggestions=suggestions, readable_cache=self._readable_cache
            )
        )

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
