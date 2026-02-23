from collections.abc import Sequence
from typing import Any

from ax import ChoiceParameterConfig, Client, RangeParameterConfig

from ..protocols import ID_KEY, CanRegisterSuggestions, Checkpointable, Optimizer


class AxOptimizer(Optimizer, Checkpointable, CanRegisterSuggestions):
    """
    An optimizer that uses Ax as the backend for optimization and experiment tracking.

    This is the built-in implementation of the :class:`blop.protocols.Optimizer` protocol.

    Parameters
    ----------
    parameters : Sequence[RangeParameterConfig | ChoiceParameterConfig]
        The parameters to optimize.
    objective : str
        The objective to optimize.
    parameter_constraints : Sequence[str] | None, optional
        The parameter constraints to apply to the optimization.
    outcome_constraints : Sequence[str] | None, optional
        The outcome constraints to apply to the optimization.
    checkpoint_path : str | None, optional
        The path to the checkpoint file to save the optimizer's state to.
    client_kwargs : dict[str, Any] | None, optional
        Additional keyword arguments to configure the Ax client.
    **kwargs : Any
        Additional keyword arguments to configure the Ax experiment.

    See Also
    --------
    blop.ax.Agent : High-level interface that uses AxOptimizer internally.
    blop.protocols.Optimizer : The protocol this class implements.
    """

    def __init__(
        self,
        parameters: Sequence[RangeParameterConfig | ChoiceParameterConfig],
        objective: str,
        parameter_constraints: Sequence[str] | None = None,
        outcome_constraints: Sequence[str] | None = None,
        checkpoint_path: str | None = None,
        client_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        self._parameter_names = [parameter.name for parameter in parameters]
        self._checkpoint_path = checkpoint_path
        self._client = Client(**(client_kwargs or {}))
        self._client.configure_experiment(
            parameters=parameters,
            parameter_constraints=parameter_constraints,
            **kwargs,
        )
        self._client.configure_optimization(
            objective=objective,
            outcome_constraints=outcome_constraints,
        )

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str) -> "AxOptimizer":
        """
        Load an optimizer from a checkpoint file.

        Parameters
        ----------
        checkpoint_path : str
            The path to the checkpoint file to load the optimizer from.

        Returns
        -------
        AxOptimizer
            An instance of the optimizer class, initialized from the checkpoint.
        """
        client = Client.load_from_json_file(checkpoint_path)
        instance = object.__new__(cls)
        instance._parameter_names = list(client._experiment.parameters.keys())
        instance._checkpoint_path = checkpoint_path
        instance._client = client

        return instance

    @property
    def checkpoint_path(self) -> str | None:
        return self._checkpoint_path

    @property
    def ax_client(self) -> Client:
        return self._client

    def suggest(self, num_points: int | None = None) -> list[dict]:
        """
        Get the next point(s) to evaluate in the search space.

        Uses Ax's Bayesian optimization to suggest promising points based on the
        current model and acquisition function.

        Parameters
        ----------
        num_points : int | None, optional
            The number of points to suggest. If not provided, will default to 1.

        Returns
        -------
        list[dict]
            A list of dictionaries, each containing a parameterization of a point to
            evaluate next. Each dictionary includes an "_id" key for tracking.
        """
        if num_points is None:
            num_points = 1
        next_trials = self._client.get_next_trials(max_trials=num_points)
        return [
            {
                "_id": trial_index,
                **parameterization,
            }
            for trial_index, parameterization in next_trials.items()
        ]

    def _split_point(self, point: dict) -> tuple[dict, dict]:
        """Helper function to split a point into parameters and outcomes."""
        parameters = {}
        outcomes = {}
        for k, v in point.items():
            if k == ID_KEY:
                continue
            elif k in self._parameter_names:
                parameters[k] = v
            else:
                outcomes[k] = v
        return parameters, outcomes

    def ingest(self, points: list[dict]) -> None:
        """
        Ingest evaluation results into the optimizer.

        Updates Ax's experiment with new data, which will be used to train the model
        for future suggestions. Handles both suggested points and external data.

        Parameters
        ----------
        points : list[dict]
            A list of dictionaries, each containing outcomes for a trial. For suggested
            points (from :meth:`suggest`), include the "_id" key. For external data,
            include parameter names and objective values, and omit "_id".

        Notes
        -----
        Points with ``"_id": "baseline"`` are treated as baseline trials for reference.
        """
        for point in points:
            trial_idx = point.get(ID_KEY, None)
            parameters, outcomes = self._split_point(point)
            if trial_idx is None:
                trial_idx = self._client.attach_trial(parameters=parameters)
            elif trial_idx == "baseline":
                trial_idx = self._client.attach_baseline(parameters=parameters)
            self._client.complete_trial(trial_index=trial_idx, raw_data=outcomes)

    def register_suggestions(self, suggestions: list[dict]) -> list[dict]:
        """
        Register manual suggestions with the Ax experiment.

        Attaches trials to the experiment and returns the suggestions with "_id" keys
        added for tracking. This enables manual point injection alongside optimizer-driven
        suggestions.

        Parameters
        ----------
        suggestions : list[dict]
            Parameter combinations to register. The "_id" key will be overwritten if present.

        Returns
        -------
        list[dict]
            The same suggestions with "_id" keys added.
        """
        registered = []
        for suggestion in suggestions:
            # Extract parameters (ignore _id if present)
            parameters = {k: v for k, v in suggestion.items() if k != ID_KEY}

            # Attach trial to Ax experiment
            trial_idx = self._client.attach_trial(parameters=parameters)

            # Return with trial ID
            registered.append({ID_KEY: trial_idx, **parameters})

        return registered

    def checkpoint(self) -> None:
        """
        Save the optimizer's state to JSON file.
        """
        if not self.checkpoint_path:
            raise ValueError("Checkpoint path is not set. Please set a checkpoint path when initializing the optimizer.")
        self._client.save_to_json_file(self.checkpoint_path)
