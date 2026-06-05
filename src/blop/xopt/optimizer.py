import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
from xopt import VOCS
from xopt.generator import Generator

from ..protocols import ID_KEY, CanRegisterSuggestions, Checkpointable, Optimizer, TrialFaultAware


def _objective_minimize_flag(objective: Any) -> bool:
    # Handle string objective specs first (common VOCS representation).
    if isinstance(objective, str):
        return objective.strip().upper() == "MINIMIZE"

    # Fall back to class-name inspection for typed objective objects.
    objective_name = objective.__class__.__name__.lower()
    if "minimize" in objective_name:
        return True
    if "maximize" in objective_name:
        return False

    return True


def _constraint_satisfied(value: float, op: str, threshold: float) -> bool:
    # Evaluate one normalized constraint against a single numeric value.
    if op == "LESS_THAN":
        return value <= threshold
    if op == "GREATER_THAN":
        return value >= threshold
    raise ValueError(f"Unsupported VOCS constraint operator: {op!r}")


def _constraint_to_pair(constraint: Any) -> tuple[str, float]:
    # Convert common VOCS list/tuple form into a normalized operator/value pair.
    if isinstance(constraint, (list, tuple)) and len(constraint) == 2:
        return str(constraint[0]).upper(), float(constraint[1])

    # Support typed constraint objects from gest_api.vocs by class-name convention.
    name = constraint.__class__.__name__.lower()
    if hasattr(constraint, "value"):
        value = float(constraint.value)
        if "lessthan" in name:
            return "LESS_THAN", value
        if "greaterthan" in name:
            return "GREATER_THAN", value

    raise ValueError(f"Unsupported VOCS constraint representation: {constraint!r}")


class XoptOptimizer(Optimizer, Checkpointable, CanRegisterSuggestions, TrialFaultAware):
    """Adapter that exposes an arbitrary Xopt generator through blop's Optimizer protocol."""

    def __init__(
        self,
        generator: Generator,
        *,
        checkpoint_path: str | None = None,
    ):
        # Keep API simple: caller provides a fully configured Xopt generator instance.
        self._generator = generator

        # Internal state tracks IDs, pending/known parameterizations, and checkpoint metadata.
        self._checkpoint_path = checkpoint_path
        self._fixed_parameters: dict[str, Any] | None = None
        self._next_id = 0
        self._params_by_id: dict[int | str, dict[str, Any]] = {}
        self._seed_state_from_existing_data()

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str) -> "XoptOptimizer":
        # Restore all persistent adapter state from pickle payload.
        path = Path(checkpoint_path)
        with path.open("rb") as stream:
            payload = pickle.load(stream)

        instance = object.__new__(cls)
        instance._generator = payload["generator"]
        instance._checkpoint_path = str(path)
        instance._fixed_parameters = payload.get("fixed_parameters")
        instance._next_id = payload.get("next_id", 0)
        instance._params_by_id = payload.get("params_by_id", {})
        instance._seed_state_from_existing_data()
        return instance

    @property
    def checkpoint_path(self) -> str | None:
        return self._checkpoint_path

    @property
    def generator(self) -> Generator:
        """Return the underlying Xopt generator instance."""
        return self._generator

    @property
    def vocs(self) -> VOCS:
        return self._generator.vocs

    @property
    def fixed_parameters(self) -> dict[str, Any] | None:
        return self._fixed_parameters

    @fixed_parameters.setter
    def fixed_parameters(self, fixed_parameters: dict[str, Any] | None) -> None:
        if not fixed_parameters:
            self._fixed_parameters = None
            return

        unknown_names = set(fixed_parameters) - set(self.vocs.variable_names)
        if unknown_names:
            raise KeyError(f"Unknown fixed parameter(s): {sorted(unknown_names)}")

        self._fixed_parameters = dict(fixed_parameters)

    def _seed_state_from_existing_data(self) -> None:
        # Recover known trial IDs/parameters from existing generator data when available.
        data = self._generator.data
        if data is None or len(data) == 0:
            return

        for _, row in data.iterrows():
            # Reuse stored IDs when present, otherwise allocate synthetic IDs.
            if ID_KEY in row and pd.notna(row[ID_KEY]):
                trial_id = row[ID_KEY]
            else:
                trial_id = self._next_id
                self._next_id += 1

            if isinstance(trial_id, float) and trial_id.is_integer():
                trial_id = int(trial_id)

            self._params_by_id[trial_id] = {name: row[name] for name in self.vocs.variable_names if name in row}
            if isinstance(trial_id, int):
                self._next_id = max(self._next_id, trial_id + 1)

    def suggest(self, num_points: int | None = None) -> list[dict]:
        # Default to single-point suggestion when caller does not specify cardinality.
        if num_points is None:
            num_points = 1

        # Delegate candidate generation to Xopt and optionally enforce fixed variables.
        suggestions = self._generator.generate(num_points)
        if self._fixed_parameters:
            suggestions = [{**suggestion, **self._fixed_parameters} for suggestion in suggestions]
        return self.register_suggestions(suggestions)

    def register_suggestions(self, suggestions: list[dict]) -> list[dict]:
        # Attach stable blop trial IDs and cache suggested parameterizations by ID.
        registered: list[dict] = []
        for suggestion in suggestions:
            trial_id = self._next_id
            self._next_id += 1

            params = {name: suggestion[name] for name in self.vocs.variable_names if name in suggestion}
            self._params_by_id[trial_id] = params
            registered.append({ID_KEY: trial_id, **suggestion})

        return registered

    def ingest(self, points: list[dict]) -> None:
        # Convert outcome payloads to DataFrame rows expected by Xopt generator.add_data().
        rows: list[dict[str, Any]] = []

        for point in points:
            # Preserve provided IDs when available, else allocate a new one.
            trial_id = point.get(ID_KEY)
            if trial_id is None:
                trial_id = self._next_id
                self._next_id += 1

            # Merge known suggested parameters with any explicit parameters in incoming point.
            point_parameters = {name: point[name] for name in self.vocs.variable_names if name in point}
            if trial_id in self._params_by_id:
                parameters = {**self._params_by_id[trial_id], **point_parameters}
            else:
                parameters = point_parameters

            self._params_by_id[trial_id] = parameters
            # Everything not in variables and not _id is treated as measured output.
            outcomes = {k: v for k, v in point.items() if k not in set(self.vocs.variable_names) | {ID_KEY}}
            rows.append({ID_KEY: trial_id, **parameters, **outcomes})

        # Persist all new observations into the underlying generator state.
        new_data = pd.DataFrame(rows)
        self._generator.add_data(new_data)

    def register_failures(self, suggestions: list[dict]) -> None:
        # Remove failed suggestions from pending parameter cache.
        for suggestion in suggestions:
            trial_id = suggestion.get(ID_KEY)
            if trial_id in self._params_by_id:
                self._params_by_id.pop(trial_id)

    def _feasible_mask(self, data: pd.DataFrame) -> pd.Series:
        # Compute row-wise feasibility mask from VOCS constraints.
        if not self.vocs.constraints:
            return pd.Series([True] * len(data), index=data.index)

        mask = pd.Series([True] * len(data), index=data.index)
        for constraint_name, constraint in self.vocs.constraints.items():
            # Missing constraint columns imply non-feasible rows.
            if constraint_name not in data:
                mask &= False
                continue

            op, threshold = _constraint_to_pair(constraint)
            mask &= (
                data[constraint_name]
                .astype(float)
                .apply(lambda x, op=op, threshold=threshold: _constraint_satisfied(x, op, threshold))
            )

        return mask

    def _objective_names(self) -> list[str]:
        # Preserve VOCS-defined objective ordering where available.
        return list(self.vocs.objectives.keys()) if self.vocs.objectives else []

    def _output_names(self) -> list[str]:
        # Outputs include objectives, constraints, and optional observables.
        names = self._objective_names()
        if self.vocs.constraints:
            names.extend(self.vocs.constraints.keys())
        if getattr(self.vocs, "observables", None):
            names.extend(self.vocs.observables)
        return names

    def get_best_points(self) -> list[tuple[int | str, Mapping, Mapping]]:
        # Return no points when no data has been ingested.
        data = self._generator.data
        if data is None or len(data) == 0:
            return []

        # Prefer feasible points first; if none exist, fall back to all data.
        candidates = data[self._feasible_mask(data)]
        if len(candidates) == 0:
            candidates = data

        objective_names = self._objective_names()
        # For single-objective problems, return the single extremum according to direction.
        if len(objective_names) == 1 and objective_names[0] in candidates:
            objective_name = objective_names[0]
            objective_spec = self.vocs.objectives[objective_name]
            minimize = _objective_minimize_flag(objective_spec)
            objective_values = candidates[objective_name].astype(float)
            best_index = objective_values.idxmin() if minimize else objective_values.idxmax()
            selected = candidates.loc[[best_index]]
        else:
            # For multi-objective and objective-less modes, return the available candidate set.
            selected = candidates

        output_names = self._output_names()
        results: list[tuple[int | str, Mapping, Mapping]] = []
        for _, row in selected.iterrows():
            # Normalize IDs and split into parameter and outcome mappings.
            trial_id = row[ID_KEY] if ID_KEY in row else _
            if isinstance(trial_id, float) and trial_id.is_integer():
                trial_id = int(trial_id)

            parameterization = {name: row[name] for name in self.vocs.variable_names if name in row}
            outcomes = {name: row[name] for name in output_names if name in row}
            results.append((trial_id, parameterization, outcomes))

        return results

    def checkpoint(self) -> None:
        # Enforce explicit checkpoint path configuration before writing state.
        if not self._checkpoint_path:
            raise ValueError("Checkpoint path is not set. Please set a checkpoint path when initializing the optimizer.")

        # Persist generator and adapter bookkeeping to a single pickle artifact.
        payload = {
            "generator": self._generator,
            "fixed_parameters": self._fixed_parameters,
            "next_id": self._next_id,
            "params_by_id": self._params_by_id,
        }
        path = Path(self._checkpoint_path)
        with path.open("wb") as stream:
            pickle.dump(payload, stream)
