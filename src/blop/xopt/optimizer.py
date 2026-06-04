import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
from xopt import VOCS
from xopt.generator import Generator

from ..protocols import ID_KEY, CanRegisterSuggestions, Checkpointable, Optimizer, TrialFaultAware


def _objective_minimize_flag(objective: Any) -> bool:
    if isinstance(objective, str):
        return objective.strip().upper() == "MINIMIZE"

    objective_name = objective.__class__.__name__.lower()
    if "minimize" in objective_name:
        return True
    if "maximize" in objective_name:
        return False

    return True


def _constraint_satisfied(value: float, op: str, threshold: float) -> bool:
    if op == "LESS_THAN":
        return value <= threshold
    if op == "GREATER_THAN":
        return value >= threshold
    raise ValueError(f"Unsupported VOCS constraint operator: {op!r}")


def _constraint_to_pair(constraint: Any) -> tuple[str, float]:
    if isinstance(constraint, (list, tuple)) and len(constraint) == 2:
        return str(constraint[0]).upper(), float(constraint[1])

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
        generator: Generator | type[Generator],
        *,
        vocs: VOCS | None = None,
        generator_kwargs: dict[str, Any] | None = None,
        checkpoint_path: str | None = None,
    ):
        generator_kwargs = generator_kwargs or {}

        if isinstance(generator, type):
            if vocs is None and "vocs" not in generator_kwargs:
                raise ValueError("vocs must be provided when initializing XoptOptimizer with a generator class.")
            if "vocs" not in generator_kwargs:
                self._generator = generator(vocs=vocs, **generator_kwargs)
            else:
                self._generator = generator(**generator_kwargs)
        else:
            self._generator = generator
            if vocs is not None and self._generator.vocs != vocs:
                raise ValueError("Provided vocs does not match generator.vocs.")

        self._checkpoint_path = checkpoint_path
        self._fixed_parameters: dict[str, Any] | None = None
        self._next_id = 0
        self._params_by_id: dict[int | str, dict[str, Any]] = {}
        self._seed_state_from_existing_data()

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str) -> "XoptOptimizer":
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
        data = self._generator.data
        if data is None or len(data) == 0:
            return

        for _, row in data.iterrows():
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
        if num_points is None:
            num_points = 1

        suggestions = self._generator.generate(num_points)
        if self._fixed_parameters:
            suggestions = [{**suggestion, **self._fixed_parameters} for suggestion in suggestions]
        return self.register_suggestions(suggestions)

    def register_suggestions(self, suggestions: list[dict]) -> list[dict]:
        registered: list[dict] = []
        for suggestion in suggestions:
            trial_id = self._next_id
            self._next_id += 1

            params = {name: suggestion[name] for name in self.vocs.variable_names if name in suggestion}
            self._params_by_id[trial_id] = params
            registered.append({ID_KEY: trial_id, **suggestion})

        return registered

    def ingest(self, points: list[dict]) -> None:
        rows: list[dict[str, Any]] = []

        for point in points:
            trial_id = point.get(ID_KEY)
            if trial_id is None:
                trial_id = self._next_id
                self._next_id += 1

            point_parameters = {name: point[name] for name in self.vocs.variable_names if name in point}
            if trial_id in self._params_by_id:
                parameters = {**self._params_by_id[trial_id], **point_parameters}
            else:
                parameters = point_parameters

            self._params_by_id[trial_id] = parameters
            outcomes = {k: v for k, v in point.items() if k not in set(self.vocs.variable_names) | {ID_KEY}}
            rows.append({ID_KEY: trial_id, **parameters, **outcomes})

        new_data = pd.DataFrame(rows)
        self._generator.add_data(new_data)

    def register_failures(self, suggestions: list[dict]) -> None:
        for suggestion in suggestions:
            trial_id = suggestion.get(ID_KEY)
            if trial_id in self._params_by_id:
                self._params_by_id.pop(trial_id)

    def _feasible_mask(self, data: pd.DataFrame) -> pd.Series:
        if not self.vocs.constraints:
            return pd.Series([True] * len(data), index=data.index)

        mask = pd.Series([True] * len(data), index=data.index)
        for constraint_name, constraint in self.vocs.constraints.items():
            if constraint_name not in data:
                mask &= False
                continue

            op, value = _constraint_to_pair(constraint)
            mask &= data[constraint_name].astype(float).apply(lambda x: _constraint_satisfied(x, op, value))

        return mask

    def _objective_names(self) -> list[str]:
        return list(self.vocs.objectives.keys()) if self.vocs.objectives else []

    def _output_names(self) -> list[str]:
        names = self._objective_names()
        if self.vocs.constraints:
            names.extend(self.vocs.constraints.keys())
        if getattr(self.vocs, "observables", None):
            names.extend(self.vocs.observables)
        return names

    def get_best_points(self) -> list[tuple[int | str, Mapping, Mapping]]:
        data = self._generator.data
        if data is None or len(data) == 0:
            return []

        candidates = data[self._feasible_mask(data)]
        if len(candidates) == 0:
            candidates = data

        objective_names = self._objective_names()
        if len(objective_names) == 1 and objective_names[0] in candidates:
            objective_name = objective_names[0]
            objective_spec = self.vocs.objectives[objective_name]
            minimize = _objective_minimize_flag(objective_spec)
            objective_values = candidates[objective_name].astype(float)
            best_index = objective_values.idxmin() if minimize else objective_values.idxmax()
            selected = candidates.loc[[best_index]]
        else:
            selected = candidates

        output_names = self._output_names()
        results: list[tuple[int | str, Mapping, Mapping]] = []
        for _, row in selected.iterrows():
            trial_id = row[ID_KEY] if ID_KEY in row else _
            if isinstance(trial_id, float) and trial_id.is_integer():
                trial_id = int(trial_id)

            parameterization = {name: row[name] for name in self.vocs.variable_names if name in row}
            outcomes = {name: row[name] for name in output_names if name in row}
            results.append((trial_id, parameterization, outcomes))

        return results

    def checkpoint(self) -> None:
        if not self._checkpoint_path:
            raise ValueError("Checkpoint path is not set. Please set a checkpoint path when initializing the optimizer.")

        payload = {
            "generator": self._generator,
            "fixed_parameters": self._fixed_parameters,
            "next_id": self._next_id,
            "params_by_id": self._params_by_id,
        }
        path = Path(self._checkpoint_path)
        with path.open("wb") as stream:
            pickle.dump(payload, stream)
