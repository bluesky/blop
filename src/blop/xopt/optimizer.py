import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
from xopt import VOCS
from xopt.generator import Generator
from xopt.vocs import get_feasibility_data, random_inputs, select_best

from ..protocols import ID_KEY, CanRegisterSuggestions, Checkpointable, Optimizer, TrialFaultAware


def _normalize_trial_id(value: Any) -> int | str:
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return str(value)


def _is_missing_scalar(value: Any) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value))


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

    def _seed_state_from_existing_data(self) -> None:
        # Recover known trial IDs/parameters from existing generator data when available.
        data = self._generator.data
        if not isinstance(data, pd.DataFrame) or data.empty:
            return

        for _, row in data.iterrows():
            # Reuse stored IDs when present, otherwise allocate synthetic IDs.
            if ID_KEY in row and not _is_missing_scalar(row[ID_KEY]):
                trial_id = _normalize_trial_id(row[ID_KEY])
            else:
                trial_id = self._next_id
                self._next_id += 1

            self._params_by_id[trial_id] = {name: row[name] for name in self.vocs.variable_names if name in row}
            if isinstance(trial_id, int):
                self._next_id = max(self._next_id, trial_id + 1)

    def suggest(self, num_points: int | None = None) -> list[dict]:
        # Default to single-point suggestion when caller does not specify cardinality.
        if num_points is None:
            num_points = 1

        # Bootstrap first call with random VOCS inputs to avoid model-based generators requiring prior data.
        data = self._generator.data
        first_suggest_call = self._next_id == 0 and len(self._params_by_id) == 0
        has_data = isinstance(data, pd.DataFrame) and not data.empty

        if first_suggest_call and not has_data:
            suggestions = random_inputs(self.vocs, n=num_points)
        else:
            suggestions = self._generator.suggest(num_points)
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
            raw_trial_id = point.get(ID_KEY)
            if raw_trial_id is None:
                trial_id: int | str = self._next_id
                self._next_id += 1
            else:
                trial_id = _normalize_trial_id(raw_trial_id)

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
        self._generator.ingest(rows)

    def register_failures(self, suggestions: list[dict]) -> None:
        # Remove failed suggestions from pending parameter cache.
        for suggestion in suggestions:
            trial_id = suggestion.get(ID_KEY)
            if trial_id in self._params_by_id:
                self._params_by_id.pop(trial_id)

    def _feasible_mask(self, data: pd.DataFrame) -> pd.Series:
        # Delegate feasibility computation to Xopt's native VOCS helper.
        constraints = self.vocs.constraints
        if not isinstance(constraints, Mapping) or len(constraints) == 0:
            return pd.Series([True] * len(data), index=data.index)

        feasibility = get_feasibility_data(self.vocs, data)
        if "feasible" not in feasibility:
            return pd.Series([True] * len(data), index=data.index)
        return pd.Series(feasibility["feasible"], index=data.index, dtype=bool)

    def _output_names(self) -> list[str]:
        # Outputs include objectives, constraints, and optional observables.
        names = list(self.vocs.objective_names)
        if self.vocs.constraints:
            names.extend(self.vocs.constraint_names)
        if getattr(self.vocs, "observables", None):
            names.extend(self.vocs.observables)
        return names

    def get_best_points(self) -> list[tuple[int | str, Mapping, Mapping]]:
        # Return no points when no data has been ingested.
        data = self._generator.data
        if not isinstance(data, pd.DataFrame) or data.empty:
            return []

        # Best points are only defined over feasible observations.
        candidates: pd.DataFrame = data.loc[self._feasible_mask(data)]
        if len(candidates) == 0:
            return []

        objective_names = list(self.vocs.objective_names)
        # For single-objective problems, return the single extremum according to direction.
        if len(objective_names) == 1 and objective_names[0] in candidates:
            best_indices, _, _ = select_best(self.vocs, candidates, n=1)
            best_index = best_indices[0]
            selected = candidates.loc[[best_index]]
        else:
            # For multi-objective and objective-less modes, return the available candidate set.
            selected = candidates

        output_names = self._output_names()
        results: list[tuple[int | str, Mapping, Mapping]] = []
        for _, row in selected.iterrows():
            # Normalize IDs and split into parameter and outcome mappings.
            trial_id = _normalize_trial_id(row[ID_KEY] if ID_KEY in row else _)

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
            "next_id": self._next_id,
            "params_by_id": self._params_by_id,
        }
        path = Path(self._checkpoint_path)
        with path.open("wb") as stream:
            pickle.dump(payload, stream)
