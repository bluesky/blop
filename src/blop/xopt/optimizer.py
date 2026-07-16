"""Optimizer implementation using Xopt as the optimization backend."""

import json
from collections.abc import Mapping
from importlib import import_module
from pathlib import Path
from typing import Any

import pandas as pd
from xopt import VOCS
from xopt.generator import Generator
from xopt.vocs import FeasibilityError, random_inputs, select_best

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
        """Load optimizer from a checkpoint."""
        # Restore all persistent adapter state from JSON payload.
        path = Path(checkpoint_path)
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)

        instance = object.__new__(cls)

        generator_class_name = payload["generator"]["class"]
        generator_module_name = payload["generator"]["module"]
        generator_state = payload["generator"]["state"]
        generator_data = payload["generator"].get("data")

        # Dynamically import the generator class from its module.
        generator_module = import_module(generator_module_name)
        generator_class = getattr(generator_module, generator_class_name)
        instance._generator = generator_class.model_validate(generator_state)
        if generator_data is not None:
            instance._generator.ingest(generator_data)

        instance._checkpoint_path = str(path)
        instance._next_id = payload.get("next_id", 0)
        instance._params_by_id = payload.get("params_by_id", {})
        instance._seed_state_from_existing_data()
        return instance

    @property
    def checkpoint_path(self) -> str | None:
        """Checkpoint path for saving optimizer state."""
        return self._checkpoint_path

    @property
    def generator(self) -> Generator:
        """The underlying Xopt generator instance."""
        return self._generator

    @property
    def vocs(self) -> VOCS:
        """Xopt VOCS structure."""
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

    def _generator_has_data(self) -> bool:
        data = self._generator.data
        return isinstance(data, pd.DataFrame) and not data.empty

    def suggest(self, num_points: int | None = None) -> list[dict]:
        """Suggested points to sample from the optimizer."""
        # Default to single-point suggestion when caller does not specify cardinality.
        if num_points is None:
            num_points = 1

        # Bootstrap first call with random VOCS inputs to avoid model-based generators requiring prior data.
        first_suggest_call = self._next_id == 0 and not self._params_by_id
        has_data = self._generator_has_data()

        if first_suggest_call and not has_data:
            suggestions = random_inputs(self.vocs, n=num_points)
        else:
            suggestions = self._generator.suggest(num_points)
        return self.register_suggestions(suggestions)

    def register_suggestions(self, suggestions: list[dict]) -> list[dict]:
        """Register external suggestions with the optimizer."""
        # Attach stable blop trial IDs and cache suggested parameterizations by ID.
        registered: list[dict] = []
        for suggestion in suggestions:
            trial_id = self._next_id
            self._next_id += 1

            params = {name: suggestion[name] for name in self.vocs.variable_names if name in suggestion}
            self._params_by_id[trial_id] = params
            registered.append({ID_KEY: trial_id, **suggestion})

        # add the registered suggestions to the generator's data
        self._generator.ingest(registered)

        return registered

    def ingest(self, points: list[dict]) -> None:
        """Ingest outcomes into the optimizer."""
        if not points:
            return

        # Convert outcomes into trial rows, keeping only the latest entry per trial ID.
        variable_names = list(self.vocs.variable_names)
        variable_name_set = set(variable_names)
        rows_by_id: dict[int | str, dict[str, Any]] = {}

        for point in points:
            # Preserve provided IDs when available, else allocate a new one.
            raw_trial_id = point.get(ID_KEY)
            if raw_trial_id is None:
                trial_id: int | str = self._next_id
                self._next_id += 1
            else:
                trial_id = _normalize_trial_id(raw_trial_id)

            # Merge known suggested parameters with any explicit parameters in incoming point.
            point_parameters = {name: point[name] for name in variable_names if name in point}
            parameters = {**self._params_by_id.get(trial_id, {}), **point_parameters}

            self._params_by_id[trial_id] = parameters
            # Everything not in variables and not _id is treated as measured output.
            outcomes = {k: v for k, v in point.items() if k not in variable_name_set | {ID_KEY}}
            rows_by_id[trial_id] = {ID_KEY: trial_id, **parameters, **outcomes}

        # Update existing trial rows in-place by ID; only append truly new IDs.
        data = self._generator.data
        if not (isinstance(data, pd.DataFrame) and not data.empty and ID_KEY in data.columns):
            # Persist all observations when no existing data is present.
            self._generator.ingest(list(rows_by_id.values()))
            return

        # Build a mapping from trial ID to row indices in the existing generator data.
        index_by_id: dict[int | str, list] = {}
        for index, raw_trial_id in data[ID_KEY].items():
            trial_id = _normalize_trial_id(raw_trial_id)
            index_by_id.setdefault(trial_id, []).append(index)

        # Update existing rows in-place and collect new rows to append.
        rows_to_append: list[dict[str, Any]] = []
        for trial_id, row in rows_by_id.items():
            indices = index_by_id.get(trial_id)
            if indices is None:
                rows_to_append.append(row)
                continue

            for key, value in row.items():
                data.loc[indices, key] = value

        # Append any new rows to the generator's data after in-place updates.
        if rows_to_append:
            self._generator.ingest(rows_to_append)

    def register_failures(self, suggestions: list[dict]) -> None:
        """Register failures with the optimizer."""
        # Remove failed suggestions from pending parameter cache.
        for suggestion in suggestions:
            trial_id = suggestion.get(ID_KEY)
            if trial_id is not None:
                self._params_by_id.pop(trial_id, None)

    def get_best_points(self) -> list[tuple[int | str, Mapping, Mapping]]:
        """Best points from the optimizer."""
        # Return no points when no data has been ingested.
        if not self._generator_has_data():
            return []

        data = self._generator.data
        if not isinstance(data, pd.DataFrame):
            return []

        objective_names = list(self.vocs.objective_names)
        # For single-objective problems, return the single extremum according to direction.
        if len(objective_names) == 1 and objective_names[0] in data:
            try:
                best_indices, _, _ = select_best(self.vocs, data, n=1)
                best_index = best_indices[0]
                selected = data.loc[[best_index]]
            except FeasibilityError:
                # If no feasible points exist, return an empty list.
                return []
        else:
            # For multi-objective and objective-less modes, return the available candidate set.
            selected = data

        output_names = self.vocs.output_names
        results: list[tuple[int | str, Mapping, Mapping]] = []
        for _, row in selected.iterrows():
            # Normalize IDs and split into parameter and outcome mappings.
            trial_id = _normalize_trial_id(row[ID_KEY] if ID_KEY in row else _)

            parameterization = {name: row[name] for name in self.vocs.variable_names if name in row}
            outcomes = {name: row[name] for name in output_names if name in row}
            results.append((trial_id, parameterization, outcomes))

        return results

    def checkpoint(self) -> None:
        """Dump serialized optimizer state to the configured checkpoint JSON file."""
        # Enforce explicit checkpoint path configuration before writing state.
        if not self._checkpoint_path:
            raise ValueError("Checkpoint path is not set. Please set a checkpoint path when initializing the optimizer.")

        # Persist generator and adapter bookkeeping to a single JSON artifact.
        payload = {
            "generator": {
                "class": self._generator.__class__.__name__,
                "module": self._generator.__class__.__module__,
                "state": self._generator.model_dump(),
                "data": (
                    self._generator.data.to_dict(orient="records")
                    if isinstance(self._generator.data, pd.DataFrame)
                    else self._generator.data
                ),
            },
            "next_id": self._next_id,
            "params_by_id": self._params_by_id,
        }

        # Write the payload to the configured checkpoint path.
        path = Path(self._checkpoint_path)
        with path.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, default=str)
