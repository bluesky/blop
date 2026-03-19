from collections import defaultdict
from typing import Any, cast

from bluesky.callbacks import CallbackBase
from event_model import Event, EventDescriptor, RunRouter, RunStart, RunStop

from ..utils import Source
from ..plans import OPTIMIZE_RUN_KEY


class OptimizationLogger(CallbackBase):
    """
    A Bluesky callback for displaying optimization progress and live plots.

    This callback provides structured stdout output during
    optimization runs. It listens for events from the ``optimize`` plan.

    Notes
    -----
    Multiple consecutive optimization runs will accumulate data in the plots.
    """

    def __init__(
        self,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self._data_keys = {}
        self._sorted_data_keys_by_source: dict[Source, list[str]] = {}
        self._total_iterations = 0
        self._current_iteration = 0

    def start(self, doc: RunStart) -> None:
        iterations = doc.get("iterations", None)
        print(f"\n{'=' * 60}")
        if iterations:
            self._total_iterations = self._current_iteration + iterations
            if self._current_iteration > 0:
                print(f"Starting optimization for {iterations} more iterations")
                print(f"Last iteration complete: {self._current_iteration}")
                print(f"Total iterations to complete: {self._total_iterations}")
            else:
                print(f"Starting optimization for {iterations} iterations")
        else:
            print("Starting optimization for ? iterations")
        print(f"{'=' * 60}\n")

    def descriptor(self, doc: EventDescriptor) -> None:
        """Cache data keys and group by their source"""
        data_keys = doc.get("data_keys", {})
        data_keys_by_source: dict[Source, list[str]] = defaultdict(list)
        for key, data_key in data_keys.items():
            data_keys_by_source[cast(Source, data_key.get("source", Source.OTHER))].append(key)

        sorted_data_keys_by_source = {key: sorted(data_keys) for key, data_keys in data_keys_by_source.items()}

        print(f"\n{'=' * 60}")
        print("Parameters: ")
        for p in sorted_data_keys_by_source[Source.PARAMETER]:
            print(f"- {p}")
        print("Outcomes: ")
        for o in sorted_data_keys_by_source[Source.OUTCOME]:
            print(f"- {o}")
        print(f"{'=' * 60}")

        self._data_keys = data_keys
        self._sorted_data_keys_by_source = sorted_data_keys_by_source

    def event(self, doc: Event) -> Event:
        data = doc.get("data", {})
        if not data:
            return doc

        self._current_iteration += 1
        print(f"\nIteration {self._current_iteration} / {self._total_iterations}:")
        parameter_keys = self._sorted_data_keys_by_source[Source.PARAMETER]
        for param in parameter_keys:
            param_data = data.get(param, None)
            if param_data is None:
                continue
            print(f"{param}: {param_data}")
        outcome_keys = self._sorted_data_keys_by_source[Source.OUTCOME]
        for outcome in outcome_keys:
            outcome_data = data.get(outcome, None)
            if outcome_data is None:
                continue
            print(f"{outcome}: {outcome_data}")
        return doc

    def stop(self, doc: RunStop) -> None:
        print(f"\n{'=' * 60}")
        print("Optimization complete")
        print(f"{'=' * 60}\n")


class BestEffortOptimizationCallback:
    """Best effort callback for displaying optimization information."""

    def __init__(
        self,
        stdout: bool = True,
        live_plots: bool = True,
        figsize: tuple[int, int] = (12, 6),
    ) -> None:
        self._run_router = RunRouter([self._factory])
        self._callbacks = self._setup_callbacks(stdout, live_plots, figsize)

    def _setup_callbacks(
        self, stdout: bool = True, live_plots: bool = True, figsize: tuple[int, int] = (12, 6)
    ) -> list[CallbackBase]:
        callbacks = []
        if stdout:
            callbacks.append(OptimizationLogger())
        return callbacks

    def _factory(self, name, doc):
        if name == "start" and doc["run_key"] == OPTIMIZE_RUN_KEY:
            return self._callbacks, []
        return [], []

    def __call__(self, name, doc):
        self._run_router(name, doc)
