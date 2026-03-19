from collections import defaultdict
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from bluesky.callbacks import CallbackBase
from event_model import Event, EventDescriptor, RunRouter, RunStart, RunStop

from ..plans import OPTIMIZE_RUN_KEY


class _OptimizationCallback(CallbackBase):
    """
    A Bluesky callback for displaying optimization progress and live plots.

    This callback provides structured stdout output and live visualizations during
    optimization runs. It listens for events from the ``optimize`` plan and displays:
    - Progress information (iteration count, best value, last value)
    - Live histograms of parameter distributions
    - Line plots of objective values over iterations

    Parameters
    ----------
    stdout : bool, optional
        Whether to show stdout progress output. Default is True.
    live_plots : bool, optional
        Whether to show live plots. Default is True.
    figsize : tuple[float, float], optional
        Figure size for the plots. Default is (12, 6).

    Notes
    -----
    This callback is automatically used by the Agent when running ``optimize``
    if no custom callback is specified. Multiple consecutive optimization runs
    will accumulate data in the plots.

    """

    def __init__(
        self,
        parameter_names: Sequence[str],
        objective_names: Sequence[str],
        stdout: bool = True,
        live_plots: bool = True,
        figsize: tuple[float, float] = (12, 6),
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._parameters = parameter_names
        self._objectives = objective_names
        self._stdout = stdout
        self._live_plots = live_plots
        self._figsize = figsize

        self._optimize_run_uid: str | None = None
        self._optimize_descriptor_uids: set[str] = set()
        self._iteration: int = 0

        self._parameter_data: dict[str, list[Any]] = defaultdict(list)
        self._objective_data: dict[str, list[float]] = defaultdict(list)
        self._suggestion_ids_history: list[str] = []

        self._initialized = False

        if self._live_plots:
            self._setup_plots()

    def _setup_plots(self) -> None:

        plt.ion()
        self._fig, self._axes = plt.subplots(1, 2, figsize=self._figsize)
        self._fig.canvas.draw()
        plt.show(block=False)

    def start(self, doc: RunStart) -> None:
        self._optimize_run_uid = doc["uid"]
        self._iteration = 0

        if self._stdout:
            iterations = doc.get("iterations", "?")
            print(f"\n{'=' * 60}")
            print(f"Starting optimization for {iterations} iterations")
            print(f"{'=' * 60}\n")

    def descriptor(self, doc: EventDescriptor) -> None:
        # TODO: Something specific with data types?
        ...

    def event(self, doc: Event) -> Event:
        if self._optimize_run_uid is None:
            return doc

        descriptor_uid = doc.get("descriptor")
        if descriptor_uid not in self._optimize_descriptor_uids:
            return doc

        data = doc.get("data", {})
        if not data:
            return doc

        self._iteration += 1

        suggestion_ids = data.get("suggestion_ids", [])
        if isinstance(suggestion_ids, str):
            suggestion_ids = [suggestion_ids]

        for sid in suggestion_ids:
            if sid and sid not in self._suggestion_ids_history:
                self._suggestion_ids_history.append(sid)

        for parameter_name in self._parameters or []:
            values = data.get(parameter_name)
            if values is not None:
                if isinstance(values, (int, float, str)):
                    values = [values]
                for v in values:
                    if isinstance(v, (int, float)) and not np.isnan(v):
                        self._parameter_data[parameter_name].append(v)

        for obj_name in self._objectives or []:
            values = data.get(obj_name)
            if values is not None:
                if isinstance(values, (int, float)):
                    values = [values]
                for v in values:
                    if isinstance(v, (int, float)) and not np.isnan(v):
                        self._objective_data[obj_name].append(v)

        if self._stdout:
            self._print_progress(suggestion_ids)

        if self._live_plots:
            self._update_plots()

        return doc

    def _print_progress(self, suggestion_ids: list[str]) -> None:
        n_evaluated = len(self._suggestion_ids_history)

        best_values: list[str] = []
        last_values: list[str] = []

        for obj_name, values in self._objective_data.items():
            if values:
                best_val = np.nanmin(values) if self._is_minimizing(obj_name) else np.nanmax(values)
                last_val = values[-1]
                best_values.append(f"{obj_name}: {best_val:.4f}")
                last_values.append(f"{obj_name}: {last_val:.4f}")

        print(f"Iteration {self._iteration} ({n_evaluated} evaluated)")
        if best_values:
            print(f"  best: {', '.join(best_values)}")
        if last_values:
            print(f"  last: {', '.join(last_values)}")
        print()

    def _is_minimizing(self, objective_name: str) -> bool:
        return True

    def stop(self, doc: RunStop) -> None:
        if self._optimize_run_uid is None:
            return

        if doc.get("run_start") == self._optimize_run_uid:
            if self._stdout:
                print(f"\n{'=' * 60}")
                print("Optimization complete")
                print(f"{'=' * 60}\n")

            if self._live_plots:
                self._finalize_plots()

            self._optimize_run_uid = None
            self._optimize_descriptor_uids.clear()

    def _update_plots(self) -> None:

        if not hasattr(self, "_fig"):
            return

        axes = self._axes
        axes[0].clear()
        axes[1].clear()

        parameters_to_plot = list(self._parameter_data.keys())[:4]
        for parameter_name in parameters_to_plot:
            values = self._parameter_data.get(parameter_name, [])
            if values and len(values) > 1:
                axes[0].hist(values, bins=20, alpha=0.7, label=parameter_name)

        if parameters_to_plot:
            axes[0].set_xlabel("Value")
            axes[0].set_ylabel("Count")
            axes[0].set_title("Parameter Distributions")
            axes[0].legend()
        else:
            axes[0].set_title("No parameter data yet")

        for obj_name, values in self._objective_data.items():
            if values and len(values) > 1:
                axes[1].plot(values, marker="o", label=obj_name, alpha=0.7)

        if self._objective_data:
            axes[1].set_xlabel("Iteration")
            axes[1].set_ylabel("Value")
            axes[1].set_title("Objectives over Iterations")
            axes[1].legend()
        else:
            axes[1].set_title("No objective data yet")

        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

    def _finalize_plots(self) -> None:

        if hasattr(self, "_fig"):
            plt.ioff()
            plt.show(block=True)

    def reset(self) -> None:
        """
        Reset the callback's accumulated data.

        This is useful when starting a new optimization experiment but keeping
        the same callback instance.
        """
        self._iteration = 0
        self._parameter_data.clear()
        self._objective_data.clear()
        self._suggestion_ids_history.clear()
        self._initialized = False

        if self._live_plots and hasattr(self, "_fig"):
            plt.close(self._fig)
            self._setup_plots()


class BestEffortOptimizationCallback:
    """Best effort callback for displaying optimization information."""

    def __init__(self, stdout: bool = True, live_plots: bool = True, figsize: tuple[int, int] = (12, 6)) -> None:
        self._run_router = RunRouter([self._factory])
        self._stdout = stdout
        self._live_plots = live_plots
        self._figsize = figsize

    def _factory(self, name, doc):
        if name == "start" and doc["run_key"] == OPTIMIZE_RUN_KEY:
            callback = _OptimizationCallback(stdout=self._stdout, live_plots=self._live_plots, figsize=self._figsize)
            return [callback], []
        return [], []

    def __call__(self, name, doc):
        self._run_router(name, doc)
