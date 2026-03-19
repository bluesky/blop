import math
from collections import defaultdict
from typing import Any, cast

from bluesky.callbacks import CallbackBase
from event_model import Event, EventDescriptor, RunRouter, RunStart, RunStop
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..plans import OPTIMIZE_RUN_KEY
from ..utils import Source

# Styling constants
_PARAM_STYLE = "cyan"
_OUTCOME_STYLE = "green"
_HEADER_STYLE = "bold"
_DIM_STYLE = "dim"
_ERROR_STYLE = "bold red"
_ITERATION_RULE_STYLE = "blue"


class _RunningStats:
    """Accumulates running statistics using Welford's online algorithm.

    Tracks count, min, max, mean, and standard deviation without
    storing individual observations. Numerically stable for variance
    computation.
    """

    __slots__ = ("count", "min", "max", "_mean", "_m2")

    def __init__(self) -> None:
        self.count: int = 0
        self.min: float = math.inf
        self.max: float = -math.inf
        self._mean: float = 0.0
        self._m2: float = 0.0

    def update(self, value: float) -> None:
        """Incorporate a new observation."""
        if math.isnan(value) or math.isinf(value):
            return
        self.count += 1
        self.min = min(self.min, value)
        self.max = max(self.max, value)
        delta = value - self._mean
        self._mean += delta / self.count
        delta2 = value - self._mean
        self._m2 += delta * delta2

    @property
    def mean(self) -> float:
        return self._mean if self.count > 0 else math.nan

    @property
    def std(self) -> float:
        if self.count < 2:
            return math.nan
        return math.sqrt(self._m2 / (self.count - 1))


def _format_value(value: Any) -> str:
    """Format a numeric or other value for display.

    Uses 6 significant figures for floats, passes through everything else.
    """
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        return f"{value:.6g}"
    return str(value)


def _format_stat(value: float) -> str:
    """Format a statistic value, returning '--' for NaN."""
    if math.isnan(value) or math.isinf(value):
        return "--"
    return f"{value:.6g}"


def _to_list(value: Any) -> list:
    """Coerce a value into a list, handling scalars, numpy arrays, and iterables."""
    if hasattr(value, "tolist"):
        result = value.tolist()
        return result if isinstance(result, list) else [result]
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _is_numeric(value: Any) -> bool:
    """Check if a value is numeric (int or float)."""
    return isinstance(value, (int, float))


class OptimizationLogger(CallbackBase):
    """A Bluesky callback for displaying optimization progress to the console.

    This callback provides structured, styled console output during
    optimization runs using the ``rich`` library. It listens for documents
    from the ``optimize`` plan and displays:

    - A header panel with optimizer configuration at run start
    - A formatted table of parameter and outcome values for each iteration
    - A compact inline summary of outcome statistics after each iteration
    - A full summary statistics table at run completion

    Notes
    -----
    Multiple consecutive optimization runs will accumulate iteration counts
    and statistics.

    When ``n_points > 1``, each iteration is displayed as a multi-row table
    showing the batch of points suggested together by the optimizer, with
    NaN-padded entries (from incomplete batches) filtered out.
    """

    def __init__(self, console: Console | None = None, **kwargs: Any):
        super().__init__(**kwargs)

        self._console = console or Console()
        self._data_keys: dict = {}
        self._sorted_data_keys_by_source: dict[Source, list[str]] = {}
        self._total_iterations = 0
        self._current_iteration = 0
        self._stats: dict[str, _RunningStats] = {}

    def start(self, doc: RunStart) -> None:
        iterations = doc.get("iterations", None)
        n_points = doc.get("n_points", 1)
        optimizer = doc.get("optimizer", "Unknown")
        actuators = doc.get("actuators", [])
        sensors = doc.get("sensors", [])
        run_uid = doc.get("uid", "")

        if iterations:
            self._total_iterations = self._current_iteration + iterations

        # Build the header content
        lines = Text()
        lines.append("Optimizer  ", style=_DIM_STYLE)
        lines.append(f"{optimizer}\n", style=_HEADER_STYLE)
        lines.append("Actuators  ", style=_DIM_STYLE)
        lines.append(f"{', '.join(actuators) if actuators else 'N/A'}\n")
        lines.append("Sensors    ", style=_DIM_STYLE)
        lines.append(f"{', '.join(sensors) if sensors else 'N/A'}\n")
        lines.append("Iterations ", style=_DIM_STYLE)

        if self._current_iteration > 0:
            lines.append(f"{iterations} more ({self._current_iteration} completed, ")
            lines.append(f"{self._total_iterations} total)")
        else:
            lines.append(f"{iterations}" if iterations else "?")

        if n_points and n_points > 1:
            lines.append("  ")
            lines.append("Points/iter ", style=_DIM_STYLE)
            lines.append(f"{n_points}")

        if run_uid:
            lines.append("\n")
            lines.append("Run UID    ", style=_DIM_STYLE)
            lines.append(run_uid)

        panel = Panel(
            lines,
            title="[bold]Optimization[/bold]",
            border_style="blue",
            padding=(0, 1),
        )
        self._console.print()
        self._console.print(panel)

    def descriptor(self, doc: EventDescriptor) -> None:
        """Cache data keys and group by their source."""
        data_keys = doc.get("data_keys", {})
        data_keys_by_source: dict[Source, list[str]] = defaultdict(list)
        for key, data_key in data_keys.items():
            data_keys_by_source[cast(Source, data_key.get("source", Source.OTHER))].append(key)

        self._sorted_data_keys_by_source = {key: sorted(keys) for key, keys in data_keys_by_source.items()}
        self._data_keys = data_keys

    def _update_stats(self, columns: dict[str, list], valid_indices: list[int]) -> None:
        """Update running statistics for each key with the valid values from this event."""
        for key, values in columns.items():
            for idx in valid_indices:
                if idx < len(values) and _is_numeric(values[idx]):
                    if key not in self._stats:
                        self._stats[key] = _RunningStats()
                    self._stats[key].update(float(values[idx]))

    def event(self, doc: Event) -> Event:
        data = doc.get("data", {})
        if not data:
            return doc

        self._current_iteration += 1

        parameter_keys: list[str] = self._sorted_data_keys_by_source.get(Source.PARAMETER, [])
        outcome_keys: list[str] = self._sorted_data_keys_by_source.get(Source.OUTCOME, [])

        # Extract values, normalizing to lists for uniform handling
        param_columns: dict[str, list] = {k: _to_list(data[k]) for k in parameter_keys if k in data}
        outcome_columns: dict[str, list] = {k: _to_list(data[k]) for k in outcome_keys if k in data}

        # Extract suggestion IDs and acquisition UID
        suggestion_ids = _to_list(data.get("suggestion_ids", []))
        acquire_uid = data.get("bluesky_uid", "")
        # Scalar string comes through as-is; ensure it's a plain string
        if isinstance(acquire_uid, list):
            acquire_uid = acquire_uid[0] if acquire_uid else ""

        n_total = max(
            (len(v) for v in [*param_columns.values(), *outcome_columns.values()]),
            default=1,
        )
        # Filter out NaN-padded entries: suggestion_ids padded with "" indicate padding
        if suggestion_ids:
            valid_indices = [i for i, sid in enumerate(suggestion_ids) if sid != "" and str(sid).strip() != ""]
        else:
            valid_indices = list(range(n_total))
        n_valid = len(valid_indices) if valid_indices else n_total

        # Update running statistics
        self._update_stats(param_columns, valid_indices)
        self._update_stats(outcome_columns, valid_indices)

        # Iteration header rule
        iter_label = f"Iteration {self._current_iteration} / {self._total_iterations}"
        if n_valid > 1:
            iter_label += f"  ({n_valid} points)"
        self._console.rule(iter_label, style=_ITERATION_RULE_STYLE)

        # Show acquisition UID for this iteration
        if acquire_uid:
            uid_line = Text()
            uid_line.append("  Acquire UID  ", style=_DIM_STYLE)
            uid_line.append(str(acquire_uid))
            self._console.print(uid_line)

        # Build the results table
        table = Table(
            show_header=True,
            header_style=_HEADER_STYLE,
            border_style=_DIM_STYLE,
            pad_edge=True,
            padding=(0, 1),
        )

        # Iteration and suggestion ID columns (always shown)
        table.add_column("Event", style=_DIM_STYLE, justify="right", no_wrap=True)
        table.add_column("Suggestion ID", style=_DIM_STYLE, justify="right", no_wrap=True)

        for key in parameter_keys:
            if key in param_columns:
                table.add_column(key, style=_PARAM_STYLE, justify="right", no_wrap=True)
        for key in outcome_keys:
            if key in outcome_columns:
                table.add_column(key, style=_OUTCOME_STYLE, justify="right", no_wrap=True)

        # Populate rows
        for row_idx, data_idx in enumerate(valid_indices):
            row: list[str] = []
            row.append(str(row_idx))
            # Suggestion ID for this point
            sid = suggestion_ids[data_idx] if data_idx < len(suggestion_ids) else ""
            row.append(str(sid))
            for key in parameter_keys:
                if key in param_columns:
                    vals = param_columns[key]
                    row.append(_format_value(vals[data_idx] if data_idx < len(vals) else ""))
            for key in outcome_keys:
                if key in outcome_columns:
                    vals = outcome_columns[key]
                    row.append(_format_value(vals[data_idx] if data_idx < len(vals) else ""))
            table.add_row(*row)

        self._console.print(table)

        # Inline outcome summary
        outcome_point_count = next(
            (self._stats[k].count for k in outcome_keys if k in self._stats and self._stats[k].count > 0),
            0,
        )
        trackable_outcomes = [k for k in outcome_keys if k in self._stats and self._stats[k].count > 0]
        if trackable_outcomes and outcome_point_count > 0:
            summary = Text()
            summary.append("  ")
            for i, key in enumerate(trackable_outcomes):
                s = self._stats[key]
                if i > 0:
                    summary.append("\n  ", style=_DIM_STYLE)
                summary.append(key, style=_OUTCOME_STYLE)
                summary.append("  min: ", style=_DIM_STYLE)
                summary.append(_format_stat(s.min))
                summary.append("  max: ", style=_DIM_STYLE)
                summary.append(_format_stat(s.max))
                summary.append("  mean: ", style=_DIM_STYLE)
                summary.append(_format_stat(s.mean))
            summary.append(f"\n  ({outcome_point_count} pts sampled)", style=_DIM_STYLE)
            self._console.print(summary)

        return doc

    def stop(self, doc: RunStop) -> None:
        exit_status = doc.get("exit_status", "success")
        reason = doc.get("reason", "")

        parameter_keys: list[str] = self._sorted_data_keys_by_source.get(Source.PARAMETER, [])
        outcome_keys: list[str] = self._sorted_data_keys_by_source.get(Source.OUTCOME, [])

        # Build and print the summary statistics table
        trackable_keys = [k for k in [*parameter_keys, *outcome_keys] if k in self._stats and self._stats[k].count > 0]
        if trackable_keys:
            self._console.print()
            summary_table = Table(
                title="Summary Statistics",
                title_style="bold",
                show_header=True,
                header_style=_HEADER_STYLE,
                border_style=_DIM_STYLE,
                pad_edge=True,
                padding=(0, 1),
            )
            summary_table.add_column("Name", no_wrap=True)
            summary_table.add_column("Type", style=_DIM_STYLE, no_wrap=True)
            summary_table.add_column("Min", justify="right", no_wrap=True)
            summary_table.add_column("Max", justify="right", no_wrap=True)
            summary_table.add_column("Mean", justify="right", no_wrap=True)
            summary_table.add_column("Std", justify="right", no_wrap=True)
            summary_table.add_column("Count", justify="right", style=_DIM_STYLE, no_wrap=True)

            for key in trackable_keys:
                s = self._stats[key]
                is_param = key in parameter_keys
                name_style = _PARAM_STYLE if is_param else _OUTCOME_STYLE
                type_label = "param" if is_param else "outcome"

                summary_table.add_row(
                    Text(key, style=name_style),
                    type_label,
                    _format_stat(s.min),
                    _format_stat(s.max),
                    _format_stat(s.mean),
                    _format_stat(s.std),
                    str(s.count),
                )

            self._console.print(summary_table)

        if exit_status == "success":
            self._console.rule("[bold]Optimization Complete[/bold]", style="green")
        elif exit_status == "abort":
            label = "[bold]Optimization Aborted[/bold]"
            if reason:
                label += f"  ({reason})"
            self._console.rule(label, style=_ERROR_STYLE)
        else:
            label = f"[bold]Optimization Stopped[/bold]  ({exit_status})"
            if reason:
                label += f"  {reason}"
            self._console.rule(label, style="yellow")

        self._console.print()


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
