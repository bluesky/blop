"""Logging callback for Bluesky optimization runs."""

import math
from collections import defaultdict
from typing import Any, cast

from bluesky.callbacks import CallbackBase
from event_model import Event, EventDescriptor, RunStart, RunStop
from rich.box import Box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..plan_stubs import _ACQUISITION_UID_KEY, _ITERATION_KEY, _SUGGESTION_IDS_KEY
from ..utils import Source
from .utils import RunningStats

# Styling constants
_PARAM_STYLE = "cyan"
_OUTCOME_STYLE = "green"
_HEADER_STYLE = "bold"
_DIM_STYLE = "dim"
_ERROR_STYLE = "bold red"
_ITERATION_RULE_STYLE = "blue"
_ITER_COLORS = ["#E1C052", "#8FA88B", "#C78B94", "#7A93A6", "#9B8BA8"]
_BOX_VERT = Box("┃ ┃┃\n┃┃┃┃\n┃┃┃┃\n┃┃┃┃\n┃┃┃┃\n┃┃┃┃\n┃┃┃┃\n┡─╇┩")


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


def _is_numeric(value: Any) -> bool:
    """Check if a value is numeric (int or float)."""
    return isinstance(value, (int, float))


class OptimizationLogger(CallbackBase):
    """A Bluesky callback for displaying optimization progress to the console.

    This callback provides structured, styled console output during
    optimization runs using the ``rich`` library. It listens for documents
    from the ``optimize`` plan and displays:

    - A header panel with optimizer configuration at run start
    - A formatted table of parameter and outcome values for each step
    - - a box coloring indicating sectioning by iteration
    - A compact inline summary of outcome statistics after every 5 steps
    - A full summary statistics table at run completion

    Notes
    -----
    Multiple consecutive optimization runs will accumulate iteration counts
    and statistics.
    """

    def __init__(self, console: Console | None = None, **kwargs: Any):
        super().__init__(**kwargs)

        self._console = console or Console()
        self._data_keys: dict = {}
        self._sorted_data_keys_by_source: dict[Source, list[str]] = {}
        self._total_iterations: int | None = 0
        self._base_iteration: int = 0
        self._current_iteration: int = 0
        self._current_step: int = 0
        self._stats: dict[str, RunningStats] = {}

    def start(self, doc: RunStart) -> None:
        """
        Process the start document from the Bluesky run.

        Logs the optimization setup observed from metadata.
        """
        iterations = doc.get("iterations", None)
        n_points = doc.get("n_points", 1)
        optimizer = doc.get("optimizer", "Unknown")
        actuators = doc.get("actuators", [])
        sensors = doc.get("sensors", [])
        run_uid = doc.get("uid", "")

        self._total_iterations = None if iterations is None else self._base_iteration + iterations

        # Build the header content
        lines = Text()
        lines.append("Optimizer  ", style=_DIM_STYLE)
        lines.append(f"{optimizer}\n", style=_HEADER_STYLE)
        lines.append("Actuators  ", style=_DIM_STYLE)
        lines.append(f"{', '.join(actuators) if actuators else 'N/A'}\n")
        lines.append("Sensors    ", style=_DIM_STYLE)
        lines.append(f"{', '.join(sensors) if sensors else 'N/A'}\n")
        lines.append("Iterations ", style=_DIM_STYLE)

        if iterations is None:
            lines.append("Until stopping criterion")
            if self._base_iteration > 0:
                lines.append(f" ({self._base_iteration + 1} completed)")
        elif self._base_iteration > 0:
            lines.append(f"{iterations} more ({self._base_iteration} completed, ")
            lines.append(f"{self._total_iterations} total)")
        else:
            lines.append(f"{iterations}")

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
        self._console.print(panel)

    def descriptor(self, doc: EventDescriptor) -> None:
        """Cache data keys and group by their source."""
        data_keys = doc.get("data_keys", {})
        data_keys_by_source: dict[Source, list[str]] = defaultdict(list)
        for key, data_key in data_keys.items():
            data_keys_by_source[cast(Source, data_key.get("source", Source.OTHER))].append(key)

        self._sorted_data_keys_by_source = {key: sorted(keys) for key, keys in data_keys_by_source.items()}
        self._parameter_keys: list[str] = self._sorted_data_keys_by_source.get(Source.PARAMETER, [])
        self._outcome_keys: list[str] = self._sorted_data_keys_by_source.get(Source.OUTCOME, [])

        # build header item for iteratively generated table
        table = Table(expand=True)
        table.add_column("Suggestion ID", style=_DIM_STYLE, no_wrap=True, ratio=1)
        for param in self._parameter_keys:
            table.add_column(param, style=_PARAM_STYLE, no_wrap=True, ratio=1)

        for outcome in self._outcome_keys:
            table.add_column(outcome, style=_OUTCOME_STYLE, no_wrap=True, ratio=1)

        self._data_keys = data_keys
        self._console.print(table)

    def _update_stats(self, columns: dict[str, Any]) -> None:
        """Update running statistics for each key with the valid values from this event."""
        for key, value in columns.items():
            if _is_numeric(value):
                if key not in self._stats:
                    self._stats[key] = RunningStats()
                self._stats[key].update(float(value))

    def event(self, doc: Event) -> Event:
        """
        Process an event document from a Bluesky run.

        Logs what occurred in this event along with running stats.
        """
        data = doc.get("data", {})
        if not data:
            return doc

        self._current_step += 1
        parameter_keys = self._parameter_keys
        outcome_keys = self._outcome_keys

        # Extract values, normalizing to regular parametrization
        param_columns: dict[str, Any] = {k: data[k] for k in parameter_keys if k in data}
        outcome_columns: dict[str, Any] = {k: data[k] for k in outcome_keys if k in data}

        # Extract suggestion IDs and acquisition identifier
        suggestion_ids = data.get(_SUGGESTION_IDS_KEY, [])
        acquire_uid = data.get(_ACQUISITION_UID_KEY, "")
        run_iteration = data.get(_ITERATION_KEY, 0)
        self._current_iteration = self._base_iteration + run_iteration + 1

        # Scalar string comes through as-is; ensure it's a plain string
        if isinstance(acquire_uid, list):
            acquire_uid = acquire_uid[0] if acquire_uid else ""

        # Update running statistics
        self._update_stats(param_columns)
        self._update_stats(outcome_columns)

        # Build the results table
        table = Table(
            show_header=False,
            header_style=_HEADER_STYLE,
            border_style=_ITER_COLORS[run_iteration % 5],
            box=_BOX_VERT,
            expand=True,
        )
        row: list[str] = []

        # Iteration and suggestion ID columns (always shown)
        table.add_column("Suggestion ID", style=_DIM_STYLE, justify="left", no_wrap=True, ratio=1)
        row.append(str(suggestion_ids))

        for key in parameter_keys:
            if key in param_columns:
                table.add_column(key, style=_PARAM_STYLE, justify="right", no_wrap=True, ratio=1)
                val = param_columns[key]
                row.append(_format_value(val))
        for key in outcome_keys:
            if key in outcome_columns:
                table.add_column(key, style=_OUTCOME_STYLE, justify="right", no_wrap=True, ratio=1)
                val = outcome_columns[key]
                row.append(_format_value(val))

        table.add_row(*row)
        self._console.print(table)
        if self._current_step % 5 == 0:
            # Iteration header rule
            iter_label = f"Iteration {self._current_iteration + 1}"
            if self._total_iterations is not None:
                iter_label += f" / {self._total_iterations}"
            self._console.rule(iter_label, style=_ITERATION_RULE_STYLE)

            trackable_outcomes = [k for k in outcome_keys if k in self._stats and self._stats[k].count > 0]
            if trackable_outcomes:
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
                summary.append(f"\n  ({self._current_step} pts sampled)", style=_DIM_STYLE)
                self._console.print(summary)

            # closing rule
            self._console.rule(style=_ITERATION_RULE_STYLE)
        return doc

    def stop(self, doc: RunStop) -> None:
        """
        Handle the stop document of a Bluesky run.

        Prints summary statistics of what has been observed so far.
        """
        exit_status = doc.get("exit_status", "success")
        reason = doc.get("reason", "")

        parameter_keys: list[str] = self._sorted_data_keys_by_source.get(Source.PARAMETER, [])
        outcome_keys: list[str] = self._sorted_data_keys_by_source.get(Source.OUTCOME, [])

        # housekeeping iteration tracking
        self._base_iteration = self._current_iteration
        self._total_iterations = self._base_iteration if self._total_iterations else None

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
