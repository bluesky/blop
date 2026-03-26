"""Live-updating matplotlib plots for optimization progress."""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Any, cast

from bluesky.callbacks.core import make_class_safe
from bluesky.callbacks.mpl_plotting import QtAwareCallback
from event_model import Event, EventDescriptor, RunStart, RunStop

from ..utils import Source

logger = logging.getLogger(__name__)

# Styling constants
_PARAM_COLOR = "#2196F3"
_OUTCOME_COLOR = "#4CAF50"
_OUTCOME_MARKER_SIZE = 12


def _to_list(value: Any) -> list:
    """Coerce a value into a list, handling scalars, numpy arrays, and iterables."""
    if hasattr(value, "tolist"):
        result = value.tolist()
        return result if isinstance(result, list) else [result]
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _is_numeric_dtype(data_key: dict) -> bool:
    """Check if a DataKey represents numeric data.

    Handles both scalar dtypes (``"number"``, ``"integer"``) and array
    dtypes where the element type must be inferred from ``dtype_numpy``.
    """
    dtype = data_key.get("dtype", "")
    if dtype in ("number", "integer"):
        return True
    if dtype == "array":
        dtype_numpy = data_key.get("dtype_numpy", "")
        # numpy dtype strings use 'f' for float, 'i' for signed int,
        # 'u' for unsigned int; 'U' is Unicode string, 'S' is byte string.
        if not dtype_numpy:
            return False
        kind = dtype_numpy.lstrip("<>|=")[:1]
        return kind in ("f", "i", "u")
    return False


def _import_matplotlib():
    """Import matplotlib with a clear error message if not installed."""
    try:
        import matplotlib
        import matplotlib.pyplot as plt

        return matplotlib, plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for OptimizationPlotter. Install it with: pip install blop[plot]"
        ) from None


def _subplot_grid(n: int) -> tuple[int, int]:
    """Compute a (nrows, ncols) grid for *n* subplots.

    Uses a single column for up to 3 subplots, then 2 columns.
    """
    if n <= 3:
        return n, 1
    ncols = 2
    nrows = math.ceil(n / ncols)
    return nrows, ncols


@make_class_safe(logger=logger)
class OptimizationPlotter(QtAwareCallback):
    """A Bluesky callback that displays live-updating plots during optimization.

    Creates two matplotlib figures:

    - **Parameters figure** -- one axis per parameter showing the distribution
      of sampled values.  Numeric parameters are displayed as histograms;
      categorical (string) parameters are displayed as bar charts.
    - **Outcomes figure** -- one axis per outcome showing a scatter/line plot
      of values over a global sample index.

    Data persists across consecutive optimization runs, allowing
    visualization of cumulative progress.

    The class inherits from :class:`~bluesky.callbacks.mpl_plotting.QtAwareCallback`
    so that callback methods are automatically dispatched to the main thread
    when a Qt matplotlib backend is in use.  It is also wrapped with
    :func:`~bluesky.callbacks.core.make_class_safe` so that any plotting
    errors are logged rather than interrupting data acquisition.

    Parameters
    ----------
    **kwargs
        Passed through to :class:`~bluesky.callbacks.mpl_plotting.QtAwareCallback`.

    Notes
    -----
    Requires ``matplotlib``.  Install with ``pip install blop[plot]``.

    If your figure blocks the main thread when you are trying to scan with
    this callback, call ``plt.ion()`` in your IPython session.

    If ``DataKey`` entries contain ``limits`` (with ``control.low`` /
    ``control.high``) or ``choices`` metadata, the plotter will use them
    for histogram ranges and bar-chart categories respectively.  These
    fields are currently not populated by the default blop plans but may
    be in the future.

    Usage::

        plotter = OptimizationPlotter()
        agent.subscribe(plotter)
        RE(agent.optimize(iterations=10))
    """

    def __init__(self, **kwargs: Any):
        super().__init__(use_teleporter=kwargs.pop("use_teleporter", None), **kwargs)

        # Validate that matplotlib is available at construction time.
        _import_matplotlib()

        # Descriptor state
        self._data_keys: dict = {}
        self._sorted_data_keys_by_source: dict[Source, list[str]] = {}

        # Accumulated history (persists across runs)
        self._parameter_history: dict[str, list] = defaultdict(list)
        self._outcome_history: dict[str, list] = defaultdict(list)
        self._outcome_sample_indices: list[int] = []
        self._sample_counter: int = 0
        self._run_boundaries: list[int] = []

        # Figure state (created lazily)
        self._param_fig: Any | None = None
        self._param_axes: dict[str, Any] = {}
        self._outcome_fig: Any | None = None
        self._outcome_axes: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Bluesky callback hooks
    # ------------------------------------------------------------------

    def start(self, doc: RunStart) -> None:
        """Record run metadata.  Figures are not created until the
        descriptor arrives (we need to know the keys first)."""

    def descriptor(self, doc: EventDescriptor) -> None:
        """Cache data keys grouped by source and lazily create figures."""
        data_keys = doc.get("data_keys", {})
        data_keys_by_source: dict[Source, list[str]] = defaultdict(list)
        for key, data_key in data_keys.items():
            data_keys_by_source[cast(Source, data_key.get("source", Source.OTHER))].append(key)

        self._sorted_data_keys_by_source = {src: sorted(keys) for src, keys in data_keys_by_source.items()}
        self._data_keys = data_keys

        self._ensure_figures()

    def event(self, doc: Event) -> Event:
        """Append new samples to history and refresh plots."""
        data = doc.get("data", {})
        if not data:
            return doc

        parameter_keys: list[str] = self._sorted_data_keys_by_source.get(Source.PARAMETER, [])
        outcome_keys: list[str] = self._sorted_data_keys_by_source.get(Source.OUTCOME, [])

        # Normalise to lists
        param_columns: dict[str, list] = {k: _to_list(data[k]) for k in parameter_keys if k in data}
        outcome_columns: dict[str, list] = {k: _to_list(data[k]) for k in outcome_keys if k in data}

        # Determine valid indices (filter NaN-padded batch entries)
        suggestion_ids = _to_list(data.get("suggestion_ids", []))
        n_total = max(
            (len(v) for v in [*param_columns.values(), *outcome_columns.values()]),
            default=1,
        )
        if suggestion_ids:
            valid_indices = [i for i, sid in enumerate(suggestion_ids) if sid != "" and str(sid).strip() != ""]
        else:
            valid_indices = list(range(n_total))

        # Accumulate parameter history
        for key, values in param_columns.items():
            for idx in valid_indices:
                if idx < len(values):
                    val = values[idx]
                    # Skip NaN for numeric values
                    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                        continue
                    self._parameter_history[key].append(val)

        # Accumulate outcome history
        for key, values in outcome_columns.items():
            for idx in valid_indices:
                if idx < len(values):
                    val = values[idx]
                    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                        continue
                    self._outcome_history[key].append(val)

        # Record sample indices for outcomes (one per valid sample)
        for idx in valid_indices:
            # Only record if there was at least one valid outcome value for this sample
            has_valid_outcome = False
            for key in outcome_keys:
                if key in outcome_columns:
                    vals = outcome_columns[key]
                    if idx < len(vals):
                        val = vals[idx]
                        if not (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
                            has_valid_outcome = True
                            break
            if has_valid_outcome:
                self._sample_counter += 1
                self._outcome_sample_indices.append(self._sample_counter)

        self._update_plots()
        return doc

    def stop(self, doc: RunStop) -> None:
        """Mark run boundary on outcome plots and tighten layout."""
        if self._sample_counter > 0:
            self._run_boundaries.append(self._sample_counter)

        self._update_plots()
        self._tighten_layout()

    # ------------------------------------------------------------------
    # Figure management
    # ------------------------------------------------------------------

    def _ensure_figures(self) -> None:
        """Create or recreate figures if needed."""
        _, plt = _import_matplotlib()

        parameter_keys = self._sorted_data_keys_by_source.get(Source.PARAMETER, [])
        outcome_keys = self._sorted_data_keys_by_source.get(Source.OUTCOME, [])

        # Parameters figure
        if parameter_keys:
            if not self._figures_alive(self._param_fig) or set(self._param_axes.keys()) != set(parameter_keys):
                self._create_param_figure(parameter_keys)

        # Outcomes figure
        if outcome_keys:
            if not self._figures_alive(self._outcome_fig) or set(self._outcome_axes.keys()) != set(outcome_keys):
                self._create_outcome_figure(outcome_keys)

    @staticmethod
    def _figures_alive(fig: Any) -> bool:
        """Check whether a matplotlib figure still exists."""
        if fig is None:
            return False
        _, plt = _import_matplotlib()
        return plt.fignum_exists(fig.number)

    def _create_param_figure(self, keys: list[str]) -> None:
        _, plt = _import_matplotlib()
        nrows, ncols = _subplot_grid(len(keys))
        fig, axes = plt.subplots(nrows, ncols, squeeze=False, figsize=(5 * ncols, 3 * nrows))
        fig.canvas.manager.set_window_title("Optimization Parameters")
        flat_axes = axes.flatten()

        self._param_axes = {}
        for i, key in enumerate(keys):
            self._param_axes[key] = flat_axes[i]
            flat_axes[i].set_title(key)

        # Hide unused axes
        for j in range(len(keys), len(flat_axes)):
            flat_axes[j].set_visible(False)

        fig.tight_layout()
        self._param_fig = fig

    def _create_outcome_figure(self, keys: list[str]) -> None:
        _, plt = _import_matplotlib()
        nrows, ncols = _subplot_grid(len(keys))
        fig, axes = plt.subplots(nrows, ncols, squeeze=False, figsize=(5 * ncols, 3 * nrows))
        fig.canvas.manager.set_window_title("Optimization Outcomes")
        flat_axes = axes.flatten()

        self._outcome_axes = {}
        for i, key in enumerate(keys):
            self._outcome_axes[key] = flat_axes[i]
            flat_axes[i].set_title(key)

        # Hide unused axes
        for j in range(len(keys), len(flat_axes)):
            flat_axes[j].set_visible(False)

        fig.tight_layout()
        self._outcome_fig = fig

    # ------------------------------------------------------------------
    # Plot updates
    # ------------------------------------------------------------------

    def _update_plots(self) -> None:
        """Redraw all plots with current history."""
        self._ensure_figures()
        self._update_parameter_plots()
        self._update_outcome_plots()
        self._redraw()

    def _update_parameter_plots(self) -> None:
        """Redraw parameter histograms / bar charts."""
        for key, ax in self._param_axes.items():
            values = self._parameter_history.get(key, [])
            ax.clear()
            ax.set_title(key)

            if not values:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, color="gray")
                continue

            data_key = self._data_keys.get(key, {})

            if _is_numeric_dtype(data_key):
                self._draw_numeric_histogram(ax, key, values, data_key)
            else:
                self._draw_categorical_bar(ax, key, values, data_key)

    def _draw_numeric_histogram(self, ax: Any, key: str, values: list, data_key: dict) -> None:
        """Draw a histogram for a numeric parameter."""
        # Use limits from DataKey if available for bin range
        hist_range = None
        limits = data_key.get("limits", {})
        control = limits.get("control", {}) if isinstance(limits, dict) else {}
        low = control.get("low") if isinstance(control, dict) else None
        high = control.get("high") if isinstance(control, dict) else None
        if low is not None and high is not None:
            hist_range = (float(low), float(high))

        numeric_values = [v for v in values if isinstance(v, (int, float)) and not (math.isnan(v) or math.isinf(v))]
        if not numeric_values:
            ax.text(0.5, 0.5, "No numeric data", ha="center", va="center", transform=ax.transAxes, color="gray")
            return

        ax.hist(numeric_values, bins="auto", range=hist_range, color=_PARAM_COLOR, edgecolor="white", alpha=0.85)
        ax.set_xlabel(key)
        ax.set_ylabel("Count")

    def _draw_categorical_bar(self, ax: Any, key: str, values: list, data_key: dict) -> None:
        """Draw a bar chart for a categorical parameter."""
        # Use choices from DataKey if available, otherwise derive from data
        choices = data_key.get("choices")
        if choices is not None:
            categories = list(choices)
        else:
            # Derive from observed data, sorted for stability
            categories = sorted({str(v) for v in values})

        counts = dict.fromkeys(categories, 0)
        for v in values:
            s = str(v)
            if s in counts:
                counts[s] += 1
            else:
                # Value not in predefined choices — add it
                counts[s] = 1
                categories.append(s)

        ax.bar(categories, [counts[c] for c in categories], color=_PARAM_COLOR, edgecolor="white", alpha=0.85)
        ax.set_xlabel(key)
        ax.set_ylabel("Count")
        if len(categories) > 4:
            ax.tick_params(axis="x", rotation=45)

    def _update_outcome_plots(self) -> None:
        """Redraw outcome scatter/line plots."""
        for key, ax in self._outcome_axes.items():
            values = self._outcome_history.get(key, [])
            ax.clear()
            ax.set_title(key)

            if not values:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, color="gray")
                continue

            # Plot individual samples
            n = min(len(self._outcome_sample_indices), len(values))
            indices = self._outcome_sample_indices[:n]
            ax.scatter(indices, values[:n], s=_OUTCOME_MARKER_SIZE, color=_OUTCOME_COLOR, alpha=0.7, zorder=2)

            # Draw run boundaries as vertical lines
            for boundary in self._run_boundaries:
                if boundary < max(indices):
                    ax.axvline(x=boundary, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

            ax.set_xlabel("Sample")
            ax.set_ylabel(key)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _redraw(self) -> None:
        """Non-blocking redraw of all live figures.

        Only calls ``draw_idle()`` — matching Bluesky's ``LivePlot``
        convention.  The actual rendering is handled by the GUI event
        loop (e.g. ``plt.ion()`` in IPython or a Qt event loop).
        """
        for fig in (self._param_fig, self._outcome_fig):
            if fig is not None and self._figures_alive(fig):
                fig.canvas.draw_idle()

    def _tighten_layout(self) -> None:
        """Apply tight_layout to figures that are still open."""
        for fig in (self._param_fig, self._outcome_fig):
            if fig is not None and self._figures_alive(fig):
                fig.tight_layout()
