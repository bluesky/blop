"""Unit tests for the OptimizationPlotter callback."""

from unittest.mock import patch

# ---------------------------------------------------------------------------
# Use the non-interactive Agg backend so no windows are created during tests
# ---------------------------------------------------------------------------
import matplotlib
import pytest
from event_model import Event, EventDescriptor, RunStart, RunStop

from blop.callbacks.plotter import OptimizationPlotter, _is_numeric_dtype, _subplot_grid, _to_list
from blop.utils import Source

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers – mirrors the patterns in test_logger.py
# ---------------------------------------------------------------------------


def _make_start(**overrides) -> RunStart:
    doc = {"uid": "start-001", "time": 0.0}
    doc.update(overrides)
    return RunStart(**doc)


def _make_descriptor(data_keys: dict | None = None, **overrides) -> EventDescriptor:
    if data_keys is None:
        data_keys = {
            "x": {"dtype": "number", "shape": [], "source": Source.PARAMETER.value},
            "y": {"dtype": "number", "shape": [], "source": Source.OUTCOME.value},
        }
    doc = {
        "uid": "desc-001",
        "time": 0.0,
        "run_start": "start-001",
        "data_keys": data_keys,
    }
    doc.update(overrides)
    return EventDescriptor(**doc)


def _make_event(data: dict, seq_num: int = 1, **overrides) -> Event:
    doc = {
        "uid": "event-001",
        "time": 0.0,
        "descriptor": "desc-001",
        "seq_num": seq_num,
        "data": data,
        "timestamps": dict.fromkeys(data, 0.0),
    }
    doc.update(overrides)
    return Event(**doc)


def _make_stop(exit_status: str = "success", **overrides) -> RunStop:
    doc = {
        "uid": "stop-001",
        "time": 0.0,
        "run_start": "start-001",
        "exit_status": exit_status,
    }
    doc.update(overrides)
    return RunStop(**doc)


@pytest.fixture(autouse=True)
def _close_all_figures():
    """Close all matplotlib figures after every test."""
    yield
    plt.close("all")


@pytest.fixture()
def plotter():
    return OptimizationPlotter()


# ---------------------------------------------------------------------------
# Utility function tests
# ---------------------------------------------------------------------------


class TestToList:
    def test_scalar(self):
        assert _to_list(5) == [5]

    def test_list(self):
        assert _to_list([1, 2, 3]) == [1, 2, 3]

    def test_tuple(self):
        assert _to_list((1, 2)) == [1, 2]

    def test_numpy_array(self):
        import numpy as np

        assert _to_list(np.array([1.0, 2.0])) == [1.0, 2.0]

    def test_numpy_scalar(self):
        import numpy as np

        assert _to_list(np.float64(3.14)) == [3.14]


class TestIsNumericDtype:
    def test_number(self):
        assert _is_numeric_dtype({"dtype": "number"}) is True

    def test_integer(self):
        assert _is_numeric_dtype({"dtype": "integer"}) is True

    def test_string(self):
        assert _is_numeric_dtype({"dtype": "string"}) is False

    def test_array_float(self):
        assert _is_numeric_dtype({"dtype": "array", "dtype_numpy": "<f8"}) is True

    def test_array_int(self):
        assert _is_numeric_dtype({"dtype": "array", "dtype_numpy": "<i4"}) is True

    def test_array_unsigned_int(self):
        assert _is_numeric_dtype({"dtype": "array", "dtype_numpy": "<u4"}) is True

    def test_array_unicode(self):
        assert _is_numeric_dtype({"dtype": "array", "dtype_numpy": "<U6"}) is False

    def test_array_no_dtype_numpy(self):
        assert _is_numeric_dtype({"dtype": "array"}) is False

    def test_boolean(self):
        assert _is_numeric_dtype({"dtype": "boolean"}) is False


class TestSubplotGrid:
    def test_single(self):
        assert _subplot_grid(1) == (1, 1)

    def test_two(self):
        assert _subplot_grid(2) == (2, 1)

    def test_three(self):
        assert _subplot_grid(3) == (3, 1)

    def test_four(self):
        assert _subplot_grid(4) == (2, 2)

    def test_five(self):
        assert _subplot_grid(5) == (3, 2)


# ---------------------------------------------------------------------------
# Constructor tests
# ---------------------------------------------------------------------------


class TestInit:
    def test_default_construction(self):
        plotter = OptimizationPlotter()
        assert plotter._param_fig is None
        assert plotter._outcome_fig is None
        assert plotter._sample_counter == 0

    def test_import_matplotlib_raises_without_matplotlib(self):
        """Verify the _import_matplotlib helper raises a clear message."""
        from blop.callbacks.plotter import _import_matplotlib

        with patch.dict("sys.modules", {"matplotlib": None, "matplotlib.pyplot": None}):
            with pytest.raises(ImportError, match="matplotlib is required"):
                _import_matplotlib()


# ---------------------------------------------------------------------------
# Descriptor tests
# ---------------------------------------------------------------------------


class TestDescriptor:
    def test_caches_data_keys(self, plotter):
        desc = _make_descriptor(
            data_keys={
                "x1": {"dtype": "number", "shape": [], "source": Source.PARAMETER.value},
                "x2": {"dtype": "number", "shape": [], "source": Source.PARAMETER.value},
                "y": {"dtype": "number", "shape": [], "source": Source.OUTCOME.value},
            }
        )
        plotter.descriptor(desc)

        assert Source.PARAMETER in plotter._sorted_data_keys_by_source
        assert Source.OUTCOME in plotter._sorted_data_keys_by_source
        assert sorted(plotter._sorted_data_keys_by_source[Source.PARAMETER]) == ["x1", "x2"]
        assert plotter._sorted_data_keys_by_source[Source.OUTCOME] == ["y"]

    def test_creates_figures(self, plotter):
        plotter.descriptor(_make_descriptor())
        assert plotter._param_fig is not None
        assert plotter._outcome_fig is not None

    def test_no_param_figure_without_param_keys(self, plotter):
        plotter.descriptor(
            _make_descriptor(
                data_keys={
                    "y": {"dtype": "number", "shape": [], "source": Source.OUTCOME.value},
                }
            )
        )
        assert plotter._param_fig is None
        assert plotter._outcome_fig is not None

    def test_no_outcome_figure_without_outcome_keys(self, plotter):
        plotter.descriptor(
            _make_descriptor(
                data_keys={
                    "x": {"dtype": "number", "shape": [], "source": Source.PARAMETER.value},
                }
            )
        )
        assert plotter._param_fig is not None
        assert plotter._outcome_fig is None


# ---------------------------------------------------------------------------
# Event tests
# ---------------------------------------------------------------------------


class TestEvent:
    def _setup(self, plotter, data_keys=None):
        plotter.start(_make_start(iterations=5))
        plotter.descriptor(_make_descriptor(data_keys=data_keys))

    def test_empty_data_returns_early(self, plotter):
        self._setup(plotter)
        doc = _make_event(data={})
        result = plotter.event(doc)
        assert result is doc
        assert plotter._sample_counter == 0

    def test_scalar_data_accumulates(self, plotter):
        self._setup(plotter)
        plotter.event(_make_event(data={"x": 1.5, "y": 3.14}))

        assert plotter._parameter_history["x"] == [1.5]
        assert plotter._outcome_history["y"] == [3.14]
        assert plotter._sample_counter == 1

    def test_batch_data_accumulates(self, plotter):
        self._setup(plotter)
        plotter.event(
            _make_event(
                data={
                    "x": [1.0, 2.0, 3.0],
                    "y": [10.0, 20.0, 30.0],
                    "suggestion_ids": ["s1", "s2", "s3"],
                }
            )
        )
        assert plotter._parameter_history["x"] == [1.0, 2.0, 3.0]
        assert plotter._outcome_history["y"] == [10.0, 20.0, 30.0]
        assert plotter._sample_counter == 3

    def test_nan_padded_entries_filtered(self, plotter):
        self._setup(plotter)
        plotter.event(
            _make_event(
                data={
                    "x": [1.0, 2.0, float("nan")],
                    "y": [10.0, 20.0, float("nan")],
                    "suggestion_ids": ["s1", "s2", ""],
                }
            )
        )
        assert plotter._parameter_history["x"] == [1.0, 2.0]
        assert plotter._outcome_history["y"] == [10.0, 20.0]
        assert plotter._sample_counter == 2

    def test_multiple_events_accumulate(self, plotter):
        self._setup(plotter)
        plotter.event(_make_event(data={"x": 1.0, "y": 10.0}))
        plotter.event(_make_event(data={"x": 2.0, "y": 20.0}))
        plotter.event(_make_event(data={"x": 3.0, "y": 30.0}))

        assert plotter._parameter_history["x"] == [1.0, 2.0, 3.0]
        assert plotter._outcome_history["y"] == [10.0, 20.0, 30.0]
        assert plotter._sample_counter == 3
        assert plotter._outcome_sample_indices == [1, 2, 3]

    def test_string_parameter(self, plotter):
        self._setup(
            plotter,
            data_keys={
                "coating": {"dtype": "string", "shape": [], "source": Source.PARAMETER.value},
                "y": {"dtype": "number", "shape": [], "source": Source.OUTCOME.value},
            },
        )
        plotter.event(_make_event(data={"coating": "gold", "y": 5.0}))
        plotter.event(_make_event(data={"coating": "silver", "y": 8.0}))
        plotter.event(_make_event(data={"coating": "gold", "y": 6.0}))

        assert plotter._parameter_history["coating"] == ["gold", "silver", "gold"]

    def test_returns_doc(self, plotter):
        self._setup(plotter)
        doc = _make_event(data={"x": 1.0, "y": 5.0})
        result = plotter.event(doc)
        assert result is doc

    def test_nan_outcome_skipped(self, plotter):
        self._setup(plotter)
        plotter.event(_make_event(data={"x": 1.0, "y": float("nan")}))
        assert plotter._outcome_history["y"] == []

    def test_inf_values_skipped(self, plotter):
        self._setup(plotter)
        plotter.event(_make_event(data={"x": float("inf"), "y": float("-inf")}))
        assert plotter._parameter_history["x"] == []
        assert plotter._outcome_history["y"] == []


# ---------------------------------------------------------------------------
# Stop tests
# ---------------------------------------------------------------------------


class TestStop:
    def test_records_run_boundary(self, plotter):
        plotter.start(_make_start(iterations=1))
        plotter.descriptor(_make_descriptor())
        plotter.event(_make_event(data={"x": 1.0, "y": 5.0}))
        plotter.stop(_make_stop())

        assert plotter._run_boundaries == [1]

    def test_no_boundary_if_no_events(self, plotter):
        plotter.start(_make_start())
        plotter.descriptor(_make_descriptor())
        plotter.stop(_make_stop())

        assert plotter._run_boundaries == []


# ---------------------------------------------------------------------------
# Persistence across runs
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_data_persists_across_runs(self, plotter):
        # Run 1
        plotter.start(_make_start(iterations=2))
        plotter.descriptor(_make_descriptor())
        plotter.event(_make_event(data={"x": 1.0, "y": 10.0}))
        plotter.event(_make_event(data={"x": 2.0, "y": 20.0}))
        plotter.stop(_make_stop())

        # Run 2
        plotter.start(_make_start(iterations=2))
        plotter.descriptor(_make_descriptor())
        plotter.event(_make_event(data={"x": 3.0, "y": 30.0}))
        plotter.event(_make_event(data={"x": 4.0, "y": 40.0}))
        plotter.stop(_make_stop())

        assert plotter._parameter_history["x"] == [1.0, 2.0, 3.0, 4.0]
        assert plotter._outcome_history["y"] == [10.0, 20.0, 30.0, 40.0]
        assert plotter._sample_counter == 4
        assert plotter._outcome_sample_indices == [1, 2, 3, 4]
        assert plotter._run_boundaries == [2, 4]


# ---------------------------------------------------------------------------
# Figure recreation
# ---------------------------------------------------------------------------


class TestFigureRecreation:
    def test_figures_recreated_if_closed(self, plotter):
        plotter.start(_make_start(iterations=1))
        plotter.descriptor(_make_descriptor())
        assert plotter._param_fig is not None

        old_param_num = plotter._param_fig.number
        plt.close(plotter._param_fig)

        # Next event should recreate figures
        plotter.event(_make_event(data={"x": 1.0, "y": 5.0}))
        assert plotter._param_fig is not None
        assert plotter._param_fig.number != old_param_num


# ---------------------------------------------------------------------------
# Full lifecycle
# ---------------------------------------------------------------------------


class TestFullLifecycle:
    def test_full_run_with_batches(self, plotter):
        plotter.start(
            _make_start(
                iterations=3,
                n_points=2,
                optimizer="BoTorch",
                actuators=["x"],
                sensors=["y"],
            )
        )
        plotter.descriptor(_make_descriptor())

        for i in range(3):
            plotter.event(
                _make_event(
                    data={
                        "x": [float(i), float(i) + 0.5],
                        "y": [float(i) * 10, float(i) * 10 + 5],
                        "suggestion_ids": [f"s{2 * i}", f"s{2 * i + 1}"],
                    },
                    seq_num=i + 1,
                    uid=f"event-{i:03d}",
                )
            )

        plotter.stop(_make_stop())

        assert plotter._parameter_history["x"] == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
        assert plotter._outcome_history["y"] == [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]
        assert plotter._sample_counter == 6

    def test_mixed_numeric_and_categorical(self, plotter):
        plotter.start(_make_start(iterations=2))
        plotter.descriptor(
            _make_descriptor(
                data_keys={
                    "motor_x": {"dtype": "number", "shape": [], "source": Source.PARAMETER.value},
                    "filter": {"dtype": "string", "shape": [], "source": Source.PARAMETER.value},
                    "intensity": {"dtype": "number", "shape": [], "source": Source.OUTCOME.value},
                }
            )
        )

        plotter.event(_make_event(data={"motor_x": 1.0, "filter": "Al", "intensity": 100.0}))
        plotter.event(_make_event(data={"motor_x": 2.0, "filter": "Si", "intensity": 200.0}))
        plotter.stop(_make_stop())

        assert plotter._parameter_history["motor_x"] == [1.0, 2.0]
        assert plotter._parameter_history["filter"] == ["Al", "Si"]
        assert plotter._outcome_history["intensity"] == [100.0, 200.0]
        # Param figure should have 2 axes
        assert len(plotter._param_axes) == 2

    def test_batch_array_string_parameters(self, plotter):
        """Array-valued string parameters (n_points > 1 ChoiceDOF)."""
        plotter.start(_make_start(iterations=1))
        plotter.descriptor(
            _make_descriptor(
                data_keys={
                    "coating": {"dtype": "array", "dtype_numpy": "<U6", "shape": [3], "source": Source.PARAMETER.value},
                    "y": {"dtype": "number", "shape": [], "source": Source.OUTCOME.value},
                }
            )
        )

        plotter.event(
            _make_event(
                data={
                    "coating": ["gold", "silver", "gold"],
                    "y": [10.0, 20.0, 30.0],
                    "suggestion_ids": ["s1", "s2", "s3"],
                }
            )
        )
        plotter.stop(_make_stop())

        assert plotter._parameter_history["coating"] == ["gold", "silver", "gold"]

    def test_with_datakey_limits(self, plotter):
        """Histogram should use limits from DataKey when available."""
        plotter.start(_make_start(iterations=1))
        plotter.descriptor(
            _make_descriptor(
                data_keys={
                    "x": {
                        "dtype": "number",
                        "shape": [],
                        "source": Source.PARAMETER.value,
                        "limits": {"control": {"low": 0.0, "high": 10.0}},
                    },
                    "y": {"dtype": "number", "shape": [], "source": Source.OUTCOME.value},
                }
            )
        )
        plotter.event(_make_event(data={"x": 5.0, "y": 50.0}))
        plotter.stop(_make_stop())

        # Verify it ran without error — the limits are consumed internally
        assert plotter._parameter_history["x"] == [5.0]

    def test_with_datakey_choices(self, plotter):
        """Bar chart should use choices from DataKey when available."""
        plotter.start(_make_start(iterations=1))
        plotter.descriptor(
            _make_descriptor(
                data_keys={
                    "filter": {
                        "dtype": "string",
                        "shape": [],
                        "source": Source.PARAMETER.value,
                        "choices": ["Al", "Si", "Cu"],
                    },
                    "y": {"dtype": "number", "shape": [], "source": Source.OUTCOME.value},
                }
            )
        )
        # Only sample "Al" but "Si" and "Cu" should still appear in the chart
        plotter.event(_make_event(data={"filter": "Al", "y": 50.0}))
        plotter.stop(_make_stop())

        assert plotter._parameter_history["filter"] == ["Al"]
