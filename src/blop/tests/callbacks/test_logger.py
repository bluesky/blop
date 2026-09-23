"""Unit tests for the OptimizationLogger callback."""

import math
import re
from unittest.mock import MagicMock

import numpy as np
import pytest
from event_model import Event, EventDescriptor, RunStart, RunStop
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from blop.callbacks.logger import OptimizationLogger
from blop.callbacks.utils import RunningStats
from blop.plan_stubs import _ITERATION_KEY
from blop.utils import Source


@pytest.fixture()
def console():
    return MagicMock(spec=Console)


@pytest.fixture()
def logger(console):
    return OptimizationLogger(console=console)


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


def _setup_descriptor(logger, data_keys=None):
    """Feed a descriptor so the logger knows about param/outcome keys."""
    logger.descriptor(_make_descriptor(data_keys=data_keys))


def _run_to_stop(logger, exit_status="success", reason=""):
    """Feed start -> descriptor -> event -> stop so summary stats exist."""
    logger.start(_make_start(iterations=1))
    logger.descriptor(_make_descriptor())
    logger.event(_make_event(data={"x": 1.0, "y": 5.0}))
    logger.stop(_make_stop(exit_status=exit_status, reason=reason))


def _collect_ruler_info(call):
    args, kwargs = call
    return args[0] if args else None


def _collect_iterations_from_header(call):
    args, kwargs = call
    console_header = args[0]
    if not isinstance(console_header, Panel):
        return None
    header_text = str(console_header.renderable)
    matching = re.search(r"Iterations (\d+)\s*(?:more \((\d+) completed, (\d+) total\))?", header_text)
    if not matching:
        print(header_text)
        return re.search(r"Iterations.*", header_text).group()
    return [int(i) if i else None for i in matching.groups()]


def _collect_values_from_table(call):
    args, kwargs = call
    table = args[0]
    if not isinstance(table, Table):
        return None
    return np.array([[i if i else -np.inf for i in col.cells] for col in table.columns], dtype=float)


def test_start_minimal(logger, console):
    logger.start(_make_start())
    assert console.print.call_count >= 1


def test_start_with_full_metadata(logger, console):
    logger.start(
        _make_start(
            iterations=10,
            n_points=3,
            optimizer="BoTorch",
            actuators=["mirror_x", "mirror_y"],
            sensors=["detector"],
        )
    )
    assert console.print.call_count >= 1


def test_descriptor(logger):
    """descriptor() should accept a well-formed EventDescriptor without error."""
    logger.descriptor(
        _make_descriptor(
            data_keys={
                "x1": {"dtype": "number", "shape": [], "source": Source.PARAMETER.value},
                "x2": {"dtype": "number", "shape": [], "source": Source.PARAMETER.value},
                "y": {"dtype": "number", "shape": [], "source": Source.OUTCOME.value},
                "other": {"dtype": "string", "shape": [], "source": Source.OTHER.value},
            }
        )
    )


def test_event_empty_data_returns_early(logger, console):
    _setup_descriptor(logger)
    doc = _make_event(data={})
    result = logger.event(doc)
    assert result is doc
    assert console.print.call_count == 1


def test_event_scalar_data(logger, console):
    _setup_descriptor(logger)
    doc = _make_event(data={"x": 1.5, "y": 3.14})
    result = logger.event(doc)
    assert result is doc
    assert console.print.call_count >= 1


def test_event_produces_table_containing_results(logger, console):
    _setup_descriptor(logger)
    for x in range(4):
        data = {"x": 1.5 / (1 + x), "y": 3.14**-x}
        doc = _make_event(data=data)
        logger.event(doc)
        call = console.print.call_args_list[-1]
        table_vals = _collect_values_from_table(call)
        for item in data.values():
            assert np.any(np.isclose(table_vals, item, rtol=.001, atol=.001))


def test_event_without_iteration_limit_omits_total(logger, console):
    """An unbounded optimization should not display a misleading total."""
    logger.start(_make_start(iterations=None))
    _setup_descriptor(logger)
    for _ in range(5):
        logger.event(_make_event(data={"x": 1.5, "y": 3.14}))

    console_header = _collect_iterations_from_header(console.print.call_args_list[0])
    assert "Until stopping criterion" in console_header
    for call in console.rule.call_args_list:
        title = _collect_ruler_info(call)
        if title:
            assert "/" not in title


def test_event_multiple_iterations(logger, console):
    """Successive events should accumulate without error and runtime metadata should display"""
    _setup_descriptor(logger)
    for _ in range(4):
        logger.event(_make_event(data={"x": 1.0, "y": 10.0}))
        logger.event(_make_event(data={"x": 3.0, "y": 20.0}))
    assert console.print.call_count >= 1
    assert console.rule.call_count >= 1


def test_event_non_numeric_data(logger, console):
    """String-valued outcomes should not crash the logger."""
    _setup_descriptor(
        logger,
        data_keys={
            "x": {"dtype": "number", "shape": [], "source": Source.PARAMETER.value},
            "label": {"dtype": "string", "shape": [], "source": Source.OUTCOME.value},
        },
    )
    doc = _make_event(data={"x": 1.0, "label": "good"})
    result = logger.event(doc)
    assert result is doc


def test_stop_success(logger, console):
    _run_to_stop(logger, exit_status="success")
    assert console.rule.call_count >= 1
    assert "Complete" in _collect_ruler_info(console.rule.call_args_list[-1])


def test_stop_abort_with_reason(logger, console):
    _run_to_stop(logger, exit_status="abort", reason="user interrupt")
    assert console.rule.call_count >= 1
    assert "Aborted" in _collect_ruler_info(console.rule.call_args_list[-1])


def test_stop_other_status(logger, console):
    _run_to_stop(logger, exit_status="fail", reason="hardware fault")
    assert console.rule.call_count >= 1
    assert "Stopped" in _collect_ruler_info(console.rule.call_args_list[-1])


def test_stop_without_events(logger, console):
    """stop() should not crash even if no events were received."""
    logger.start(_make_start())
    logger.descriptor(_make_descriptor())
    logger.stop(_make_stop())
    assert console.rule.call_count >= 1


def test_full_lifecycle(logger, console):
    """Run through the full document sequence without error."""
    logger.start(
        _make_start(
            iterations=3,
            n_points=2,
            optimizer="BoTorch",
            actuators=["x"],
            sensors=["y"],
        )
    )
    logger.descriptor(_make_descriptor())

    for i in range(3):
        logger.event(
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

    logger.stop(_make_stop())

    assert console.print.call_count >= 1
    assert console.rule.call_count >= 1


def test_multi_run_accumulates_iterations(logger, console):
    """Calling start() twice should bump the internal iteration bookkeeping by how many iterations have been recorded"""
    logger.start(_make_start(iterations=5))
    logger.descriptor(_make_descriptor())
    for i in range(3):
        doc = _make_event(data={"x": 1.5 * i, "y": 3.14, _ITERATION_KEY: i})
        logger.event(doc)
    logger.stop(_make_stop())
    logger.start(_make_start(iterations=3))

    call_list = console.print.call_args_list

    iter_metadata = _collect_iterations_from_header(call_list[0])
    assert iter_metadata == [5, None, None]

    iter_metadata = _collect_iterations_from_header(call_list[-1])
    assert iter_metadata == [3, 3, 6]


def test_running_stats_empty():
    stats = RunningStats()
    assert stats.count == 0
    assert stats.min == math.inf
    assert stats.max == -math.inf
    assert math.isnan(stats.mean)
    assert math.isnan(stats.std)


def test_running_stats_single_value():
    stats = RunningStats()
    stats.update(5.0)
    assert stats.count == 1
    assert stats.min == 5.0
    assert stats.max == 5.0
    assert stats.mean == 5.0
    assert math.isnan(stats.std)


def test_running_stats_two_identical_values():
    stats = RunningStats()
    stats.update(3.0)
    stats.update(3.0)
    assert stats.count == 2
    assert stats.mean == 3.0
    assert stats.std == 0.0


def test_running_stats_known_values():
    values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    stats = RunningStats()
    for v in values:
        stats.update(v)

    assert stats.count == len(values)
    assert stats.min == np.min(values)
    assert stats.max == np.max(values)
    assert math.isclose(stats.mean, np.mean(values))
    assert math.isclose(stats.std, np.std(values, ddof=1))


def test_running_stats_large_offset():
    """Welford's algorithm should be numerically stable with large offsets."""
    values = [1e9 + 1, 1e9 + 2, 1e9 + 3]
    stats = RunningStats()
    for v in values:
        stats.update(v)

    assert math.isclose(stats.mean, np.mean(values))
    assert math.isclose(stats.std, np.std(values, ddof=1))


def test_running_stats_negative_values():
    values = [-10.0, -5.0, 0.0, 5.0, 10.0]
    stats = RunningStats()
    for v in values:
        stats.update(v)

    assert stats.min == -10.0
    assert stats.max == 10.0
    assert math.isclose(stats.mean, np.mean(values))
    assert math.isclose(stats.std, np.std(values, ddof=1))


def test_running_stats_skips_nan():
    stats = RunningStats()
    stats.update(1.0)
    stats.update(float("nan"))
    stats.update(3.0)
    assert stats.count == 2
    assert stats.min == 1.0
    assert stats.max == 3.0
    assert math.isclose(stats.mean, 2.0)


def test_running_stats_skips_inf():
    stats = RunningStats()
    stats.update(1.0)
    stats.update(float("inf"))
    stats.update(float("-inf"))
    stats.update(3.0)
    assert stats.count == 2
    assert stats.min == 1.0
    assert stats.max == 3.0
