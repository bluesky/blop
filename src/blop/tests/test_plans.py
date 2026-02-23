from unittest.mock import MagicMock, Mock, patch

import bluesky.plan_stubs as bps
import pytest
from bluesky.run_engine import RunEngine
from bluesky.utils import plan

from blop.plans import acquire_baseline, acquire_with_background, default_acquire, optimize, optimize_step
from blop.protocols import AcquisitionPlan, Checkpointable, EvaluationFunction, OptimizationProblem, Optimizer

from .conftest import MovableSignal, ReadableSignal


@plan
def _test_acquisition_plan(suggestions, actuators, sensors, *args, **kwargs):
    """Acquisition plan that returns a predictable uid for testing."""
    yield from bps.null()
    return "test-uid-123"


def _collect_optimize_events():
    """Return a callback and list that collect event docs from the outer optimize run."""
    events = []
    optimize_run_uid = None
    optimize_descriptors = set()

    def callback(name, doc):
        nonlocal optimize_run_uid
        if name == "start" and doc.get("run_key") == "optimize":
            optimize_run_uid = doc["uid"]
        elif name == "descriptor" and doc.get("run_start") == optimize_run_uid:
            optimize_descriptors.add(doc["uid"])
        elif name == "event" and doc.get("descriptor") in optimize_descriptors:
            events.append(doc)

    return callback, events


class CheckpointableOptimizer(Optimizer, Checkpointable): ...


@pytest.fixture(scope="function")
def RE():
    return RunEngine({})


def test_optimize(RE):
    optimizer = MagicMock(spec=Optimizer)
    optimizer.suggest.return_value = [{"x1": 0.0, "_id": 0}]
    evaluation_function = MagicMock(spec=EvaluationFunction, return_value=[{"objective": 0.0, "_id": 0}])
    optimization_problem = OptimizationProblem(
        optimizer=optimizer,
        actuators=[MovableSignal("x1", initial_value=-1.0)],
        sensors=[ReadableSignal("objective")],
        evaluation_function=evaluation_function,
    )

    callback, events = _collect_optimize_events()
    RE.subscribe(callback)
    try:
        RE(optimize(optimization_problem))
    finally:
        RE.unsubscribe(callback)

    optimizer.suggest.assert_called_once_with(1)
    optimizer.ingest.assert_called_once_with([{"objective": 0.0, "_id": 0}])
    assert evaluation_function.call_count == 1

    # Validate event documents from outer-plan _read_step
    assert len(events) == 1
    data = events[0]["data"]
    assert "suggestion_ids" in data
    assert "bluesky_uid" in data
    assert "x1" in data
    assert "objective" in data
    assert data["x1"] == 0.0
    assert data["objective"] == 0.0
    assert data["bluesky_uid"] and isinstance(data["bluesky_uid"], str)


def test_optimize_multiple(RE):
    optimizer = MagicMock(spec=Optimizer)
    optimizer.suggest.return_value = [{"x1": 0.0, "_id": 0}]
    evaluation_function = MagicMock(spec=EvaluationFunction, return_value=[{"objective": 0.0, "_id": 0}])
    optimization_problem = OptimizationProblem(
        optimizer=optimizer,
        actuators=[MovableSignal("x1", initial_value=-1.0)],
        sensors=[ReadableSignal("objective")],
        evaluation_function=evaluation_function,
    )

    callback, events = _collect_optimize_events()
    RE.subscribe(callback)
    try:
        RE(optimize(optimization_problem, iterations=5))
    finally:
        RE.unsubscribe(callback)

    optimizer.suggest.assert_called_with(1)
    optimizer.ingest.assert_called_with([{"objective": 0.0, "_id": 0}])
    assert optimizer.suggest.call_count == 5
    assert optimizer.ingest.call_count == 5
    assert evaluation_function.call_count == 5

    # Validate event documents from outer-plan _read_step
    assert len(events) == 5
    for event in events:
        data = event["data"]
        assert "suggestion_ids" in data
        assert "bluesky_uid" in data
        assert "x1" in data
        assert "objective" in data
        assert data["x1"] == 0.0
        assert data["objective"] == 0.0


def test_optimize_multiple_with_n_points(RE):
    optimizer = MagicMock(spec=Optimizer)
    optimizer.suggest.return_value = [{"x1": 0.0, "_id": 0}, {"x1": 0.1, "_id": 1}]
    evaluation_function = MagicMock(
        spec=EvaluationFunction, return_value=[{"objective": 0.0, "_id": 0}, {"objective": 0.1, "_id": 1}]
    )
    optimization_problem = OptimizationProblem(
        optimizer=optimizer,
        actuators=[MovableSignal("x1", initial_value=-1.0)],
        sensors=[ReadableSignal("objective")],
        evaluation_function=evaluation_function,
    )
    callback, events = _collect_optimize_events()
    RE.subscribe(callback)
    try:
        RE(optimize(optimization_problem, iterations=5, n_points=2))
    finally:
        RE.unsubscribe(callback)
    optimizer.suggest.assert_called_with(2)
    optimizer.ingest.assert_called_with([{"objective": 0.0, "_id": 0}, {"objective": 0.1, "_id": 1}])
    assert optimizer.suggest.call_count == 5
    assert optimizer.ingest.call_count == 5
    assert evaluation_function.call_count == 5

    # Validate event documents from outer-plan _read_step
    assert len(events) == 5
    for event in events:
        data = event["data"]
        assert "suggestion_ids" in data
        assert "bluesky_uid" in data
        assert "x1" in data
        assert "objective" in data
        sid = data["suggestion_ids"]
        assert len(list(sid)) == 2
        x1_vals = list(data["x1"]) if hasattr(data["x1"], "__iter__") and not isinstance(data["x1"], str) else [data["x1"]]
        obj_vals = (
            list(data["objective"])
            if hasattr(data["objective"], "__iter__") and not isinstance(data["objective"], str)
            else [data["objective"]]
        )
        assert x1_vals == [0.0, 0.1]
        assert obj_vals == [0.0, 0.1]


def test_optimize_complex_case(RE):
    """Test with multi-suggest, multi-parameter, multi-objective, multi-readable case."""

    def _to_list(x):
        return list(x) if hasattr(x, "__iter__") and not isinstance(x, str) else [x]

    optimizer = MagicMock(spec=Optimizer)
    optimizer.suggest.return_value = [
        {"x1": 0.0, "x2": 0.0, "x3": 0.0, "_id": 0},
        {"x1": 0.1, "x2": 0.2, "x3": 0.3, "_id": 1},
    ]
    evaluation_function = MagicMock(
        spec=EvaluationFunction,
        return_value=[
            {"objective1": 0.0, "objective2": 0.1, "_id": 0},
            {"objective1": 0.1, "objective2": 0.2, "_id": 1},
        ],
    )
    optimization_problem = OptimizationProblem(
        optimizer=optimizer,
        actuators=[
            MovableSignal("x1", initial_value=-1.0),
            MovableSignal("x2", initial_value=-1.0),
            MovableSignal("x3", initial_value=-1.0),
        ],
        sensors=[ReadableSignal("readable1"), ReadableSignal("readable2")],
        evaluation_function=evaluation_function,
    )

    callback, events = _collect_optimize_events()
    RE.subscribe(callback)
    try:
        uids = RE(optimize(optimization_problem, iterations=2, n_points=2))
    finally:
        RE.unsubscribe(callback)

    optimizer.suggest.assert_called_with(2)
    optimizer.ingest.assert_called_with(
        [
            {"objective1": 0.0, "objective2": 0.1, "_id": 0},
            {"objective1": 0.1, "objective2": 0.2, "_id": 1},
        ]
    )
    assert optimizer.suggest.call_count == 2
    assert optimizer.ingest.call_count == 2
    assert evaluation_function.call_count == 2

    # Validate event documents from outer-plan _read_step
    assert len(events) == 2
    for event in events:
        data = event["data"]
        assert "suggestion_ids" in data
        assert "bluesky_uid" in data
        assert "x1" in data
        assert "x2" in data
        assert "x3" in data
        assert "objective1" in data
        assert "objective2" in data
        assert _to_list(data["x1"]) == [0.0, 0.1]
        assert _to_list(data["x2"]) == [0.0, 0.2]
        assert _to_list(data["x3"]) == [0.0, 0.3]
        assert _to_list(data["objective1"]) == [0.0, 0.1]
        assert _to_list(data["objective2"]) == [0.1, 0.2]
        assert _to_list(data["suggestion_ids"]) == ["0", "1"]
        assert data["bluesky_uid"] in uids


@pytest.mark.parametrize("checkpoint_interval", [0, 1, 2, 3])
def test_optimize_with_checkpoint_every_iteration(RE, checkpoint_interval):
    optimizer = MagicMock(spec=CheckpointableOptimizer)
    optimizer.suggest.return_value = [{"x1": 0.0, "_id": 0}]
    evaluation_function = MagicMock(spec=EvaluationFunction, return_value=[{"objective": 0.0, "_id": 0}])
    optimization_problem = OptimizationProblem(
        optimizer=optimizer,
        actuators=[MovableSignal("x1", initial_value=-1.0)],
        sensors=[ReadableSignal("objective")],
        evaluation_function=evaluation_function,
    )

    with patch.object(optimizer, "checkpoint", wraps=optimizer.checkpoint) as mock_checkpoint:
        RE(optimize(optimization_problem, iterations=5, n_points=2, checkpoint_interval=checkpoint_interval))
        if checkpoint_interval == 0:
            assert mock_checkpoint.call_count == 0
        else:
            assert mock_checkpoint.call_count == 5 // checkpoint_interval


def test_optimize_with_non_checkpointable_optimizer(RE):
    optimizer = MagicMock(spec=Optimizer)
    optimizer.suggest.return_value = [{"x1": 0.0, "_id": 0}]
    evaluation_function = MagicMock(spec=EvaluationFunction, return_value=[{"objective": 0.0, "_id": 0}])
    optimization_problem = OptimizationProblem(
        optimizer=optimizer,
        actuators=[MovableSignal("x1", initial_value=-1.0)],
        sensors=[ReadableSignal("objective")],
        evaluation_function=evaluation_function,
    )
    with pytest.raises(ValueError):
        RE(optimize(optimization_problem, iterations=5, n_points=2, checkpoint_interval=1))


def test_optimize_step_default(RE):
    optimizer = MagicMock(spec=Optimizer)
    optimizer.suggest.return_value = [{"x1": 0.0, "_id": 0}]
    evaluation_function = MagicMock(spec=EvaluationFunction, return_value=[{"objective": 0.0, "_id": 0}])
    optimization_problem = OptimizationProblem(
        optimizer=optimizer,
        actuators=[MovableSignal("x1", initial_value=-1.0)],
        sensors=[ReadableSignal("objective")],
        evaluation_function=evaluation_function,
    )

    RE(optimize_step(optimization_problem))

    optimizer.suggest.assert_called_once_with(1)
    optimizer.ingest.assert_called_once_with([{"objective": 0.0, "_id": 0}])
    assert evaluation_function.call_count == 1


def test_optimize_event_document_structure(RE):
    """Validate the event document structure from the outer-plan _read_step in detail."""
    optimizer = MagicMock(spec=Optimizer)
    optimizer.suggest.return_value = [{"x1": 0.5, "_id": 0}]
    evaluation_function = MagicMock(spec=EvaluationFunction, return_value=[{"objective": 1.25, "_id": 0}])
    optimization_problem = OptimizationProblem(
        optimizer=optimizer,
        actuators=[MovableSignal("x1", initial_value=-1.0)],
        sensors=[ReadableSignal("objective")],
        evaluation_function=evaluation_function,
        acquisition_plan=_test_acquisition_plan,
    )

    callback, events = _collect_optimize_events()
    RE.subscribe(callback)
    try:
        RE(optimize(optimization_problem))
    finally:
        RE.unsubscribe(callback)

    assert len(events) == 1
    data = events[0]["data"]

    # Validate required fields from _read_step
    assert "suggestion_ids" in data
    assert "bluesky_uid" in data
    assert "x1" in data
    assert "objective" in data

    # Validate predictable values from custom acquisition plan
    assert data["bluesky_uid"] == "test-uid-123"
    assert data["x1"] == 0.5
    assert data["objective"] == 1.25
    assert data["suggestion_ids"] == "0"


def test_optimize_step_custom_acquisition_plan(RE):
    acquisition_plan = MagicMock(spec=AcquisitionPlan)
    optimizer = MagicMock(spec=Optimizer)
    optimizer.suggest.return_value = [{"x1": 0.0, "_id": 0}]
    evaluation_function = MagicMock(spec=EvaluationFunction, return_value=[{"objective": 0.0, "_id": 0}])
    movable = MovableSignal("x1", initial_value=-1.0)
    readable = ReadableSignal("objective")
    optimization_problem = OptimizationProblem(
        optimizer=optimizer,
        actuators=[movable],
        sensors=[readable],
        evaluation_function=evaluation_function,
        acquisition_plan=acquisition_plan,
    )

    RE(optimize_step(optimization_problem))
    optimizer.suggest.assert_called_once_with(1)
    acquisition_plan.assert_called_once_with(
        [{"x1": 0.0, "_id": 0}],
        [movable],
        [readable],
    )
    optimizer.ingest.assert_called_once_with([{"objective": 0.0, "_id": 0}])
    assert evaluation_function.call_count == 1


def test_default_acquire_single_movable_readable(RE):
    """Test with single movable, position, and readable."""
    movable = MovableSignal("x1", initial_value=-1.0)
    readable = ReadableSignal("objective")
    with patch.object(readable, "read", wraps=readable.read) as mock_read:
        RE(
            default_acquire(
                [{"x1": 0.0, "_id": 0}],
                [movable],
                [readable],
            )
        )
        assert mock_read.call_count == 1

    assert movable.read()["x1"]["value"] == 0.0


def test_default_acquire_multiple_movables_readables(RE):
    """Test with multiple movables, positions, and readables."""
    movable1 = MovableSignal("x1", initial_value=-1.0)
    movable2 = MovableSignal("x2", initial_value=-1.0)
    readable1 = ReadableSignal("objective1")
    readable2 = ReadableSignal("objective2")

    with (
        patch.object(movable1, "set", wraps=movable1.set) as mock_set1,
        patch.object(movable2, "set", wraps=movable2.set) as mock_set2,
        patch.object(readable1, "read", wraps=readable1.read) as mock_read1,
        patch.object(readable2, "read", wraps=readable2.read) as mock_read2,
    ):
        RE(
            default_acquire(
                [{"x1": 0.0, "x2": 0.0, "_id": 0}, {"x1": 0.1, "x2": 0.1, "_id": 1}],
                [movable1, movable2],
                [readable1, readable2],
            )
        )

        # Verify movables were set in correct order
        assert mock_set1.call_count == 2
        assert mock_set2.call_count == 2
        assert mock_set1.call_args_list[0][0][0] == 0.0  # First call
        assert mock_set2.call_args_list[0][0][0] == 0.0
        assert mock_set1.call_args_list[1][0][0] == 0.1  # Second call
        assert mock_set2.call_args_list[1][0][0] == 0.1

        # Verify reads happened twice
        assert mock_read1.call_count == 2
        assert mock_read2.call_count == 2

    # Verify final positions
    assert movable1.read()["x1"]["value"] == 0.1
    assert movable2.read()["x2"]["value"] == 0.1


def test_acquire_with_background(RE):
    """Test background acquisition with multiple movables, positions, and readables"""

    def block_beam():
        yield from bps.null()

    def unblock_beam():
        yield from bps.null()

    movable = MovableSignal("x1", initial_value=-1.0)
    readable = ReadableSignal("objective")

    mock_block_beam = Mock(wraps=block_beam)
    mock_unblock_beam = Mock(wraps=unblock_beam)

    with patch.object(readable, "read", wraps=readable.read) as mock_read:
        RE(
            acquire_with_background(
                [{"x1": 0.0, "_id": 0}],
                [movable],
                sensors=[readable],
                block_beam=mock_block_beam,
                unblock_beam=mock_unblock_beam,
            )
        )
        # Two reads, one blocked, one unblocked
        assert mock_read.call_count == 2
        assert mock_block_beam.call_count == 1
        assert mock_unblock_beam.call_count == 1

    assert movable.read()["x1"]["value"] == 0.0


def test_acquire_baseline(RE):
    """Test acquiring a baseline reading from suggested parameterizations."""
    optimizer = MagicMock(spec=Optimizer)
    evaluation_function = MagicMock(spec=EvaluationFunction, return_value=[{"objective": 0.0, "_id": "baseline"}])

    optimization_problem = OptimizationProblem(
        optimizer=optimizer,
        actuators=[MovableSignal("x1", initial_value=-1.0)],
        sensors=[ReadableSignal("objective")],
        evaluation_function=evaluation_function,
    )

    RE(acquire_baseline(optimization_problem, parameterization={"x1": 0.0}))

    # No suggestions are made since this is a baseline reading
    assert optimizer.suggest.call_count == 0

    optimizer.ingest.assert_called_once_with([{"objective": 0.0, "_id": "baseline", "x1": 0.0}])
    assert evaluation_function.call_count == 1


def test_acquire_baseline_from_current(RE):
    """Test acquiring a baseline reading from the current movable positions."""
    optimizer = MagicMock(spec=Optimizer)
    evaluation_function = MagicMock(spec=EvaluationFunction, return_value=[{"objective": 0.0, "_id": "baseline"}])
    movable = MovableSignal("x1", initial_value=-1.0)

    optimization_problem = OptimizationProblem(
        optimizer=optimizer,
        actuators=[movable],
        sensors=[ReadableSignal("objective")],
        evaluation_function=evaluation_function,
    )

    with (
        patch.object(movable, "set", wraps=movable.set) as mock_set,
        patch.object(movable, "read", wraps=movable.read) as mock_read,
    ):
        RE(acquire_baseline(optimization_problem))

        # Ensure the movable was read twice (once for the baseline, once during the acquisition)
        assert mock_read.call_count == 2
        # Ensure the movable was set once to the current value
        assert mock_set.call_count == 1
        assert mock_set.call_args_list[0][0][0] == -1.0

    # No suggestions are made since this is a baseline reading from the current movable positions
    assert optimizer.suggest.call_count == 0

    optimizer.ingest.assert_called_once_with([{"objective": 0.0, "_id": "baseline", "x1": -1.0}])
    assert evaluation_function.call_count == 1
