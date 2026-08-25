from unittest.mock import MagicMock, patch

import bluesky.plan_stubs as bps
import pytest
from bluesky.run_engine import RunEngine
from bluesky.utils import plan

from blop.plan_stubs import list_scan_in_run
from blop.plans import (
    acquire_baseline,
    default_acquire,
    optimize,
    optimize_in_run,
    optimize_step,
)
from blop.protocols import (
    AcquisitionPlan,
    EvaluationFunction,
    OptimizationProblem,
    Optimizer,
    SupportsStoppingCriteria,
    TrialFaultAware,
)

from .conftest import CheckpointableOptimizer, MovableSignal, ReadableSignal


@plan
def _test_acquisition_plan(suggestions, actuators, sensors, *args, **kwargs):
    """Acquisition plan that returns a predictable uid for testing."""
    yield from bps.null()
    return "test-uid-123"


@plan
def _tuple_acquisition_plan(suggestions, actuators, sensors, *args, **kwargs):
    """Acquisition plan that returns a tuple acquisition identifier."""
    yield from bps.null()
    return ("event-a", "event-b")


class _CustomAcquisitionIdentifier:
    def __hash__(self):
        return 1

    def __repr__(self):
        return "CustomAcquisitionIdentifier()"


_CUSTOM_ACQUISITION_IDENTIFIER = _CustomAcquisitionIdentifier()


@plan
def _custom_identifier_acquisition_plan(suggestions, actuators, sensors, *args, **kwargs):
    """Acquisition plan that returns a custom hashable acquisition identifier."""
    yield from bps.null()
    return _CUSTOM_ACQUISITION_IDENTIFIER


class StageableReadable(ReadableSignal):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.stage_count = 0
        self.unstage_count = 0

    def stage(self):
        self.stage_count += 1
        return [self]

    def unstage(self):
        self.unstage_count += 1
        return [self]


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


def _collect_documents():
    """Return a callback and list that collect all RunEngine documents."""
    documents = []

    def callback(name, doc):
        documents.append((name, dict(doc)))

    return callback, documents


def _events_by_stream(documents):
    """Group event documents by descriptor stream name."""
    descriptors = {doc["uid"]: doc["name"] for name, doc in documents if name == "descriptor"}
    events = {}
    for name, doc in documents:
        if name == "event":
            events.setdefault(descriptors[doc["descriptor"]], []).append(doc)
    return events


def _as_list(value):
    return list(value) if hasattr(value, "__iter__") and not isinstance(value, str) else [value]


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
    assert "acquisition_uid" in data
    assert "x1" in data
    assert "objective" in data
    assert data["x1"] == 0.0
    assert data["objective"] == 0.0
    assert data["acquisition_uid"] and isinstance(data["acquisition_uid"], str)


def test_optimization_failure(RE):
    class Alpha(Optimizer, TrialFaultAware): ...

    suggestion = [{"x1": 0.0, "_id": 0}]
    optimizer = MagicMock(spec=Alpha)
    optimizer.suggest.return_value = suggestion
    evaluation_function = MagicMock(spec=EvaluationFunction, return_value=[{"objective": 0.0, "_id": 0}])
    aquisition_function = MagicMock(spec=AcquisitionPlan, side_effect=RuntimeError())
    optimization_problem = OptimizationProblem(
        optimizer=optimizer,
        actuators=[MovableSignal("x1", initial_value=-1.0)],
        sensors=[ReadableSignal("objective")],
        evaluation_function=evaluation_function,
        acquisition_plan=aquisition_function,
    )

    callback, events = _collect_optimize_events()
    RE.subscribe(callback)
    try:
        RE(optimize(optimization_problem))
    except RuntimeError:
        ...
    finally:
        RE.unsubscribe(callback)

    optimizer.register_failures.assert_called_once_with(suggestion)
    assert evaluation_function.call_count == 0


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
        assert "acquisition_uid" in data
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
        assert "acquisition_uid" in data
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
        assert "acquisition_uid" in data
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
        assert data["acquisition_uid"] in uids


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
    with pytest.raises(ValueError, match="optimizer is not checkpointable"):
        RE(optimize(optimization_problem, iterations=5, n_points=2, checkpoint_interval=1))


def test_optimize_in_run_defaults_to_ordered_suggestion_ids(RE):
    optimizer = MagicMock(spec=Optimizer)
    optimizer.suggest.return_value = [{"x1": 10.0, "_id": "far"}, {"x1": 1.0, "_id": "near"}]
    movable = MovableSignal("x1", initial_value=0.0)
    readable = ReadableSignal("objective")
    captured = []

    callback, documents = _collect_documents()
    RE.subscribe(callback)

    def evaluation_function(uid: tuple[str, ...], suggestions: list[dict]) -> list[dict]:
        captured.append((uid, suggestions))
        events_by_stream = _events_by_stream(documents)
        primary_values = [event["data"]["x1"] for event in events_by_stream["primary"]]
        assert uid == ("near", "far")
        assert suggestions == [{"x1": 10.0, "_id": "far"}, {"x1": 1.0, "_id": "near"}]
        assert primary_values == [1.0, 10.0]
        return [{"objective": float(index), "_id": suggestion_id} for index, suggestion_id in enumerate(uid)]

    optimization_problem = OptimizationProblem(
        optimizer=optimizer,
        actuators=[movable],
        sensors=[readable],
        evaluation_function=evaluation_function,
    )

    try:
        RE(optimize_in_run(optimization_problem, n_points=2))
    finally:
        RE.unsubscribe(callback)

    assert captured == [(("near", "far"), [{"x1": 10.0, "_id": "far"}, {"x1": 1.0, "_id": "near"}])]
    optimizer.ingest.assert_called_once_with([{"objective": 0.0, "_id": "near"}, {"objective": 1.0, "_id": "far"}])
    start_docs = [doc for name, doc in documents if name == "start"]
    assert len(start_docs) == 1
    assert start_docs[0]["run_key"] == "optimize_in_run"
    events_by_stream = _events_by_stream(documents)
    assert len(events_by_stream["primary"]) == 2
    assert len(events_by_stream["optimization"]) == 1
    optimization_data = events_by_stream["optimization"][0]["data"]
    assert _as_list(optimization_data["suggestion_ids"]) == ["far", "near"]
    assert _as_list(optimization_data["acquisition_uid"]) == ["near", "far"]


def test_list_scan_in_run_allows_custom_per_step_streams(RE):
    def per_step(detectors, step, pos_cache):
        yield from bps.one_nd_step(detectors, step, pos_cache)
        yield from bps.trigger_and_read(detectors, name="monitor")
        yield from bps.trigger_and_read(detectors, name="monitor")

    optimizer = MagicMock(spec=Optimizer)
    optimizer.suggest.return_value = [{"x1": 0.0, "_id": 0}, {"x1": 1.0, "_id": 1}]
    readable = ReadableSignal("objective")

    def evaluation_function(uid: tuple[int, ...], suggestions: list[dict]) -> list[dict]:
        assert uid == (0, 1)
        assert suggestions == [{"x1": 0.0, "_id": 0}, {"x1": 1.0, "_id": 1}]
        return [{"objective": 0.0, "_id": 0}, {"objective": 1.0, "_id": 1}]

    optimization_problem = OptimizationProblem(
        optimizer=optimizer,
        actuators=[MovableSignal("x1", initial_value=-1.0)],
        sensors=[readable],
        evaluation_function=evaluation_function,
        acquisition_plan=list_scan_in_run,
    )
    callback, documents = _collect_documents()
    RE.subscribe(callback)
    try:
        RE(optimize_in_run(optimization_problem, n_points=2, per_step=per_step))
    finally:
        RE.unsubscribe(callback)

    optimizer.ingest.assert_called_once_with([{"objective": 0.0, "_id": 0}, {"objective": 1.0, "_id": 1}])
    events_by_stream = _events_by_stream(documents)
    assert len(events_by_stream["primary"]) == 2
    assert len(events_by_stream["monitor"]) == 4
    assert len(events_by_stream["optimization"]) == 1


def test_list_scan_in_run_preserves_stage_lifecycle(RE):
    optimizer = MagicMock(spec=Optimizer)
    optimizer.suggest.return_value = [{"x1": 0.0, "_id": 0}]
    readable = StageableReadable("objective")
    optimization_problem = OptimizationProblem(
        optimizer=optimizer,
        actuators=[MovableSignal("x1", initial_value=-1.0)],
        sensors=[readable],
        evaluation_function=MagicMock(return_value=[{"objective": 0.0, "_id": 0}]),
    )

    RE(optimize_in_run(optimization_problem))

    assert readable.stage_count == 1
    assert readable.unstage_count == 1


def test_optimize_in_run_rejects_child_run_control(RE):
    @plan
    def acquisition_plan(suggestions, actuators, sensors, *args, **kwargs):
        yield from bps.open_run(md={"run_key": "child"})
        yield from bps.trigger_and_read(sensors or [])
        yield from bps.close_run()
        return "child-run"

    optimizer = MagicMock(spec=Optimizer)
    optimizer.suggest.return_value = [{"x1": 0.0, "_id": 0}]
    evaluation_function = MagicMock(return_value=[{"objective": 0.0, "_id": 0}])
    optimization_problem = OptimizationProblem(
        optimizer=optimizer,
        actuators=[MovableSignal("x1", initial_value=-1.0)],
        sensors=[ReadableSignal("objective")],
        evaluation_function=evaluation_function,
        acquisition_plan=acquisition_plan,
    )

    with pytest.raises(ValueError, match="must not issue 'open_run'"):
        RE(optimize_in_run(optimization_problem))

    evaluation_function.assert_not_called()
    optimizer.ingest.assert_not_called()


def test_optimize_in_run_rejects_unhashable_suggestion_ids(RE):
    optimizer = MagicMock(spec=Optimizer)
    optimizer.suggest.return_value = [{"x1": 0.0, "_id": ["not-hashable"]}]
    evaluation_function = MagicMock(return_value=[{"objective": 0.0, "_id": ["not-hashable"]}])
    optimization_problem = OptimizationProblem(
        optimizer=optimizer,
        actuators=[MovableSignal("x1", initial_value=-1.0)],
        sensors=[ReadableSignal("objective")],
        evaluation_function=evaluation_function,
    )

    with pytest.raises(TypeError, match="hashable '_id'"):
        RE(optimize_in_run(optimization_problem))

    evaluation_function.assert_not_called()
    optimizer.ingest.assert_not_called()


def test_optimize_in_run_rejects_duplicate_suggestion_ids(RE):
    optimizer = MagicMock(spec=Optimizer)
    optimizer.suggest.return_value = [{"x1": 0.0, "_id": "same"}, {"x1": 1.0, "_id": "same"}]
    evaluation_function = MagicMock(return_value=[{"objective": 0.0, "_id": "same"}])
    optimization_problem = OptimizationProblem(
        optimizer=optimizer,
        actuators=[MovableSignal("x1", initial_value=-1.0)],
        sensors=[ReadableSignal("objective")],
        evaluation_function=evaluation_function,
    )

    with pytest.raises(ValueError, match="unique '_id'"):
        RE(optimize_in_run(optimization_problem, n_points=2))

    evaluation_function.assert_not_called()
    optimizer.ingest.assert_not_called()


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
    assert "acquisition_uid" in data
    assert "x1" in data
    assert "objective" in data

    # Validate predictable values from custom acquisition plan
    assert data["acquisition_uid"] == "test-uid-123"
    assert data["x1"] == 0.5
    assert data["objective"] == 1.25
    assert data["suggestion_ids"] == "0"


def test_optimize_with_tuple_acquisition_identifier(RE):
    """Pass tuple acquisition identifiers unchanged to evaluation and event data."""
    suggestion = {"x1": 0.5, "_id": 0}
    outcome = {"objective": 1.25, "_id": 0}
    optimizer = MagicMock(spec=Optimizer)
    optimizer.suggest.return_value = [suggestion]
    evaluation_function = MagicMock(spec=EvaluationFunction, return_value=[outcome])
    optimization_problem = OptimizationProblem(
        optimizer=optimizer,
        actuators=[MovableSignal("x1", initial_value=-1.0)],
        sensors=[ReadableSignal("objective")],
        evaluation_function=evaluation_function,
        acquisition_plan=_tuple_acquisition_plan,
    )

    callback, events = _collect_optimize_events()
    RE.subscribe(callback)
    try:
        RE(optimize(optimization_problem))
    finally:
        RE.unsubscribe(callback)

    acquisition_identifier = ("event-a", "event-b")
    evaluation_function.assert_called_once_with(acquisition_identifier, [suggestion])
    assert len(events) == 1
    assert events[0]["data"]["acquisition_uid"] == acquisition_identifier


def test_optimize_with_custom_hashable_acquisition_identifier_uses_repr(RE):
    """Store repr for hashable acquisition identifiers that are not native array-like values."""
    suggestion = {"x1": 0.5, "_id": 0}
    outcome = {"objective": 1.25, "_id": 0}
    optimizer = MagicMock(spec=Optimizer)
    optimizer.suggest.return_value = [suggestion]
    evaluation_function = MagicMock(spec=EvaluationFunction, return_value=[outcome])
    optimization_problem = OptimizationProblem(
        optimizer=optimizer,
        actuators=[MovableSignal("x1", initial_value=-1.0)],
        sensors=[ReadableSignal("objective")],
        evaluation_function=evaluation_function,
        acquisition_plan=_custom_identifier_acquisition_plan,
    )

    callback, events = _collect_optimize_events()
    RE.subscribe(callback)
    try:
        RE(optimize(optimization_problem))
    finally:
        RE.unsubscribe(callback)

    evaluation_function.assert_called_once_with(_CUSTOM_ACQUISITION_IDENTIFIER, [suggestion])
    assert len(events) == 1
    assert events[0]["data"]["acquisition_uid"] == repr(_CUSTOM_ACQUISITION_IDENTIFIER)


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


def test_optimize_max_number_of_iterations_before_stop(RE):
    """Tests that the optimization stops at a set number of iterations"""

    class StoppingOptimizer(Optimizer, SupportsStoppingCriteria): ...

    optimizer = MagicMock(spec=StoppingOptimizer)
    optimizer.suggest.return_value = [{"x1": 0.0, "_id": 0}]

    # Set up the optimizer to stop after 2 iterations
    optimizer.should_stop.side_effect = [
        (False, None),
        (True, "converged"),
    ]
    evaluation_function = MagicMock(spec=EvaluationFunction, return_value=[{"objective": 0.0, "_id": 0}])
    optimization_problem = OptimizationProblem(
        optimizer=optimizer,
        actuators=[MovableSignal("x1")],
        sensors=[ReadableSignal("objective")],
        evaluation_function=evaluation_function,
    )

    RE(optimize(optimization_problem, iterations=5))

    assert optimizer.suggest.call_count == 2
    assert optimizer.should_stop.call_count == 2


def test_optimize_stop_condition_not_hit(RE):
    """Tests that optimization stops before stop condition is met"""

    class StoppingOptimizer(Optimizer, SupportsStoppingCriteria): ...

    optimizer = MagicMock(spec=StoppingOptimizer)
    optimizer.suggest.return_value = [{"x1": 0.0, "_id": 0}]

    # Allow for 3 iterations
    optimizer.should_stop.side_effect = [(False, None), (False, None), (False, None)]
    evaluation_function = MagicMock(spec=EvaluationFunction, return_value=[{"objective": 0.0, "_id": 0}])
    optimization_problem = OptimizationProblem(
        optimizer=optimizer,
        actuators=[MovableSignal("x1")],
        sensors=[ReadableSignal("objective")],
        evaluation_function=evaluation_function,
    )

    # We only are running for 2 iterations, so stop condition should not be met
    RE(optimize(optimization_problem, iterations=2))

    assert optimizer.suggest.call_count == 2
    assert optimizer.should_stop.call_count == 2


def test_optimize_stops_when_change_is_within_tolerance(RE):
    """Tests that the optimization stops when the change in objective value is within a specified tolerance."""

    class ToleranceStopOptimizer(Optimizer, SupportsStoppingCriteria):
        def __init__(self, tolerance: float):
            self.tolerance = tolerance
            self._last_value: float | None = None
            self._previous_value: float | None = None

        def suggest(self, num_points: int | None = None) -> list[dict]:
            return [{"x1": 0.0, "_id": 0}]

        def ingest(self, points: list[dict]) -> None:
            self._previous_value = self._last_value
            self._last_value = points[0]["objective"]

        def should_stop(self) -> tuple[bool, str | None]:
            if self._previous_value is None or self._last_value is None:
                return (False, None)

            if abs(self._last_value - self._previous_value) <= self.tolerance:
                return (True, "objective change within tolerance")

            return (False, None)

    # Stop optimization when the change is within 0.1
    optimizer = ToleranceStopOptimizer(tolerance=0.1)
    evaluation_function = MagicMock(
        spec=EvaluationFunction,
        side_effect=[
            [{"objective": 0.5, "_id": 0}],
            [{"objective": 0.55, "_id": 0}],
        ],
    )
    optimization_problem = OptimizationProblem(
        optimizer=optimizer,
        actuators=[MovableSignal("x1")],
        sensors=[ReadableSignal("objective")],
        evaluation_function=evaluation_function,
    )

    callback, events = _collect_optimize_events()
    RE.subscribe(callback)
    try:
        RE(optimize(optimization_problem, iterations=5))
    finally:
        RE.unsubscribe(callback)

    assert evaluation_function.call_count == 2
