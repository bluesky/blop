from pathlib import PurePath
from unittest.mock import MagicMock

import bluesky.plan_stubs as bps
import numpy as np
import pytest
from bluesky.run_engine import RunEngine
from bluesky.utils import plan

pytest.importorskip("tiled")
pytest.importorskip("bluesky_tiled_plugins")

from blop_sim.backends.simple import SimpleBackend
from blop_sim.devices.detector import DetectorDevice
from ophyd_async.core import StaticPathProvider, UUIDFilenameProvider
from tiled.client import from_uri
from tiled.server import SimpleTiledServer

from blop.plans import optimize_tiled_stream
from blop.protocols import EvaluationFunction, OptimizationProblem, Optimizer, RunDataReference

from .conftest import MovableSignal, ReadableSignal


@pytest.fixture(scope="function")
def RE():
    return RunEngine({})


@pytest.fixture(scope="function")
def tiled_client():
    server = SimpleTiledServer()
    try:
        yield from_uri(server.uri)
    finally:
        _close_tiled_server(server)


def _materialize(data):
    if hasattr(data, "compute"):
        return data.compute()
    return data


def _close_tiled_server(server):
    threaded_server = server._cm.gen.gi_frame.f_locals.get("self")
    if threaded_server is not None:
        threaded_server.force_exit = True
    server.close()


def test_optimize_tiled_stream_reads_primary_table_before_stop(RE, tiled_client):
    optimizer = MagicMock(spec=Optimizer)
    optimizer.suggest.return_value = [{"x1": 0.0, "_id": 0}]
    uid_evaluator = MagicMock(spec=EvaluationFunction)
    optimization_problem = OptimizationProblem(
        optimizer=optimizer,
        actuators=[MovableSignal("x1", initial_value=-1.0)],
        sensors=[ReadableSignal("objective")],
        evaluation_function=uid_evaluator,
    )

    def evaluation_function(reference: RunDataReference, suggestions: list[dict]) -> list[dict]:
        assert reference.source == "tiled"
        assert reference.resolver is not None
        assert reference.stream_slices == {"primary": slice(0, 1)}
        objective = reference.resolver.read("primary/objective", stream_slice=reference.stream_slices["primary"])
        assert len(objective) == 1
        return [{"objective": 0.0, "_id": suggestions[0]["_id"]}]

    uids = RE(
        optimize_tiled_stream(optimization_problem, evaluation_function, tiled_client, timeout=5.0, poll_interval=0.01)
    )

    uid_evaluator.assert_not_called()
    optimizer.ingest.assert_called_once_with([{"objective": 0.0, "_id": 0}])
    table = _materialize(tiled_client[uids[0]]["optimization/internal"].read())
    assert len(table) == 1
    row = table.iloc[0]
    assert row["suggestion_ids"] == "0"
    assert row["bluesky_uid"] == uids[0]
    assert row["x1"] == 0.0
    assert row["objective"] == 0.0


def test_optimize_tiled_stream_uses_configured_primary_stream(RE, tiled_client):
    @plan
    def acquisition_plan(suggestions, actuators, sensors, *args, **kwargs):
        sensors_by_name = {sensor.name: sensor for sensor in sensors or []}
        yield from bps.trigger_and_read([sensors_by_name["objective"]], name="primary")
        yield from bps.trigger_and_read([sensors_by_name["monitor"]], name="monitor")
        yield from bps.trigger_and_read([sensors_by_name["objective"]], name="primary")
        yield from bps.trigger_and_read([sensors_by_name["monitor"]], name="monitor")
        yield from bps.trigger_and_read([sensors_by_name["objective"]], name="primary")

    optimizer = MagicMock(spec=Optimizer)
    optimizer.suggest.return_value = [{"x1": 0.0, "_id": 0}, {"x1": 0.0, "_id": 1}]
    uid_evaluator = MagicMock(spec=EvaluationFunction)
    optimization_problem = OptimizationProblem(
        optimizer=optimizer,
        actuators=[MovableSignal("x1", initial_value=-1.0)],
        sensors=[ReadableSignal("objective"), ReadableSignal("monitor")],
        evaluation_function=uid_evaluator,
        acquisition_plan=acquisition_plan,
    )
    evaluation_function = MagicMock()

    def evaluate(reference: RunDataReference, suggestions: list[dict]) -> list[dict]:
        assert reference.stream_slices == {"monitor": slice(0, 2)}
        monitor = reference.resolver.read("monitor/monitor", stream_slice=slice(0, 2))
        assert len(monitor) == 2
        return [{"objective": 0.0, "_id": suggestion["_id"]} for suggestion in suggestions]

    evaluation_function.side_effect = evaluate

    uids = RE(
        optimize_tiled_stream(
            optimization_problem,
            evaluation_function,
            tiled_client,
            n_points=2,
            primary_stream="monitor",
            timeout=5.0,
            poll_interval=0.01,
        )
    )

    run = tiled_client[uids[0]]
    assert len(_materialize(run["primary/internal"].read())) == 3
    assert len(_materialize(run["monitor/internal"].read())) == 2
    assert len(_materialize(run["optimization/internal"].read())) == 1
    uid_evaluator.assert_not_called()
    evaluation_function.assert_called_once()
    optimizer.ingest.assert_called_once()


def test_optimize_tiled_stream_times_out_when_primary_stream_short(RE, tiled_client):
    @plan
    def acquisition_plan(suggestions, actuators, sensors, *args, **kwargs):
        yield from bps.trigger_and_read(sensors or [], name="primary")

    optimizer = MagicMock(spec=Optimizer)
    optimizer.suggest.return_value = [{"x1": 0.0, "_id": 0}, {"x1": 0.0, "_id": 1}]
    evaluation_function = MagicMock(return_value=[{"objective": 0.0, "_id": 0}, {"objective": 0.0, "_id": 1}])
    optimization_problem = OptimizationProblem(
        optimizer=optimizer,
        actuators=[MovableSignal("x1", initial_value=-1.0)],
        sensors=[ReadableSignal("objective")],
        evaluation_function=MagicMock(spec=EvaluationFunction),
        acquisition_plan=acquisition_plan,
    )

    with pytest.raises(TimeoutError, match="did not reach 2 rows"):
        RE(
            optimize_tiled_stream(
                optimization_problem,
                evaluation_function,
                tiled_client,
                n_points=2,
                timeout=0.5,
                poll_interval=0.01,
            )
        )

    evaluation_function.assert_not_called()
    optimizer.ingest.assert_not_called()


def test_optimize_tiled_stream_reads_external_detector_image(RE, tmp_path):
    server = SimpleTiledServer(readable_storage=[tmp_path])
    try:
        tiled_client = from_uri(server.uri)
        optimizer = MagicMock(spec=Optimizer)
        optimizer.suggest.return_value = [{"x1": 0.0, "_id": 0}]
        det = DetectorDevice(
            SimpleBackend(noise=False),
            StaticPathProvider(UUIDFilenameProvider(), PurePath(tmp_path)),
            name="det",
        )

        @plan
        def acquisition_plan(suggestions, actuators, sensors, *args, **kwargs):
            yield from bps.trigger_and_read([det], name="primary")

        optimization_problem = OptimizationProblem(
            optimizer=optimizer,
            actuators=[MovableSignal("x1", initial_value=-1.0)],
            sensors=[det],
            evaluation_function=MagicMock(spec=EvaluationFunction),
            acquisition_plan=acquisition_plan,
        )

        def evaluation_function(reference: RunDataReference, suggestions: list[dict]) -> list[dict]:
            images = reference.resolver.read("primary/det_image", stream_slice=reference.stream_slices["primary"])
            assert images.shape[0] == 1
            return [{"intensity": float(images.sum()), "_id": suggestions[0]["_id"]}]

        RE(
            optimize_tiled_stream(
                optimization_problem,
                evaluation_function,
                tiled_client,
                timeout=5.0,
                poll_interval=0.01,
            )
        )
    finally:
        _close_tiled_server(server)

    outcomes = optimizer.ingest.call_args.args[0]
    assert len(outcomes) == 1
    assert outcomes[0]["_id"] == 0
    assert np.isfinite(outcomes[0]["intensity"])
    assert outcomes[0]["intensity"] > 0


def test_optimize_tiled_stream_rejects_stream_collision(RE, tiled_client):
    optimizer = MagicMock(spec=Optimizer)
    optimizer.suggest.return_value = [{"x1": 0.0, "_id": 0}]
    optimization_problem = OptimizationProblem(
        optimizer=optimizer,
        actuators=[MovableSignal("x1", initial_value=-1.0)],
        sensors=[ReadableSignal("objective")],
        evaluation_function=MagicMock(spec=EvaluationFunction),
    )

    with pytest.raises(
        ValueError,
        match=(
            "primary_stream and optimization_stream must be different. Got "
            "primary_stream='optimization', optimization_stream='optimization'."
        ),
    ):
        RE(
            optimize_tiled_stream(
                optimization_problem,
                MagicMock(),
                tiled_client,
                primary_stream="optimization",
                optimization_stream="optimization",
            )
        )

    optimizer.suggest.assert_not_called()
