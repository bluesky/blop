import pytest

xopt = pytest.importorskip("xopt")

from xopt.generators.random import RandomGenerator
from xopt.vocs import VOCS

from blop.xopt.optimizer import XoptOptimizer


def test_xopt_optimizer_suggest_and_ingest():
    vocs = VOCS(variables={"x": [0.0, 1.0]}, objectives={"y": "MINIMIZE"})
    optimizer = XoptOptimizer(generator=RandomGenerator, vocs=vocs)

    suggestions = optimizer.suggest(2)
    assert len(suggestions) == 2
    assert all("_id" in suggestion for suggestion in suggestions)

    outcomes = [{"_id": suggestion["_id"], "y": float(i)} for i, suggestion in enumerate(suggestions)]
    optimizer.ingest(outcomes)

    assert optimizer.generator.data is not None
    assert len(optimizer.generator.data) == 2


def test_xopt_optimizer_get_best_points_single_objective_minimize():
    vocs = VOCS(variables={"x": [0.0, 1.0]}, objectives={"y": "MINIMIZE"})
    optimizer = XoptOptimizer(generator=RandomGenerator, vocs=vocs)

    optimizer.ingest([
        {"x": 0.1, "y": 5.0},
        {"x": 0.2, "y": 1.0},
        {"x": 0.3, "y": 3.0},
    ])

    best_points = optimizer.get_best_points()
    assert len(best_points) == 1
    _, params, outcomes = best_points[0]
    assert params["x"] == 0.2
    assert outcomes["y"] == 1.0


def test_xopt_optimizer_checkpoint_roundtrip(tmp_path):
    vocs = VOCS(variables={"x": [0.0, 1.0]}, objectives={"y": "MINIMIZE"})
    checkpoint_path = tmp_path / "xopt_optimizer.pkl"
    optimizer = XoptOptimizer(generator=RandomGenerator, vocs=vocs, checkpoint_path=str(checkpoint_path))

    suggestions = optimizer.suggest(1)
    optimizer.ingest([{"_id": suggestions[0]["_id"], "y": 0.5}])
    optimizer.checkpoint()

    recovered = XoptOptimizer.from_checkpoint(str(checkpoint_path))
    assert recovered.generator.data is not None
    assert len(recovered.generator.data) == 1


def test_xopt_optimizer_applies_fixed_parameters():
    vocs = VOCS(variables={"x": [0.0, 1.0], "z": [0.0, 2.0]}, objectives={"y": "MINIMIZE"})
    optimizer = XoptOptimizer(generator=RandomGenerator, vocs=vocs)
    optimizer.fixed_parameters = {"z": 1.25}

    suggestions = optimizer.suggest(3)
    assert all(suggestion["z"] == 1.25 for suggestion in suggestions)
