from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from bluesky.run_engine import RunEngine
from xopt.generators.bayesian import ExpectedImprovementGenerator
from xopt.generators.random import RandomGenerator
from xopt.vocs import VOCS

from blop.plans import optimize
from blop.protocols import OptimizationProblem
from blop.tests.conftest import MovableSignal, ReadableSignal
from blop.xopt import optimizer as xopt_optimizer_module
from blop.xopt.optimizer import XoptOptimizer


def _random_optimizer(vocs: VOCS, *, checkpoint_path: str | None = None) -> XoptOptimizer:
    return XoptOptimizer(
        generator=RandomGenerator(vocs=vocs),
        checkpoint_path=checkpoint_path,
    )


def _bo_optimizer(vocs: VOCS, *, checkpoint_path: str | None = None) -> XoptOptimizer:
    return XoptOptimizer(
        generator=ExpectedImprovementGenerator(vocs=vocs),
        checkpoint_path=checkpoint_path,
    )


@pytest.fixture(params=[_random_optimizer, _bo_optimizer], ids=["random", "bo"])
def optimizer_factory(request: pytest.FixtureRequest) -> Callable[[VOCS], XoptOptimizer]:
    return request.param


def test_xopt_optimizer_init(optimizer_factory: Callable[[VOCS], XoptOptimizer]):
    if optimizer_factory is _bo_optimizer:
        vocs = VOCS(
            variables={"x1": [-5.0, 5.0], "x2": [-5.0, 5.0], "x3": [0.0, 5.0]},
            objectives={"y1": "MINIMIZE"},
            constraints={"y1": ["LESS_THAN", 10.0]},
        )
    else:
        vocs = VOCS(
            variables={"x1": [-5.0, 5.0], "x2": [-5.0, 5.0], "x3": [0.0, 5.0]},
            objectives={"y1": "MAXIMIZE", "y2": "MINIMIZE"},
            constraints={"y1": ["GREATER_THAN", 0.0], "y2": ["LESS_THAN", 0.0]},
        )

    optimizer = optimizer_factory(vocs)
    assert optimizer.generator is not None
    assert set(optimizer.vocs.variable_names) == {"x1", "x2", "x3"}


def test_xopt_optimizer_suggest_ids_and_keys():
    vocs = VOCS(variables={"x1": [-5.0, 5.0], "x2": [-5.0, 5.0], "x3": [0.0, 5.0]}, objectives={"y1": "MINIMIZE"})
    optimizer = _random_optimizer(vocs)

    suggestions = optimizer.suggest(num_points=2)
    assert len(suggestions) == 2
    for i, suggestion in enumerate(suggestions):
        assert suggestion["_id"] == i
        assert "x1" in suggestion
        assert "x2" in suggestion
        assert "x3" in suggestion


def test_xopt_optimizer_suggest_defaults_to_single_point():
    vocs = VOCS(variables={"x": [0.0, 1.0]}, objectives={"y": "MINIMIZE"})
    optimizer = _random_optimizer(vocs)

    suggestions = optimizer.suggest()
    assert len(suggestions) == 1
    assert suggestions[0]["_id"] == 0


def test_xopt_optimizer_first_suggest_uses_vocs_random_inputs(monkeypatch: pytest.MonkeyPatch):
    vocs = VOCS(variables={"x": [0.0, 1.0]}, objectives={"y": "MINIMIZE"})
    optimizer = _bo_optimizer(vocs)

    def _random_inputs(_vocs: VOCS, n: int | None = None, **_kwargs) -> list[dict]:
        assert set(_vocs.variable_names) == {"x"}
        assert n == 1
        return [{"x": 0.42}]

    monkeypatch.setattr(xopt_optimizer_module, "random_inputs", _random_inputs)

    suggestions = optimizer.suggest()

    assert suggestions == [{"_id": 0, "x": 0.42}]


def test_xopt_optimizer_ingest_multiple_columns(optimizer_factory: Callable[[VOCS], XoptOptimizer]):
    if optimizer_factory is _bo_optimizer:
        vocs = VOCS(
            variables={"x1": [-5.0, 5.0], "x2": [-5.0, 5.0], "x3": [0.0, 5.0]},
            objectives={"y1": "MINIMIZE"},
        )
    else:
        vocs = VOCS(
            variables={"x1": [-5.0, 5.0], "x2": [-5.0, 5.0], "x3": [0.0, 5.0]},
            objectives={"y1": "MAXIMIZE", "y2": "MINIMIZE"},
        )
    optimizer = optimizer_factory(vocs)

    if optimizer_factory is _bo_optimizer:
        optimizer.ingest(
            [
                {"x1": 0.0, "x2": 0.0, "x3": 0.0, "y1": 1.0},
                {"x1": 0.1, "x2": 0.2, "x3": 1.0, "y1": 3.0},
            ]
        )
    else:
        optimizer.ingest(
            [
                {"x1": 0.0, "x2": 0.0, "x3": 0.0, "y1": 1.0, "y2": 2.0},
                {"x1": 0.1, "x2": 0.2, "x3": 1.0, "y1": 3.0, "y2": 4.0},
            ]
        )

    data = optimizer.generator.data
    assert data is not None
    assert len(data) == 2
    assert np.allclose(data["x1"].to_numpy(dtype=float), [0.0, 0.1])
    assert np.allclose(data["x2"].to_numpy(dtype=float), [0.0, 0.2])
    assert np.allclose(data["x3"].to_numpy(dtype=float), [0.0, 1.0])
    assert np.allclose(data["y1"].to_numpy(dtype=float), [1.0, 3.0])
    if optimizer_factory is _bo_optimizer:
        assert "y2" not in data.columns
    else:
        assert np.allclose(data["y2"].to_numpy(dtype=float), [2.0, 4.0])


def test_xopt_optimizer_ingest_baseline_id(optimizer_factory: Callable[[VOCS], XoptOptimizer]):
    vocs = VOCS(variables={"x1": [-5.0, 5.0]}, objectives={"y1": "MINIMIZE"})
    optimizer = optimizer_factory(vocs)

    optimizer.ingest([{"x1": 0.0, "y1": 1.0, "_id": "baseline"}])
    data = optimizer.generator.data
    assert data is not None
    assert len(data) == 1
    assert data.iloc[0]["_id"] == "baseline"


def test_xopt_optimizer_seeds_state_when_existing_data_lacks_id():
    vocs = VOCS(variables={"x": [0.0, 1.0]}, objectives={"y": "MINIMIZE"})
    generator = RandomGenerator(vocs=vocs)
    generator.add_data(pd.DataFrame([{"x": 0.4, "y": 1.2}]))

    optimizer = XoptOptimizer(generator=generator)

    assert optimizer._params_by_id[0] == {"x": 0.4}
    assert optimizer._next_id == 1


def test_xopt_optimizer_suggest_ingest():
    vocs = VOCS(variables={"x1": [-5.0, 5.0], "x2": [-5.0, 5.0]}, objectives={"y1": "MINIMIZE", "y2": "MINIMIZE"})
    optimizer = _random_optimizer(vocs)

    suggestions = optimizer.suggest(num_points=2)
    outcomes = [
        {"_id": suggestions[0]["_id"], "y1": 1.0, "y2": 2.0},
        {"_id": suggestions[1]["_id"], "y1": 3.0, "y2": 4.0},
    ]
    optimizer.ingest(outcomes)

    data = optimizer.generator.data
    assert data is not None
    assert len(data) == 2
    assert np.allclose(data["y1"].to_numpy(dtype=float), [1.0, 3.0])
    assert np.allclose(data["y2"].to_numpy(dtype=float), [2.0, 4.0])


def test_xopt_optimizer_register_failures():
    vocs = VOCS(variables={"x1": [-5.0, 5.0], "x2": [-5.0, 5.0]}, objectives={"y1": "MINIMIZE"})
    optimizer = _random_optimizer(vocs)

    suggestions = optimizer.suggest(num_points=5)
    optimizer.register_failures(suggestions)

    assert all(suggestion["_id"] not in optimizer._params_by_id for suggestion in suggestions)


def test_xopt_optimizer_checkpoint_roundtrip(tmp_path):
    vocs = VOCS(variables={"x": [0.0, 1.0]}, objectives={"y": "MINIMIZE"})
    checkpoint_path = tmp_path / "xopt_optimizer.pkl"
    optimizer = _random_optimizer(vocs, checkpoint_path=str(checkpoint_path))

    suggestions = optimizer.suggest(1)
    optimizer.ingest([{"_id": suggestions[0]["_id"], "y": 0.5}])
    optimizer.checkpoint()

    recovered = XoptOptimizer.from_checkpoint(str(checkpoint_path))
    assert recovered.generator.data is not None
    assert len(recovered.generator.data) == 1
    assert recovered.checkpoint_path == str(checkpoint_path)


def test_xopt_optimizer_checkpoint_no_path():
    vocs = VOCS(variables={"x1": [-5.0, 5.0]}, objectives={"y1": "MINIMIZE"})
    optimizer = _random_optimizer(vocs)

    with pytest.raises(ValueError):
        optimizer.checkpoint()


def test_xopt_optimizer_get_best_points_single_objective_minimize(optimizer_factory: Callable[[VOCS], XoptOptimizer]):
    vocs = VOCS(variables={"x": [0.0, 1.0]}, objectives={"y": "MINIMIZE"})
    optimizer = optimizer_factory(vocs)

    optimizer.ingest(
        [
            {"x": 0.1, "y": 5.0},
            {"x": 0.2, "y": 1.0},
            {"x": 0.3, "y": 3.0},
        ]
    )

    best_points = optimizer.get_best_points()
    assert len(best_points) == 1
    _, params, outcomes = best_points[0]
    assert params["x"] == 0.2
    assert outcomes["y"] == 1.0


def test_xopt_optimizer_get_best_points_single_objective_maximize(optimizer_factory: Callable[[VOCS], XoptOptimizer]):
    vocs = VOCS(variables={"x": [0.0, 1.0]}, objectives={"y": "MAXIMIZE"})
    optimizer = optimizer_factory(vocs)

    optimizer.ingest(
        [
            {"x": 0.1, "y": 5.0},
            {"x": 0.2, "y": 1.0},
            {"x": 0.3, "y": 3.0},
        ]
    )

    best_points = optimizer.get_best_points()
    assert len(best_points) == 1
    _, params, outcomes = best_points[0]
    assert params["x"] == 0.1
    assert outcomes["y"] == 5.0


def test_xopt_optimizer_get_best_points_multi_objective():
    vocs = VOCS(variables={"x": [0.0, 10.0]}, objectives={"y1": "MAXIMIZE", "y2": "MAXIMIZE"})
    optimizer = _random_optimizer(vocs)

    optimizer.ingest(
        [
            {"x": 1.0, "y1": 10.0, "y2": 1.0},
            {"x": 5.0, "y1": 1.0, "y2": 10.0},
            {"x": 3.0, "y1": 2.0, "y2": 2.0},
        ]
    )

    best_points = optimizer.get_best_points()
    assert len(best_points) == 3
    for trial_id, params, metrics in best_points:
        assert isinstance(trial_id, (int, float, str))
        assert "x" in params
        assert "y1" in metrics
        assert "y2" in metrics


def test_xopt_optimizer_get_best_points_returns_empty_when_all_infeasible():
    vocs = VOCS(
        variables={"x": [0.0, 1.0]},
        objectives={"y": "MINIMIZE"},
        constraints={"c": ["LESS_THAN", 0.0]},
    )
    optimizer = _random_optimizer(vocs)

    optimizer.ingest(
        [
            {"x": 0.1, "y": 5.0, "c": 1.0},
            {"x": 0.2, "y": 1.0, "c": 2.0},
            {"x": 0.3, "y": 3.0, "c": 3.0},
        ]
    )

    best_points = optimizer.get_best_points()
    assert best_points == []


def test_xopt_optimizer_get_best_points_selects_best_feasible_only():
    vocs = VOCS(
        variables={"x": [0.0, 1.0]},
        objectives={"y": "MINIMIZE"},
        constraints={"c": ["LESS_THAN", 0.5]},
    )
    optimizer = _random_optimizer(vocs)

    optimizer.ingest(
        [
            {"x": 0.1, "y": 10.0, "c": 0.1},
            {"x": 0.2, "y": 1.0, "c": 0.9},
            {"x": 0.3, "y": 2.0, "c": 0.2},
        ]
    )

    best_points = optimizer.get_best_points()
    assert len(best_points) == 1
    _, params, outcomes = best_points[0]
    assert params["x"] == 0.3
    assert outcomes["y"] == 2.0
    assert outcomes["c"] == 0.2


def test_xopt_optimizer_get_best_points_empty_without_data():
    vocs = VOCS(variables={"x": [0.0, 1.0]}, objectives={"y": "MINIMIZE"})
    optimizer = _random_optimizer(vocs)

    assert optimizer.get_best_points() == []


def test_xopt_optimizer_get_best_points_includes_observables():
    vocs = VOCS(
        variables={"x": [0.0, 1.0]},
        objectives={"y": "MINIMIZE"},
        observables=["obs"],
    )
    optimizer = _random_optimizer(vocs)

    optimizer.ingest(
        [
            {"x": 0.1, "y": 2.0, "obs": 10.0},
            {"x": 0.2, "y": 1.0, "obs": 20.0},
        ]
    )

    best_points = optimizer.get_best_points()
    assert len(best_points) == 1
    _, _, outcomes = best_points[0]
    assert outcomes["y"] == 1.0
    assert outcomes["obs"] == 20.0


def test_xopt_expected_improvement_runs_simple_minimization():
    vocs = VOCS(variables={"x": [0.0, 1.0]}, objectives={"y": "MINIMIZE"})
    optimizer = XoptOptimizer(generator=ExpectedImprovementGenerator(vocs=vocs))

    # Seed EI with initial evaluations for model training.
    optimizer.ingest(
        [
            {"x": 0.0, "y": (0.0 - 0.25) ** 2},
            {"x": 0.5, "y": (0.5 - 0.25) ** 2},
            {"x": 1.0, "y": (1.0 - 0.25) ** 2},
        ]
    )

    for _ in range(3):
        suggestion = optimizer.suggest(1)[0]
        x_val = float(suggestion["x"])
        optimizer.ingest([{"_id": suggestion["_id"], "y": (x_val - 0.25) ** 2}])

    assert optimizer.generator.data is not None
    assert len(optimizer.generator.data) == 6

    best_points = optimizer.get_best_points()
    assert len(best_points) == 1
    _, _, outcomes = best_points[0]
    assert outcomes["y"] <= 0.0625


def test_xopt_inside_run_engine():
    vocs = VOCS(variables={"x": [0.0, 1.0]}, objectives={"y": "MINIMIZE"})
    optimizer = XoptOptimizer(generator=ExpectedImprovementGenerator(vocs=vocs))

    actuator = MovableSignal("x", initial_value=0.5)
    sensor = ReadableSignal("y")

    def evaluation_function(_uid: str, suggestions: list[dict]) -> list[dict]:
        return [
            {
                "_id": suggestion["_id"],
                "y": (float(suggestion["x"]) - 0.25) ** 2,
            }
            for suggestion in suggestions
        ]

    optimization_problem = OptimizationProblem(
        optimizer=optimizer,
        actuators=[actuator],
        sensors=[sensor],
        evaluation_function=evaluation_function,
    )

    RE = RunEngine({})
    RE(optimize(optimization_problem, iterations=4, n_points=1))

    data = optimizer.generator.data
    assert data is not None
    assert len(data) == 4
    assert "x" in data.columns
    assert "y" in data.columns

    best_points = optimizer.get_best_points()
    assert len(best_points) == 1
    _, params, outcomes = best_points[0]
    assert 0.0 <= float(params["x"]) <= 1.0
    assert float(outcomes["y"]) >= 0.0
