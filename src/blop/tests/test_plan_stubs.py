from unittest.mock import MagicMock

import pytest
from bluesky.run_engine import RunEngine

from blop.plan_stubs import navigate_to_best
from blop.protocols import HasBestPoints, Optimizer

from .conftest import MovableSignal


@pytest.fixture()
def RE():
    return RunEngine({})


class OptimizerHasBestPoints(Optimizer, HasBestPoints): ...


def test_navigate_single_objective(RE):
    """navigate_to_best moves actuators to the single best point."""
    optimizer = MagicMock(spec=OptimizerHasBestPoints)
    optimizer.get_best_points.return_value = [(123, {"x1": 1.5, "x2": 2.5}, {"objective": 10.0})]
    x1 = MovableSignal("x1", initial_value=0.0)
    x2 = MovableSignal("x2", initial_value=0.0)

    RE(navigate_to_best([x1, x2], optimizer))

    assert x1._value == 1.5
    assert x2._value == 2.5


def test_navigate_explicit_parameterization(RE):
    """navigate_to_best with explicit parameterization skips optimizer query."""
    x1 = MovableSignal("x1", initial_value=0.0)

    RE(navigate_to_best([x1], parameterization={"x1": 3.0}))

    assert x1._value == 3.0


def test_navigate_raises_on_multiple_pareto_points(RE):
    """navigate_to_best raises ValueError when optimizer returns multiple Pareto points."""
    optimizer = MagicMock(spec=OptimizerHasBestPoints)
    optimizer.get_best_points.return_value = [
        (1, {"x1": 1.0}, {"obj_a": 5.0, "obj_b": 2.0}),
        (5, {"x1": 2.0}, {"obj_a": 3.0, "obj_b": 4.0}),
        (100, {"x1": 3.0}, {"obj_a": 1.0, "obj_b": 6.0}),
    ]
    x1 = MovableSignal("x1", initial_value=0.0)

    with pytest.raises(ValueError, match="3 Pareto-optimal points"):
        RE(navigate_to_best([x1], optimizer))


def test_navigate_raises_on_missing_protocol(RE):
    """navigate_to_best raises TypeError if optimizer lacks HasBestPoints."""
    optimizer = MagicMock(spec=Optimizer)
    x1 = MovableSignal("x1", initial_value=0.0)

    with pytest.raises(TypeError, match="does not implement the HasBestPoints protocol"):
        RE(navigate_to_best([x1], optimizer))


def test_navigate_ignores_unknown_params(RE):
    """Parameterization keys not matching actuator names are ignored."""
    optimizer = MagicMock(spec=OptimizerHasBestPoints)
    optimizer.get_best_points.return_value = [(1, {"x1": 5.0, "unknown_param": 99.0}, {"objective": 1.0})]
    x1 = MovableSignal("x1", initial_value=0.0)

    RE(navigate_to_best([x1], optimizer))

    assert x1._value == 5.0
