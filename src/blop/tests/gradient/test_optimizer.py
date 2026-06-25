import time
from unittest.mock import MagicMock

import pytest

from blop.ax import Objective, RangeDOF
from blop.gradient import SCP, ScipyCFG, ScipyOptimizer
from blop.protocols import ID_KEY, AcquisitionPlan, EvaluationFunction

from ..conftest import MovableSignal


@pytest.fixture(scope="function")
def mock_evaluation_function():
    return MagicMock(spec=EvaluationFunction)


@pytest.fixture(scope="function")
def mock_acquisition_plan():
    return MagicMock(spec=AcquisitionPlan)


@pytest.fixture(scope="function")
def optimizer_prep():
    movable1 = MovableSignal(name="test_movable1")
    movable2 = MovableSignal(name="test_movable2")
    dof1 = RangeDOF(actuator=movable1, bounds=(0, 10), parameter_type="float")
    dof2 = RangeDOF(actuator=movable2, bounds=(0, 10), parameter_type="float")
    objective = Objective(name="test_objective", minimize=False)
    config = ScipyCFG(
        dofs=[dof1, dof2],
        objective=objective,
        threads=4,
        rescale=[2.0, 3.0],
    )
    return ScipyOptimizer(config, timeout=5)


# ============================================================================
# PHASE 2: Optimizer Algorithm Variations Tests
# ============================================================================


@pytest.mark.parametrize("optimizer", [SCP.Default, SCP.BFGS, SCP.LBFGS, SCP.Dual_Annealing])
def test_scipy_optimizer_algorithms(mock_evaluation_function, mock_acquisition_plan, optimizer):
    """Test ScipyOptimizer with different SCP algorithms."""
    movable1 = MovableSignal(name="test_movable1")
    movable2 = MovableSignal(name="test_movable2")
    dof1 = RangeDOF(actuator=movable1, bounds=(0, 10), parameter_type="float")
    dof2 = RangeDOF(actuator=movable2, bounds=(0, 10), parameter_type="float")
    objective = Objective(name="test_objective", minimize=False)

    config = ScipyCFG(
        dofs=[dof1, dof2],
        objective=objective,
        optimizer=optimizer,
        max_iter=10,
    )

    opt = ScipyOptimizer(config, timeout=5)
    assert opt._active is not None
    opt.close()


def test_scipy_optimizer_bfgs_specific(mock_evaluation_function, mock_acquisition_plan):
    """Test ScipyOptimizer explicitly with BFGS."""
    movable1 = MovableSignal(name="test_movable1")
    movable2 = MovableSignal(name="test_movable2")
    dof1 = RangeDOF(actuator=movable1, bounds=(0, 10), parameter_type="float")
    dof2 = RangeDOF(actuator=movable2, bounds=(0, 10), parameter_type="float")
    objective = Objective(name="test_objective", minimize=False)

    config = ScipyCFG(
        dofs=[dof1, dof2],
        objective=objective,
        optimizer=SCP.BFGS,
        max_iter=10,
    )

    opt = ScipyOptimizer(config, timeout=5)
    assert opt.final is None  # No optimization run yet
    opt.close()


def test_scipy_optimizer_dual_annealing_specific(mock_evaluation_function, mock_acquisition_plan):
    """Test ScipyOptimizer explicitly with Dual_Annealing."""
    movable = MovableSignal(name="test_movable")
    dof = RangeDOF(actuator=movable, bounds=(0, 10), parameter_type="float")
    objective = Objective(name="test_objective", minimize=False)

    config = ScipyCFG(
        dofs=[dof],
        objective=objective,
        optimizer=SCP.Dual_Annealing,
    )

    opt = ScipyOptimizer(config, timeout=5)
    assert opt.final is None  # No optimization run yet
    opt.close()


def test_scipy_optimizer_threads_none(mock_evaluation_function, mock_acquisition_plan):
    """Test ScipyOptimizer with threads=None (no parallelization)."""
    movable = MovableSignal(name="test_movable")
    dof = RangeDOF(actuator=movable, bounds=(0, 10), parameter_type="float")
    objective = Objective(name="test_objective", minimize=False)

    config = ScipyCFG(
        dofs=[dof],
        objective=objective,
        threads=None,
    )

    opt = ScipyOptimizer(config, timeout=5)
    assert opt._thread_pool is None  # No thread pool when threads=None
    opt.close()


def test_scipy_optimizer_threads_multiple(mock_evaluation_function, mock_acquisition_plan):
    """Test ScipyOptimizer with multiple threads."""
    movable = MovableSignal(name="test_movable")
    dof = RangeDOF(actuator=movable, bounds=(0, 10), parameter_type="float")
    objective = Objective(name="test_objective", minimize=False)

    config = ScipyCFG(
        dofs=[dof],
        objective=objective,
        threads=2,
    )

    opt = ScipyOptimizer(config, timeout=5)
    # Configuration accepted
    opt.close()


# ============================================================================
# PHASE 3: Rescaling & Parameter Handling Tests
# ============================================================================


def test_rescaling_suggest_output(optimizer_prep):
    """Test suggest() respects rescaling (scaled input → unscaled output)."""
    suggestions = optimizer_prep.suggest(1)
    assert len(suggestions) == 1

    # Suggested values should be in original (unscaled) space
    # Bounds are (0, 10), so suggestions should be in [0, 10]
    assert 0 <= suggestions[0]["test_movable1"] <= 10
    assert 0 <= suggestions[0]["test_movable2"] <= 10
    optimizer_prep.close()


def test_rescaling_ingest_parameters(optimizer_prep):
    """Test ingest() processes scaled parameters correctly."""
    # Suggest first
    optimizer_prep.suggest(1)
    # Ingest with outcome
    optimizer_prep.ingest([{"test_movable1": 2.5, "test_movable2": 5.0, "test_objective": 0.8, ID_KEY: 0}])
    optimizer_prep.close()


def test_get_best_points_scaling(optimizer_prep):
    """Test get_best_points() with scaling works (verify basic structure)."""
    # Set final result manually (simulate completed optimization)
    optimizer_prep.final = ScipyOptimizer.Result(
        x=[2.5, 3.0],  # Scaled values
        fun=0.85,
        nit=15,
        status=0,
    )

    best = optimizer_prep.get_best_points()
    assert isinstance(best, list)
    assert len(best) == 3  # (trial_idx, params_dict, metrics_dict)
    assert best[0] == 14  # nit - 1
    assert "test_movable1" in best[1]
    assert "test_movable2" in best[1]
    optimizer_prep.close()

# ============================================================================
# PHASE 4: Error Handling & Validation Tests
# ============================================================================


def test_ingest_raises_on_unknown_id(optimizer_prep):
    """Test ingest() raises ValueError when ID not in _active requests."""
    # Try to ingest with unknown ID
    with pytest.raises(ValueError, match="optimizer did not expect to receive an update"):
        optimizer_prep.ingest([{"test_movable1": 5.0, "test_movable2": 5.0, "test_objective": 0.5, ID_KEY: 999}])

    optimizer_prep.close()


def test_ingest_force_resiliance_skips_unknown_id(optimizer_prep):
    """Test ingest() with force_resiliance=True skips unknown IDs."""
    # Enable resiliance
    optimizer_prep.force_resiliance = True

    # This should NOT raise (unknown IDs are skipped)
    optimizer_prep.ingest([{"test_movable1": 5.0, "test_movable2": 5.0, "test_objective": 0.5, ID_KEY: 999}])

    optimizer_prep.close()


def test_ingest_missing_objective_name(optimizer_prep):
    """Test ingest() raises ValueError if objective name missing from data."""
    # Suggest first to create an active request
    optimizer_prep.suggest(1)

    # Try to ingest without objective value
    with pytest.raises(KeyError):
        optimizer_prep.ingest([{"test_movable1": 5.0, "test_movable2": 5.0, ID_KEY: 0}])  # Missing "test_objective"

    optimizer_prep.close()


def test_optimizer_close_cancels_futures(optimizer_prep):
    """Test ScipyOptimizer.close() cancels all active futures."""
    # Suggest to create active futures
    optimizer_prep.suggest(2)
    active_count = len(optimizer_prep._active)
    assert active_count > 0

    # Close should cancel all futures
    optimizer_prep.close()
    # All futures should now have exceptions set
    for future_wrapper in optimizer_prep._active.values():
        assert future_wrapper.future.done()


def test_suggest_before_optimization(optimizer_prep):
    """Test suggest() before any optimization returns expected state."""
    # Suggest before any ingest
    suggestions = optimizer_prep.suggest(1)
    assert len(suggestions) == 1
    assert "_id" in suggestions[0]
    optimizer_prep.close()


# ============================================================================
# PHASE 6: State Management & Context Manager Tests
# ============================================================================


def test_scipy_optimizer_context_manager(mock_evaluation_function, mock_acquisition_plan):
    """Test ScipyOptimizer context manager protocol (__enter__/__exit__)."""
    movable = MovableSignal(name="test_movable")
    dof = RangeDOF(actuator=movable, bounds=(0, 10), parameter_type="float")
    objective = Objective(name="test_objective", minimize=False)

    config = ScipyCFG(dofs=[dof], objective=objective)

    with ScipyOptimizer(config, timeout=5) as opt:
        assert opt is not None
        time.sleep(0.1)
        suggestions = opt.suggest(1)
        assert len(suggestions) == 1


def test_scipy_optimizer_session_reinit(mock_evaluation_function, mock_acquisition_plan):
    """Test ScipyOptimizer.session() reinitializes state."""
    movable = MovableSignal(name="test_movable")
    dof = RangeDOF(actuator=movable, bounds=(0, 10), parameter_type="float")
    objective = Objective(name="test_objective", minimize=False)

    config = ScipyCFG(dofs=[dof], objective=objective)

    opt = ScipyOptimizer(config, timeout=5)
    opt.suggest(1)

    # Call session to reinitialize
    opt.session(config, timeout=5)
    time.sleep(0.1)
    # State should be reset
    assert opt._increment == 1
    assert len(opt._active) == 1
    opt.close()


def test_get_best_points_intermediate_only(optimizer_prep):
    """Test get_best_points() with only intermediate results (final=None)."""
    # Set intermediate result manually (simulate partway through optimization)
    optimizer_prep.intermediate = ScipyOptimizer.Result(
        x=[5.0, -5.0],
        fun=0.7,
        nit=5,
        status=0,
    )

    best = optimizer_prep.get_best_points()
    assert len(best) == 3
    assert best[0] == 4  # nit - 1
    optimizer_prep.close()


def test_get_best_points_final_preferred(optimizer_prep):
    """Test get_best_points() prefers final over intermediate."""
    # Set both intermediate and final
    optimizer_prep.intermediate = ScipyOptimizer.Result(
        x=[5.0, -5.0],
        fun=0.7,
        nit=5,
        status=0,
    )
    optimizer_prep.final = ScipyOptimizer.Result(
        x=[7.0, -5.0],
        fun=0.9,
        nit=10,
        status=0,
    )

    best = optimizer_prep.get_best_points()
    # best[2] is a dict with objective as key; get the first (only) value
    objective_value = list(best[2].values())[0]
    assert objective_value == 0.9  # Uses final result
    optimizer_prep.close()


def test_get_best_points_no_optimization_raises(optimizer_prep):
    """Test get_best_points() raises ValueError if no optimization run."""
    # No optimization run: both intermediate and final are None
    with pytest.raises(ValueError, match="no optimization epoch has been recorded"):
        optimizer_prep.get_best_points()

    optimizer_prep.close()
