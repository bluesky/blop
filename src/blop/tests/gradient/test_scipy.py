import time
from unittest.mock import MagicMock

import pytest

from blop.ax import Objective, RangeDOF
from blop.gradient import SCP, Scipy, ScipyCFG, ScipyOptimizer
from blop.protocols import ID_KEY, AcquisitionPlan, EvaluationFunction

from ..conftest import MovableSignal, ReadableSignal


@pytest.fixture(scope="function")
def mock_evaluation_function():
    return MagicMock(spec=EvaluationFunction)


@pytest.fixture(scope="function")
def mock_acquisition_plan():
    return MagicMock(spec=AcquisitionPlan)


# agent._optimizer.close() is called so the standard timeout doesnt make the testing take forever


@pytest.fixture(scope="function")
def agent_prep(mock_evaluation_function, mock_acquisition_plan):
    movable1 = MovableSignal(name="test_movable1")
    movable2 = MovableSignal(name="test_movable2")
    readable = ReadableSignal(name="test_readable")
    dof1 = RangeDOF(actuator=movable1, bounds=(0, 10), parameter_type="float")
    dof2 = RangeDOF(actuator=movable2, bounds=(0, 10), parameter_type="float")
    objective = Objective(name="test_objective", minimize=False)
    config = ScipyCFG(dofs=[dof1, dof2], objective=objective, threads=4)
    agent = Scipy(
        sensors=[readable],
        config=config,
        evaluation_function=mock_evaluation_function,
        acquisition_plan=mock_acquisition_plan,
        name="test_experiment",
        timeout=5,
    )
    time.sleep(0.1)
    return agent


@pytest.fixture(scope="function")
def rescaled_agent_prep(mock_evaluation_function, mock_acquisition_plan):
    movable1 = MovableSignal(name="test_movable1")
    movable2 = MovableSignal(name="test_movable2")
    readable = ReadableSignal(name="test_readable")
    dof1 = RangeDOF(actuator=movable1, bounds=(0, 10), parameter_type="float")
    dof2 = RangeDOF(actuator=movable2, bounds=(0, 10), parameter_type="float")
    objective = Objective(name="test_objective", minimize=False)
    config = ScipyCFG(
        dofs=[dof1, dof2],
        objective=objective,
        threads=4,
        rescale=[2.0, 3.0],
    )
    agent = Scipy(
        sensors=[readable],
        config=config,
        evaluation_function=mock_evaluation_function,
        acquisition_plan=mock_acquisition_plan,
        name="test_experiment",
        timeout=5,
    )
    time.sleep(0.1)
    return agent


@pytest.fixture(scope="function")
def single_dof_agent_prep(mock_evaluation_function, mock_acquisition_plan):
    movable = MovableSignal(name="test_movable")
    readable = ReadableSignal(name="test_readable")
    dof = RangeDOF(actuator=movable, bounds=(0, 10), parameter_type="float")
    objective = Objective(name="test_objective", minimize=False)
    config = ScipyCFG(dofs=[dof], objective=objective)
    agent = Scipy(
        sensors=[readable],
        config=config,
        evaluation_function=mock_evaluation_function,
        acquisition_plan=mock_acquisition_plan,
        timeout=5,
    )
    time.sleep(0.1)
    return agent


def test_general_init(mock_evaluation_function, mock_acquisition_plan):
    """Test that the simple Scipy can be initialized."""
    movable1 = MovableSignal(name="test_movable1")
    movable2 = MovableSignal(name="test_movable2")
    readable = ReadableSignal(name="test_readable")
    dof1 = RangeDOF(actuator=movable1, bounds=(0, 10), parameter_type="float")
    dof2 = RangeDOF(actuator=movable2, bounds=(0, 10), parameter_type="float")
    objective = Objective(name="test_objective", minimize=False)
    config = ScipyCFG(
        dofs=[dof1, dof2],
        objective=objective,
    )
    agent = Scipy(
        sensors=[readable],
        config=config,
        evaluation_function=mock_evaluation_function,
        acquisition_plan=mock_acquisition_plan,
        name="test_experiment",
        timeout=5,
    )
    assert agent.sensors == [readable]
    assert agent.actuators == [dof1.actuator, dof2.actuator]
    assert agent.evaluation_function == mock_evaluation_function
    assert agent.acquisition_plan == mock_acquisition_plan
    agent._optimizer.close()


def test_agent_init(mock_evaluation_function, mock_acquisition_plan):
    """Test that the agent can be initialized."""
    movable1 = MovableSignal(name="test_movable1")
    movable2 = MovableSignal(name="test_movable2")
    readable = ReadableSignal(name="test_readable")
    dof1 = RangeDOF(actuator=movable1, bounds=(0, 10), parameter_type="float")
    dof2 = RangeDOF(actuator=movable2, bounds=(0, 10), parameter_type="float")
    objective = Objective(name="test_objective", minimize=False)
    agent = Scipy.Agent(
        sensors=[readable],
        dofs=[dof1, dof2],
        objectives=[objective],
        evaluation_function=mock_evaluation_function,
        acquisition_plan=mock_acquisition_plan,
        name="test_experiment",
        timeout=5,
    )
    assert agent.sensors == [readable]
    assert agent.actuators == [dof1.actuator, dof2.actuator]
    assert agent.evaluation_function == mock_evaluation_function
    assert agent.acquisition_plan == mock_acquisition_plan
    agent._optimizer.close()


def test_agent_to_optimization_problem(mock_evaluation_function):
    """Test that the agent can be converted to an optimization problem."""
    movable1 = MovableSignal(name="test_movable1")
    movable2 = MovableSignal(name="test_movable2")
    dof1 = RangeDOF(actuator=movable1, bounds=(0, 10), parameter_type="float")
    dof2 = RangeDOF(actuator=movable2, bounds=(0, 10), parameter_type="float")
    objective = Objective(name="test_objective", minimize=False)
    agent = Scipy.Agent(
        sensors=[], dofs=[dof1, dof2], objectives=[objective], evaluation_function=mock_evaluation_function, timeout=5
    )
    optimization_problem = agent.to_optimization_problem()
    assert optimization_problem.evaluation_function == mock_evaluation_function
    assert optimization_problem.actuators == [movable1, movable2]
    assert optimization_problem.sensors == []
    assert isinstance(optimization_problem.optimizer, ScipyOptimizer)
    assert optimization_problem.acquisition_plan is None
    agent._optimizer.close()


def test_agent_suggest(agent_prep):
    parameterizations = agent_prep.suggest(1)
    assert len(parameterizations) == 1
    assert parameterizations[0]["_id"] == 0
    assert "test_movable1" in parameterizations[0]
    assert "test_movable2" in parameterizations[0]
    assert isinstance(parameterizations[0]["test_movable1"], (int, float))
    assert isinstance(parameterizations[0]["test_movable2"], (int, float))
    agent_prep._optimizer.close()


def test_agent_ingest(agent_prep):
    agent_prep.suggest()
    agent_prep.ingest([{"test_movable1": 0.1, "test_movable2": 0.2, "test_objective": 0.3, ID_KEY: 0}])
    agent_prep._optimizer.close()


def test_agent_multithread(agent_prep):
    agent_prep.suggest(1)
    agent_prep.ingest([{"test_movable1": 0.1, "test_movable2": 0.2, "test_objective": 0.3, ID_KEY: 0}])
    time.sleep(0.1)
    params = agent_prep.suggest(4)
    print(agent_prep._optimizer._active)
    assert len(params) > 1
    agent_prep._optimizer.close()


# ============================================================================
# PHASE 1: Configuration & Initialization Tests
# ============================================================================


def test_scipy_cfg_rescaling_scalar(mock_evaluation_function, mock_acquisition_plan):
    """Test ScipyCFG with scalar rescaling."""
    movable = MovableSignal(name="test_movable")
    dof = RangeDOF(actuator=movable, bounds=(0, 10), parameter_type="float")
    objective = Objective(name="test_objective", minimize=False)

    config = ScipyCFG(
        dofs=[dof],
        objective=objective,
        rescale=2.0,
    )

    agent = Scipy(
        sensors=[],
        config=config,
        evaluation_function=mock_evaluation_function,
        acquisition_plan=mock_acquisition_plan,
        timeout=5,
    )

    # Verify rescaling was applied
    assert agent._optimizer._scale[0] == 2.0
    agent._optimizer.close()


def test_scipy_cfg_rescaling_list(mock_evaluation_function, mock_acquisition_plan):
    """Test ScipyCFG with list rescaling per parameter."""
    movable1 = MovableSignal(name="test_movable1")
    movable2 = MovableSignal(name="test_movable2")
    dof1 = RangeDOF(actuator=movable1, bounds=(0, 10), parameter_type="float")
    dof2 = RangeDOF(actuator=movable2, bounds=(0, 10), parameter_type="float")
    objective = Objective(name="test_objective", minimize=False)

    config = ScipyCFG(
        dofs=[dof1, dof2],
        objective=objective,
        rescale=[2.0, 3.0],
    )

    agent = Scipy(
        sensors=[],
        config=config,
        evaluation_function=mock_evaluation_function,
        acquisition_plan=mock_acquisition_plan,
        timeout=5,
    )

    # Verify rescaling per DOF
    assert agent._optimizer._scale[0] == 2.0
    assert agent._optimizer._scale[1] == 3.0
    agent._optimizer.close()


def test_scipy_cfg_initial_parameters(mock_evaluation_function, mock_acquisition_plan):
    """Test ScipyCFG with initial parameter values."""
    movable1 = MovableSignal(name="test_movable1")
    movable2 = MovableSignal(name="test_movable2")
    dof1 = RangeDOF(actuator=movable1, bounds=(0, 10), parameter_type="float")
    dof2 = RangeDOF(actuator=movable2, bounds=(0, 10), parameter_type="float")
    objective = Objective(name="test_objective", minimize=False)

    initial_params = [2.5, 7.5]
    config = ScipyCFG(
        dofs=[dof1, dof2],
        objective=objective,
        initial=initial_params,
    )

    agent = Scipy(
        sensors=[],
        config=config,
        evaluation_function=mock_evaluation_function,
        acquisition_plan=mock_acquisition_plan,
        timeout=5,
    )

    # Verify initial parameters are set
    agent._optimizer.close()


def test_scipy_cfg_max_iter_and_eps(mock_evaluation_function, mock_acquisition_plan):
    """Test ScipyCFG with max_iter and eps parameters."""
    movable = MovableSignal(name="test_movable")
    dof = RangeDOF(actuator=movable, bounds=(0, 10), parameter_type="float")
    objective = Objective(name="test_objective", minimize=False)

    config = ScipyCFG(
        dofs=[dof],
        objective=objective,
        max_iter=50,
        eps=1e-6,
    )

    assert config.max_iter == 50
    assert config.eps == 1e-6


def test_agent_invalid_optimizer_enum(mock_evaluation_function, mock_acquisition_plan):
    """Test Scipy.Agent raises ValueError for invalid optimizer."""
    movable = MovableSignal(name="test_movable")
    dof = RangeDOF(actuator=movable, bounds=(0, 10), parameter_type="float")
    objective = Objective(name="test_objective", minimize=False)
    readable = ReadableSignal(name="test_readable")

    with pytest.raises(ValueError, match="optimizer.*not in supported optimizers"):
        Scipy.Agent(
            sensors=[readable],
            dofs=[dof],
            objectives=[objective],
            evaluation_function=mock_evaluation_function,
            optimizer="invalid_optimizer",
        )


def test_agent_multiple_objectives_not_supported(mock_evaluation_function, mock_acquisition_plan):
    """Test Scipy.Agent raises ValueError for multiple objectives."""
    movable = MovableSignal(name="test_movable")
    dof = RangeDOF(actuator=movable, bounds=(0, 10), parameter_type="float")
    objective1 = Objective(name="test_objective_1", minimize=False)
    objective2 = Objective(name="test_objective_2", minimize=False)
    readable = ReadableSignal(name="test_readable")

    with pytest.raises(ValueError, match="Multiple Objectives are not supported"):
        Scipy.Agent(
            sensors=[readable],
            dofs=[dof],
            objectives=[objective1, objective2],
            evaluation_function=mock_evaluation_function,
        )


# ============================================================================
# PHASE 2: Optimizer Algorithm Variations Tests
# ============================================================================


@pytest.mark.parametrize("optimizer", [SCP.Default, SCP.BFGS, SCP.Dual_Annealing])
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


def test_rescaling_suggest_output(rescaled_agent_prep):
    """Test suggest() respects rescaling (scaled input → unscaled output)."""
    suggestions = rescaled_agent_prep.suggest(1)
    assert len(suggestions) == 1

    # Suggested values should be in original (unscaled) space
    # Bounds are (0, 10), so suggestions should be in [0, 10]
    assert 0 <= suggestions[0]["test_movable1"] <= 10
    assert 0 <= suggestions[0]["test_movable2"] <= 10
    rescaled_agent_prep._optimizer.close()


def test_rescaling_ingest_parameters(rescaled_agent_prep):
    """Test ingest() processes scaled parameters correctly."""
    # Suggest first
    rescaled_agent_prep.suggest(1)
    # Ingest with outcome
    rescaled_agent_prep.ingest([{"test_movable1": 2.5, "test_movable2": 5.0, "test_objective": 0.8, ID_KEY: 0}])
    rescaled_agent_prep._optimizer.close()


def test_get_best_points_scaling(rescaled_agent_prep):
    """Test get_best_points() with scaling works (verify basic structure)."""
    # Set final result manually (simulate completed optimization)
    rescaled_agent_prep._optimizer.final = ScipyOptimizer.Result(
        x=[2.5, 3.0],  # Scaled values
        fun=0.85,
        nit=15,
        status=0,
    )

    best = rescaled_agent_prep._optimizer.get_best_points()
    assert isinstance(best, list)
    assert len(best) == 3  # (trial_idx, params_dict, metrics_dict)
    assert best[0] == 14  # nit - 1
    assert "test_movable1" in best[1]
    assert "test_movable2" in best[1]
    rescaled_agent_prep._optimizer.close()


# ============================================================================
# PHASE 4: Error Handling & Validation Tests
# ============================================================================


def test_ingest_raises_on_unknown_id(agent_prep):
    """Test ingest() raises ValueError when ID not in _active requests."""
    # Try to ingest with unknown ID
    with pytest.raises(ValueError, match="optimizer did not expect to receive an update"):
        agent_prep.ingest([{"test_movable1": 5.0, "test_movable2": 5.0, "test_objective": 0.5, ID_KEY: 999}])

    agent_prep._optimizer.close()


def test_ingest_force_resiliance_skips_unknown_id(agent_prep):
    """Test ingest() with force_resiliance=True skips unknown IDs."""
    # Enable resiliance
    agent_prep._optimizer.force_resiliance = True

    # This should NOT raise (unknown IDs are skipped)
    agent_prep.ingest([{"test_movable1": 5.0, "test_movable2": 5.0, "test_objective": 0.5, ID_KEY: 999}])

    agent_prep._optimizer.close()


def test_ingest_missing_objective_name(agent_prep):
    """Test ingest() raises ValueError if objective name missing from data."""
    # Suggest first to create an active request
    agent_prep.suggest(1)

    # Try to ingest without objective value
    with pytest.raises(KeyError):
        agent_prep.ingest([{"test_movable1": 5.0, "test_movable2": 5.0, ID_KEY: 0}])  # Missing "test_objective"

    agent_prep._optimizer.close()


def test_optimizer_close_cancels_futures(agent_prep):
    """Test ScipyOptimizer.close() cancels all active futures."""
    # Suggest to create active futures
    agent_prep.suggest(2)
    active_count = len(agent_prep._optimizer._active)
    assert active_count > 0

    # Close should cancel all futures
    agent_prep._optimizer.close()
    # All futures should now have exceptions set
    for future_wrapper in agent_prep._optimizer._active.values():
        assert future_wrapper.future.done()


def test_suggest_before_optimization(agent_prep):
    """Test suggest() before any optimization returns expected state."""
    # Suggest before any ingest
    suggestions = agent_prep.suggest(1)
    assert len(suggestions) == 1
    assert "_id" in suggestions[0]
    agent_prep._optimizer.close()


# ============================================================================
# PHASE 5: Callback Management Tests
# ============================================================================


def test_subscribe_callback(single_dof_agent_prep):
    """Test subscribe() adds callback to list."""
    callback = MagicMock()
    initial_count = len(single_dof_agent_prep.callbacks)
    single_dof_agent_prep.subscribe(callback)

    assert len(single_dof_agent_prep.callbacks) == initial_count + 1
    assert callback in single_dof_agent_prep.callbacks
    single_dof_agent_prep._optimizer.close()


def test_subscribe_duplicate_raises(single_dof_agent_prep):
    """Test subscribe() raises ValueError on duplicate callback."""
    callback = MagicMock()
    single_dof_agent_prep.subscribe(callback)

    with pytest.raises(ValueError, match="already subscribed"):
        single_dof_agent_prep.subscribe(callback)

    single_dof_agent_prep._optimizer.close()


def test_unsubscribe_callback(single_dof_agent_prep):
    """Test unsubscribe() removes callback from list."""
    callback = MagicMock()
    single_dof_agent_prep.subscribe(callback)
    assert callback in single_dof_agent_prep.callbacks

    single_dof_agent_prep.unsubscribe(callback)
    assert callback not in single_dof_agent_prep.callbacks
    single_dof_agent_prep._optimizer.close()


def test_unsubscribe_not_subscribed_raises(single_dof_agent_prep):
    """Test unsubscribe() raises ValueError if not subscribed."""
    callback = MagicMock()

    with pytest.raises(ValueError):
        single_dof_agent_prep.unsubscribe(callback)

    single_dof_agent_prep._optimizer.close()


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


def test_get_best_points_intermediate_only(single_dof_agent_prep):
    """Test get_best_points() with only intermediate results (final=None)."""
    # Set intermediate result manually (simulate partway through optimization)
    single_dof_agent_prep._optimizer.intermediate = ScipyOptimizer.Result(
        x=[5.0],
        fun=0.7,
        nit=5,
        status=0,
    )

    best = single_dof_agent_prep._optimizer.get_best_points()
    assert len(best) == 3
    assert best[0] == 4  # nit - 1
    single_dof_agent_prep._optimizer.close()


def test_get_best_points_final_preferred(single_dof_agent_prep):
    """Test get_best_points() prefers final over intermediate."""
    # Set both intermediate and final
    single_dof_agent_prep._optimizer.intermediate = ScipyOptimizer.Result(
        x=[5.0],
        fun=0.7,
        nit=5,
        status=0,
    )
    single_dof_agent_prep._optimizer.final = ScipyOptimizer.Result(
        x=[7.0],
        fun=0.9,
        nit=10,
        status=0,
    )

    best = single_dof_agent_prep._optimizer.get_best_points()
    # best[2] is a dict with objective as key; get the first (only) value
    objective_value = list(best[2].values())[0]
    assert objective_value == 0.9  # Uses final result
    single_dof_agent_prep._optimizer.close()


def test_get_best_points_no_optimization_raises(single_dof_agent_prep):
    """Test get_best_points() raises ValueError if no optimization run."""
    # No optimization run: both intermediate and final are None
    with pytest.raises(ValueError, match="no optimization epoch has been recorded"):
        single_dof_agent_prep._optimizer.get_best_points()

    single_dof_agent_prep._optimizer.close()


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

    # State should be reset
    assert opt._increment == 0
    assert len(opt._active) == 0
    opt.close()


# ============================================================================
# PHASE 7: Edge Cases & Boundary Conditions Tests
# ============================================================================


def test_scipy_single_dof(single_dof_agent_prep):
    """Test Scipy with single DOF (one parameter)."""
    suggestions = single_dof_agent_prep.suggest(1)
    assert len(suggestions) == 1
    assert "test_movable" in suggestions[0]
    single_dof_agent_prep._optimizer.close()


def test_scipy_large_rescale_factors(mock_evaluation_function, mock_acquisition_plan):
    """Test Scipy with large rescale factors (extreme scaling)."""
    movable1 = MovableSignal(name="test_movable1")
    movable2 = MovableSignal(name="test_movable2")
    dof1 = RangeDOF(actuator=movable1, bounds=(0, 10), parameter_type="float")
    dof2 = RangeDOF(actuator=movable2, bounds=(0, 10), parameter_type="float")
    objective = Objective(name="test_objective", minimize=False)
    readable = ReadableSignal(name="test_readable")

    config = ScipyCFG(
        dofs=[dof1, dof2],
        objective=objective,
        rescale=[0.001, 1000.0],  # Extreme scaling
    )

    agent = Scipy(
        sensors=[readable],
        config=config,
        evaluation_function=mock_evaluation_function,
        acquisition_plan=mock_acquisition_plan,
        timeout=5,
    )
    time.sleep(0.1)

    suggestions = agent.suggest(1)
    assert len(suggestions) == 1
    # Values should still be in original bounds
    assert 0 <= suggestions[0]["test_movable1"] <= 10
    assert 0 <= suggestions[0]["test_movable2"] <= 10
    agent._optimizer.close()


def test_suggest_after_final_optimization(single_dof_agent_prep):
    """Test suggest() after final optimization returns final result parameterization."""
    # Set final optimization result
    single_dof_agent_prep._optimizer.final = ScipyOptimizer.Result(
        x=[7.0],
        fun=0.95,
        nit=20,
        status=0,
    )

    suggestions = single_dof_agent_prep._optimizer.suggest()
    assert len(suggestions) == 1
    assert suggestions[0]["test_movable"] == 7.0
    assert suggestions[0][ID_KEY] == 20
    single_dof_agent_prep._optimizer.close()
