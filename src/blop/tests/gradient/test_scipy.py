import time
from unittest.mock import MagicMock

import pytest

from blop.ax import Objective, RangeDOF
from blop.gradient import Scipy, ScipyCFG, ScipyOptimizer
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
def secoundary_agent_prep(mock_evaluation_function, mock_acquisition_plan):
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

    with pytest.raises((ValueError, NotImplementedError), match="optimizer.*not in supported optimizers"):
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
# PHASE 5: Callback Management Tests
# ============================================================================


def test_subscribe_callback(secoundary_agent_prep):
    """Test subscribe() adds callback to list."""
    callback = MagicMock()
    initial_count = len(secoundary_agent_prep.callbacks)
    secoundary_agent_prep.subscribe(callback)

    assert len(secoundary_agent_prep.callbacks) == initial_count + 1
    assert callback in secoundary_agent_prep.callbacks
    secoundary_agent_prep._optimizer.close()


def test_subscribe_duplicate_raises(secoundary_agent_prep):
    """Test subscribe() raises ValueError on duplicate callback."""
    callback = MagicMock()
    secoundary_agent_prep.subscribe(callback)

    with pytest.raises(ValueError, match="already subscribed"):
        secoundary_agent_prep.subscribe(callback)

    secoundary_agent_prep._optimizer.close()


def test_unsubscribe_callback(secoundary_agent_prep):
    """Test unsubscribe() removes callback from list."""
    callback = MagicMock()
    secoundary_agent_prep.subscribe(callback)
    assert callback in secoundary_agent_prep.callbacks

    secoundary_agent_prep.unsubscribe(callback)
    assert callback not in secoundary_agent_prep.callbacks
    secoundary_agent_prep._optimizer.close()


def test_unsubscribe_not_subscribed_raises(secoundary_agent_prep):
    """Test unsubscribe() raises ValueError if not subscribed."""
    callback = MagicMock()

    with pytest.raises(ValueError):
        secoundary_agent_prep.unsubscribe(callback)

    secoundary_agent_prep._optimizer.close()


# ============================================================================
# PHASE 7: Edge Cases & Boundary Conditions Tests
# ============================================================================


def test_scipy_secoundary(secoundary_agent_prep):
    """Test Scipy with single DOF (one parameter)."""
    suggestions = secoundary_agent_prep.suggest(1)
    assert len(suggestions) == 1
    assert "test_movable" in suggestions[0]
    secoundary_agent_prep._optimizer.close()


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


def test_suggest_after_final_optimization(secoundary_agent_prep):
    """Test suggest() after final optimization returns final result parameterization."""
    # Set final optimization result
    secoundary_agent_prep._optimizer.final = ScipyOptimizer.Result(
        x=[7.0],
        fun=0.95,
        nit=20,
        status=0,
    )

    suggestions = secoundary_agent_prep._optimizer.suggest()
    assert len(suggestions) == 1
    assert suggestions[0]["test_movable"] == 7.0
    assert suggestions[0][ID_KEY] == 20
    secoundary_agent_prep._optimizer.close()
