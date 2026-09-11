from collections.abc import Mapping, Sequence
from threading import Event
from unittest.mock import MagicMock

import pytest
from ax import Client
from bluesky_queueserver_api.zmq import REManagerAPI

from blop.ax.dof import RangeDOF
from blop.ax.objective import Objective
from blop.ax.queueserver_agent import QueueserverAgent
from blop.protocols import ID_KEY
from blop.queueserver import CORRELATION_UID_KEY, OptimizationResult, QueueserverAcquisition

from ..conftest import MovableSignal, background_call


@pytest.fixture(scope="function")
def mock_re_manager_api():
    manager = MagicMock(spec=REManagerAPI)
    manager.status.return_value = {"worker_environment_exists": True}
    manager.item_add.return_value = {"success": True, "item": {"item_uid": "item-manual"}}
    return manager


def test_queueserver_agent_submit_suggestions(mock_re_manager_api):
    outcomes_by_acquisition: dict[QueueserverAcquisition, Sequence[Mapping]] = {}
    evaluated_suggestions = []

    def evaluate(uid: QueueserverAcquisition, suggestions: Sequence[Mapping]) -> Sequence[Mapping]:
        evaluated_suggestions.extend(suggestions)
        outcomes = [
            {ID_KEY: suggestion[ID_KEY], "loss": suggestion["motor_x"] ** 2 + suggestion["motor_y"] ** 2}
            for suggestion in reversed(suggestions)
        ]
        outcomes_by_acquisition[uid] = outcomes
        return outcomes

    motor_y = MovableSignal(name="motor_y")
    agent = QueueserverAgent(
        mock_re_manager_api,
        ["detector", "monitor"],
        [
            RangeDOF(actuator="motor_x", bounds=(0.0, 10.0), parameter_type="float"),
            RangeDOF(actuator=motor_y, bounds=(0.0, 10.0), parameter_type="float"),
        ],
        [Objective(name="loss", minimize=True)],
        evaluate,
        acquisition_plan="my_acquire",
        acquisition_plan_kwargs={"exposure_time": 0.5, "num_frames": 10},
    )
    points = [{"motor_x": 2.0, "motor_y": 3.0}, {"motor_x": 4.0, "motor_y": 5.0}]
    # Supplied IDs must also be replaced by real Ax registrations.
    suggestions = [points[0], {ID_KEY: 999, **points[1]}]
    with background_call(agent.submit_suggestions, suggestions) as call:
        result = call.result(timeout=5).result(timeout=5)

    mock_re_manager_api.item_add.assert_called_once()
    plan = mock_re_manager_api.item_add.call_args.args[0]
    registered = plan.args[0]
    assert plan.name == "my_acquire"
    assert plan.args[1:] == [["motor_x", "motor_y"], ["detector", "monitor"]]
    assert [{key: value for key, value in point.items() if key != ID_KEY} for point in registered] == points
    assert registered == evaluated_suggestions
    summary = agent.ax_client.summarize().sort_values("motor_x")
    assert [point[ID_KEY] for point in registered] == summary["trial_index"].tolist()
    assert 999 not in summary["trial_index"].tolist()
    assert summary[["motor_x", "motor_y", "loss"]].to_dict("records") == [
        {"motor_x": 2.0, "motor_y": 3.0, "loss": 13.0},
        {"motor_x": 4.0, "motor_y": 5.0, "loss": 41.0},
    ]
    acquisition = next(iter(outcomes_by_acquisition))
    assert acquisition.item_uid == "item-manual"
    assert acquisition.plan_name == plan.name
    assert plan.kwargs == {
        "exposure_time": 0.5,
        "num_frames": 10,
        "md": {
            CORRELATION_UID_KEY: acquisition.correlation_uid,
            "blop_suggestions": registered,
        },
    }
    assert result == OptimizationResult(iterations_completed=1, num_points=2, uids=(acquisition,))
    assert outcomes_by_acquisition[result.uids[0]] == [
        {ID_KEY: registered[1][ID_KEY], "loss": 41.0},
        {ID_KEY: registered[0][ID_KEY], "loss": 13.0},
    ]
    assert agent.current_iteration == 1


def test_queueserver_agent_run_with_checkpoint_interval(mock_re_manager_api, monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "checkpoint.json"
    acquisitions = []
    mock_re_manager_api.item_add.side_effect = [
        {"success": True, "item": {"item_uid": f"item-{iteration}"}} for iteration in range(1, 4)
    ]

    def evaluate(uid: QueueserverAcquisition, suggestions: Sequence[Mapping]) -> Sequence[Mapping]:
        acquisitions.append(uid)
        return [{ID_KEY: suggestion[ID_KEY], "loss": suggestion["motor"] ** 2} for suggestion in suggestions]

    agent = QueueserverAgent(
        mock_re_manager_api,
        sensors=["detector"],
        dofs=[RangeDOF(actuator="motor", bounds=(0.0, 10.0), parameter_type="float")],
        objectives=[Objective(name="loss", minimize=True)],
        evaluation_function=evaluate,
        checkpoint_path=str(checkpoint_path),
    )
    optimizer = agent.to_optimization_problem().optimizer
    values = iter(range(1, 7))

    def suggest(num_points):
        return optimizer.register_suggestions([{"motor": float(next(values))} for _ in range(num_points)])

    # Replace only model-driven generation; registration, ingestion, and persistence are real.
    monkeypatch.setattr(optimizer, "suggest", suggest)
    agent.ax_client.configure_generation_strategy()
    with background_call(agent.run, iterations=3, n_points=2, checkpoint_interval=2) as call:
        result = call.result(timeout=5).result(timeout=5)

    expected = [{"motor": float(value), "loss": float(value**2)} for value in range(1, 7)]
    summary = agent.ax_client.summarize().sort_values("motor")
    assert summary[["motor", "loss"]].to_dict("records") == expected
    saved = Client.load_from_json_file(str(checkpoint_path)).summarize().sort_values("motor")
    assert saved[["motor", "loss"]].to_dict("records") == expected[:4]
    plans = [call.args[0] for call in mock_re_manager_api.item_add.call_args_list]
    assert [[point["motor"] for point in plan.args[0]] for plan in plans] == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    assert [uid.item_uid for uid in acquisitions] == ["item-1", "item-2", "item-3"]
    assert all(uid.plan_name == "default_acquire" for uid in acquisitions)
    assert result == OptimizationResult(iterations_completed=3, num_points=2, uids=tuple(acquisitions))
    assert agent.current_iteration == 3


def test_queueserver_agent_stop_waits_for_evaluation(mock_re_manager_api, monkeypatch):
    entered = Event()
    release = Event()
    acquisitions = []

    def evaluate(uid: QueueserverAcquisition, suggestions: Sequence[Mapping]) -> Sequence[Mapping]:
        acquisitions.append(uid)
        entered.set()
        assert release.wait(timeout=5)
        return [{ID_KEY: suggestion[ID_KEY], "loss": suggestion["motor"] ** 2} for suggestion in suggestions]

    agent = QueueserverAgent(
        mock_re_manager_api,
        sensors=["detector"],
        dofs=[RangeDOF(actuator="motor", bounds=(0.0, 10.0), parameter_type="float")],
        objectives=[Objective(name="loss", minimize=True)],
        evaluation_function=evaluate,
    )
    optimizer = agent.to_optimization_problem().optimizer

    def suggest(num_points):
        return optimizer.register_suggestions([{"motor": float(value)} for value in range(1, num_points + 1)])

    monkeypatch.setattr(optimizer, "suggest", suggest)
    with background_call(agent.run, iterations=3, n_points=2) as call:
        try:
            future = call.result(timeout=5)
            assert entered.wait(timeout=5)
            assert agent.current_iteration == 1
            with background_call(agent.stop) as stop_call:
                try:
                    stop_call.result(timeout=5)
                    assert not future.done()
                    assert not future.cancel()
                finally:
                    release.set()
        finally:
            release.set()
            result = call.result(timeout=5).result(timeout=5)

    mock_re_manager_api.item_add.assert_called_once()
    assert result == OptimizationResult(iterations_completed=1, num_points=2, uids=tuple(acquisitions))
    assert agent.ax_client.summarize().sort_values("motor")[["motor", "loss"]].to_dict("records") == [
        {"motor": 1.0, "loss": 1.0},
        {"motor": 2.0, "loss": 4.0},
    ]
    assert agent.current_iteration == 1
    agent.stop()
    assert future.result(timeout=5) is result
