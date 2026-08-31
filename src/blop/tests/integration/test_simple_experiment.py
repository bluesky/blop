import logging
import warnings

import pytest
from bluesky.run_engine import RunEngine
from bluesky_tiled_plugins import TiledWriter
from tiled.client import from_uri
from tiled.client.container import Container
from tiled.server import SimpleTiledServer

from blop.ax import Agent, Objective, RangeDOF
from blop.plans import optimize_in_run

from ..conftest import MovableSignal

pytestmark = pytest.mark.integration


class HimmelblauEvaluation:
    def __init__(self, tiled_client: Container) -> None:
        self.tiled_client = tiled_client
        self.uids: list[str] = []
        self.outcomes: list[dict] = []

    def __call__(self, uid: str, suggestions: list[dict]) -> list[dict]:
        self.uids.append(uid)
        run = self.tiled_client[uid]
        reordered_suggestions = run.start["blop_suggestions"]
        x1_data = run["primary/x1"].read()
        x2_data = run["primary/x2"].read()

        assert {suggestion["_id"] for suggestion in reordered_suggestions} == {
            suggestion["_id"] for suggestion in suggestions
        }

        outcomes = []
        for index, suggestion in enumerate(reordered_suggestions):
            x1 = x1_data[index]
            x2 = x2_data[index]
            value = (x1**2 + x2 - 11) ** 2 + (x1 + x2**2 - 7) ** 2
            outcomes.append({"_id": suggestion["_id"], "himmelblau_2d": value})

        self.outcomes.extend(outcomes)
        return outcomes


class InRunHimmelblauEvaluation:
    def __init__(self) -> None:
        self.run_uid: str | None = None
        self.primary_events: list[dict] = []
        self._evaluated_event_count = 0
        self.acquisition_ids: list[tuple[int, ...]] = []
        self.outcomes: list[dict] = []
        self._primary_descriptors: set[str] = set()

    def receive_document(self, name, doc) -> None:
        if name == "start" and doc.get("run_key") == "optimize_in_run":
            self.run_uid = doc["uid"]
        elif name == "descriptor" and doc["name"] == "primary":
            self._primary_descriptors.add(doc["uid"])
        elif name == "event" and doc["descriptor"] in self._primary_descriptors:
            self.primary_events.append(doc["data"])

    def __call__(self, uid: tuple[int, ...], suggestions: list[dict]) -> list[dict]:
        assert set(uid) == {suggestion["_id"] for suggestion in suggestions}

        events = self.primary_events[self._evaluated_event_count :]
        assert len(events) == len(uid)
        outcomes = []
        for suggestion_id, event in zip(uid, events, strict=True):
            x1 = event["x1"]
            x2 = event["x2"]
            value = (x1**2 + x2 - 11) ** 2 + (x1 + x2**2 - 7) ** 2
            outcomes.append({"_id": suggestion_id, "himmelblau_2d": value})

        self._evaluated_event_count += len(events)
        self.acquisition_ids.append(uid)
        self.outcomes.extend(outcomes)
        return outcomes


def test_simple_experiment_tiled_round_trip() -> None:
    logging.getLogger("httpx").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=FutureWarning)

    tiled_server = SimpleTiledServer()
    try:
        tiled_client = from_uri(tiled_server.uri)
        tiled_writer = TiledWriter(tiled_client)
        RE = RunEngine({})
        RE.subscribe(tiled_writer)

        x1 = MovableSignal("x1", initial_value=0.1)
        x2 = MovableSignal("x2", initial_value=0.23)

        dofs = [
            RangeDOF(actuator=x1, bounds=(-5, 5), parameter_type="float"),
            RangeDOF(actuator=x2, bounds=(-5, 5), parameter_type="float"),
        ]
        objective = Objective(name="himmelblau_2d", minimize=True)
        evaluation = HimmelblauEvaluation(tiled_client=tiled_client)
        agent = Agent(
            sensors=[],
            dofs=dofs,
            objectives=[objective],
            evaluation_function=evaluation,
            name="integration-simple-experiment",
            description="Scheduled integration test for the simple experiment",
        )

        RE(agent.optimize(iterations=2, n_points=2))

        assert len(evaluation.uids) == 2
        assert len(set(evaluation.uids)) == 2
        assert len(evaluation.outcomes) == 4
        assert len({outcome["_id"] for outcome in evaluation.outcomes}) == 4
        assert all(outcome["himmelblau_2d"] >= 0 for outcome in evaluation.outcomes)
        assert len(agent.ax_client.summarize()) == 4
    finally:
        tiled_server.close()


def test_optimize_in_run_tiled_round_trip() -> None:
    logging.getLogger("httpx").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=FutureWarning)

    tiled_server = SimpleTiledServer()
    try:
        tiled_client = from_uri(tiled_server.uri)
        tiled_writer = TiledWriter(tiled_client)
        RE = RunEngine({})
        RE.subscribe(tiled_writer)

        x1 = MovableSignal("x1", initial_value=0.1)
        x2 = MovableSignal("x2", initial_value=0.23)
        dofs = [
            RangeDOF(actuator=x1, bounds=(-5, 5), parameter_type="float"),
            RangeDOF(actuator=x2, bounds=(-5, 5), parameter_type="float"),
        ]
        evaluation = InRunHimmelblauEvaluation()
        RE.subscribe(evaluation.receive_document)
        agent = Agent(
            sensors=[],
            dofs=dofs,
            objectives=[Objective(name="himmelblau_2d", minimize=True)],
            evaluation_function=evaluation,
            name="integration-single-run-experiment",
            description="Scheduled integration test for single-run optimization",
        )

        # Keep each integration iteration at four points rather than crossing Ax's initialization boundary.
        agent.ax_client.configure_generation_strategy(method="random_search", initialization_random_seed=0)

        RE(optimize_in_run(agent.to_optimization_problem(), iterations=2, n_points=4))

        assert evaluation.run_uid is not None
        run = tiled_client[evaluation.run_uid]
        assert len(tiled_client) == 1
        assert run.start["plan_name"] == "optimize_in_run"
        assert run.start["run_key"] == "optimize_in_run"
        assert run.start["n_points"] == 4
        assert run.start["optimization_stream"] == "optimization"
        assert len(run["primary/x1"].read()) == 8
        assert len(run["primary/x2"].read()) == 8
        assert len(run["optimization/suggestion_ids"].read()) == 2
        assert len(run["optimization/acquisition_uid"].read()) == 2
        assert len(evaluation.acquisition_ids) == 2
        assert all(len(acquisition_ids) == 4 for acquisition_ids in evaluation.acquisition_ids)
        assert len(evaluation.primary_events) == 8
        assert len(evaluation.outcomes) == 8
        assert len({outcome["_id"] for outcome in evaluation.outcomes}) == 8
        assert all(outcome["himmelblau_2d"] >= 0 for outcome in evaluation.outcomes)
        assert len(agent.ax_client.summarize()) == 8
    finally:
        tiled_server.close()
