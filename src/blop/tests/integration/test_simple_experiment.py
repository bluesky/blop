import logging
import warnings

import pytest
from bluesky.run_engine import RunEngine
from bluesky_tiled_plugins import TiledWriter
from tiled.client import from_uri
from tiled.client.container import Container
from tiled.server import SimpleTiledServer

from blop.ax import Agent, Objective, RangeDOF

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
        acquisition_order = run.start["blop_acquisition_order"]
        suggestions_by_id = {suggestion["_id"]: suggestion for suggestion in suggestions}
        x1_data = run["primary/x1"].read()
        x2_data = run["primary/x2"].read()

        assert set(acquisition_order) == set(suggestions_by_id)

        outcomes = []
        for index, suggestion_id in enumerate(acquisition_order):
            suggestion = suggestions_by_id[suggestion_id]
            x1 = x1_data[index]
            x2 = x2_data[index]
            value = (x1**2 + x2 - 11) ** 2 + (x1 + x2**2 - 7) ** 2
            outcomes.append({"_id": suggestion["_id"], "himmelblau_2d": value})

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
