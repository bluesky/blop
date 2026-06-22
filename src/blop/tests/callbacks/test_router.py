from unittest.mock import MagicMock

import pytest
from bluesky import SupplementalData
from bluesky.callbacks import CallbackBase
from bluesky.run_engine import RunEngine

from blop.callbacks.router import OptimizationCallbackRouter
from blop.plans import OPTIMIZE_RUN_KEY, SAMPLE_SUGGESTIONS_RUN_KEY, optimize
from blop.protocols import EvaluationFunction, OptimizationProblem, Optimizer

from ..conftest import MovableSignal, ReadableSignal


class _SpyCallback(CallbackBase):
    """A CallbackBase that records calls to start()."""

    def __init__(self):
        super().__init__()
        self.start_mock = MagicMock()

    def start(self, doc):
        self.start_mock(doc)


@pytest.mark.parametrize("run_key", [OPTIMIZE_RUN_KEY, SAMPLE_SUGGESTIONS_RUN_KEY])
def test_routes_matching_run_keys(run_key):
    cb = _SpyCallback()
    router = OptimizationCallbackRouter([cb])

    router("start", {"uid": "test123", "run_key": run_key})
    cb.start_mock.assert_called_once()


def test_ignores_non_matching_run_key():
    cb = _SpyCallback()
    router = OptimizationCallbackRouter([cb])

    router("start", {"uid": "test123", "run_key": "some_other_run"})
    cb.start_mock.assert_not_called()


def test_ignores_missing_run_key():
    cb = _SpyCallback()
    router = OptimizationCallbackRouter([cb])

    router("start", {"uid": "test123"})
    cb.start_mock.assert_not_called()


def test_mutating_list_affects_next_run():
    cb1 = _SpyCallback()
    cb2 = _SpyCallback()
    callbacks: list[CallbackBase] = [cb1]
    router = OptimizationCallbackRouter(callbacks)

    # First run — only cb1
    router("start", {"uid": "run1", "run_key": OPTIMIZE_RUN_KEY})
    cb1.start_mock.assert_called_once()
    cb2.start_mock.assert_not_called()

    # Add cb2 between runs
    callbacks.append(cb2)

    # Second run — both
    router("start", {"uid": "run2", "run_key": OPTIMIZE_RUN_KEY})
    assert cb1.start_mock.call_count == 2
    cb2.start_mock.assert_called_once()


def test_empty_callback_list():
    router = OptimizationCallbackRouter([])

    # Should not raise
    router("start", {"uid": "test123", "run_key": OPTIMIZE_RUN_KEY})


@pytest.fixture()
def RE():
    return RunEngine({})


@pytest.fixture()
def optimization_problem():
    optimizer = MagicMock(spec=Optimizer)
    optimizer.suggest.return_value = [{"x1": 0.0, "_id": 0}]
    evaluation_function = MagicMock(spec=EvaluationFunction, return_value=[{"objective": 0.0, "_id": 0}])
    return OptimizationProblem(
        optimizer=optimizer,
        actuators=[MovableSignal("x1", initial_value=-1.0)],
        sensors=[ReadableSignal("objective")],
        evaluation_function=evaluation_function,
    )


def test_baseline_stream_is_filtered_out(RE, optimization_problem):
    """Router must not forward baseline stream descriptors or events to callbacks.

    SupplementalData injects a 'baseline' stream (read at run start/end) into
    every run, mimicking real beamline usage. The router should shield callbacks
    from those documents and pass through only the 'primary' stream.
    """

    class _SpyAllCallback(CallbackBase):
        def __init__(self):
            super().__init__()
            self.received: list[tuple[str, dict]] = []

        def descriptor(self, doc):
            self.received.append(("descriptor", doc))

        def event(self, doc):
            self.received.append(("event", doc))

        def stop(self, doc):
            self.received.append(("stop", doc))

    baseline_device = ReadableSignal("temperature")
    sd = SupplementalData(baseline=[baseline_device])
    RE.preprocessors.append(sd)

    cb = _SpyAllCallback()
    router = OptimizationCallbackRouter([cb])
    RE.subscribe(router)

    RE(optimize(optimization_problem))

    descriptor_stream_names = [doc["name"] for name, doc in cb.received if name == "descriptor"]
    primary_descriptor_uids = {doc["uid"] for name, doc in cb.received if name == "descriptor"}
    event_descriptor_uids = [doc["descriptor"] for name, doc in cb.received if name == "event"]
    stop_docs = [doc for name, doc in cb.received if name == "stop"]

    assert descriptor_stream_names == ["primary"], "only the primary descriptor should reach the callback"
    assert all(uid in primary_descriptor_uids for uid in event_descriptor_uids), "baseline events must be filtered"
    assert len(stop_docs) == 1, "stop must always pass through"
