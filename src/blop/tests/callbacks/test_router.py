from unittest.mock import MagicMock

import pytest
from bluesky.callbacks import CallbackBase

from blop.callbacks.router import OptimizationCallbackRouter
from blop.plans import OPTIMIZE_RUN_KEY, SAMPLE_SUGGESTIONS_RUN_KEY


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
