from unittest.mock import patch

import pytest
from blop.callbacks.best_effort import BestEffortOptimizationCallback
from blop.plans import OPTIMIZE_RUN_KEY, SAMPLE_SUGGESTIONS_RUN_KEY


@pytest.mark.parametrize(
    "run_key",
    [OPTIMIZE_RUN_KEY, SAMPLE_SUGGESTIONS_RUN_KEY],
)
@patch("blop.callbacks.logger.OptimizationLogger.start")
def test_best_effort_run_key(optimization_logger_start, run_key):
    bec = BestEffortOptimizationCallback()

    bec("start", {"uid": "test123", "run_key": run_key})
    optimization_logger_start.assert_called_once()
