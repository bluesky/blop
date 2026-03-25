from unittest.mock import MagicMock

from rich.console import Console
from event_model import RunStart

from blop.callbacks.logger import OptimizationLogger


def test_optimization_logger_custom_console():
    console_mock = MagicMock(spec=Console)
    logger = OptimizationLogger(console=console_mock)
    logger.start(RunStart(uid="123"))

    assert console_mock.print.call_count >= 1
