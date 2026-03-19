from bluesky.callbacks import CallbackBase
from event_model import RunRouter

from ..plans import OPTIMIZE_RUN_KEY, SAMPLE_SUGGESTIONS_RUN_KEY
from .logger import OptimizationLogger


class BestEffortOptimizationCallback:
    """Best effort callback for displaying optimization information."""

    def __init__(
        self,
        stdout: bool = True,
    ) -> None:
        self._run_router = RunRouter([self._factory])
        self._callbacks = self._setup_callbacks(stdout)

    def _setup_callbacks(self, stdout: bool = True) -> list[CallbackBase]:
        callbacks = []
        if stdout:
            callbacks.append(OptimizationLogger())
        return callbacks

    def _factory(self, name, doc):
        if name == "start" and (doc["run_key"] == OPTIMIZE_RUN_KEY or doc["run_key"] == SAMPLE_SUGGESTIONS_RUN_KEY):
            return self._callbacks, []
        return [], []

    def __call__(self, name, doc):
        self._run_router(name, doc)
