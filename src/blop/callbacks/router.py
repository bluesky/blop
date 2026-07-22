"""Router for filtering out non-optimization Bluesky runs."""

from bluesky.callbacks import CallbackBase
from event_model import RunRouter

from ..plans import OPTIMIZE_RUN_KEY, SAMPLE_SUGGESTIONS_RUN_KEY
from .utils import _PrimaryStreamFilter


class OptimizationCallbackRouter:
    """Routes documents from optimization runs to a list of callbacks.

    Acts as a Bluesky callback that filters documents by ``run_key``,
    forwarding only those from ``"optimize"`` or ``"sample_suggestions"``
    runs to the registered callbacks.

    The router holds a *reference* to the provided list, so external
    mutations (e.g. ``callbacks.append(cb)``) take effect on the next run.

    Parameters
    ----------
    callbacks : list[CallbackBase]
        A mutable list of callback instances. The router reads from this
        list at the start of each matching run.
    """

    def __init__(self, callbacks: list[CallbackBase]) -> None:
        self._run_router = RunRouter([self._factory])
        self._callbacks = callbacks

    def _factory(self, name, doc):
        if name == "start" and doc.get("run_key") in (OPTIMIZE_RUN_KEY, SAMPLE_SUGGESTIONS_RUN_KEY):
            return [_PrimaryStreamFilter(cb) for cb in self._callbacks], []
        return [], []

    def __call__(self, name, doc):
        """Pass the document along to the run router."""
        self._run_router(name, doc)
