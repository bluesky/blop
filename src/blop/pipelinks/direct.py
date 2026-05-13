from collections.abc import Callable

from bluesky.protocols import Readable

from blop.callbacks.cache import OptimizationCache
from blop.protocols import ID_KEY, EvaluationFunction


def direct_link(stream_store: OptimizationCache, channels: list[str] | list[Readable] | None = None):
    """
    This is a decorator class which converts a data processing function to have a side effect function allowing for
    direct uid interface mode using a local OptimizationCache callback to collect run data and process through single
    sample contexts
    """
    def deco(f: Callable[[dict], dict]):
        class DirectEval(EvaluationFunction):
            def __call__(self, uid, suggestions) -> list[dict]:
                run = stream_store[uid]

                outcomes = []
                for suggestion in suggestions:
                    if suggestion[ID_KEY] not in run:
                        raise KeyError("suggestion has not been processed within this run")
                    context = run[suggestion[ID_KEY]]

                    if channels:
                        output_vector = [ch.name if isinstance(ch, Readable) else ch for ch in channels]
                        measures = {name: context[name] for name in output_vector}
                    else:
                        measures = context

                    outcomes.append(suggestion | f(measures))
                return outcomes

        class link_wrapper:
            Evaluator = DirectEval()
            linked = Evaluator

            def __call__(self, *args, **kwargs):
                return f(*args, **kwargs)

        return link_wrapper()

    return deco
