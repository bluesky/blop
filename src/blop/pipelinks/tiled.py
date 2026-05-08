from collections.abc import Callable

from tiled.client.container import Container

from blop.protocols import EvaluationFunction, Readable


def tiled_link(
    tiled_client: Container,
    channels: list[str] | list[Readable],
    alt: dict[str, str] | None = None,
    static: dict[str, str] | None = None,
):
    """
    This is a decorator class which converts a data processing function to have a side effect function allowing for
    direct interface mode from a tiled uid servicing client

    Parameters
    ----------
    tiled_client : Container
        the uid servicing tiled container for which the runengine is to record to
    channels : list[str] | list[Readable]
        These are all of your standard detectors/ measures of which RunEngine is familiar with (implement read interface)
        RunEngine puts these into the 'primary' data stream by name. Implicit index by sample point required
    alt : dict[str, str] | None
        An alternate indexing for streams which are labeled/ seperated from the primary stream. Implicit sampling index
        like the primary stream is required (ie if sample x reports at primary index n then alt must also have x at index n)
    static : dict[str, str] | None
        this is for static channels that are sampled only once per run like a background stream or data like start time.
        data is accessed once per set of samples and passed to each evaluation as context.
        ( Can be used as a channel of last resort for when implicit indexing doesn't work with your data scheme.
        it will dump the entire sampled data structure so beware of data size)

    Decorated Return
    -------
    Wrapper >> f(dict[suggestion context]) -> dict[results] | f.linked(uid, suggestions) -> list[dict[results]]

    See Also
    --------
    tiled_wrap : a possibly more pythonic interface for directly generating and returning a tiled linked evaluation function
    """

    def deco(f: Callable[[dict], dict]):
        class TiledEval(EvaluationFunction):
            def __call__(self, uid: str, suggestions: list[dict]) -> list[dict]:
                run = tiled_client[uid]
                r_sugg = run.start["blop_suggestions"]

                if len(r_sugg) != len(suggestions):
                    print(
                        '''!!! Bluesky has not completed all sent suggestions.
                         If expected, be sure to manage failed and abandoned trials !!!'''
                    )
                chan = [(dev if not isinstance(dev, Readable) else dev.name) for dev in channels]
                data = {ch: run["primary/" + ch] for ch in chan}
                if alt:
                    data.update({k: run[v] for k, v in alt.items()})
                post = {k: run[v].read() for k, v in static.items()} if static else {}

                outcomes = []
                for index, suggestion in enumerate(r_sugg):
                    sample = {k: v[index] for k, v in data.items()} | suggestion | post
                    try:
                        outcomes.append({**f(sample), "_id": suggestion["_id"]})
                    except Exception as e:
                        raise RuntimeError(f"Error evaluating sample {suggestion['_id']}") from e
                return outcomes

        class link_wrapper:
            Evaluator = TiledEval()
            linked = Evaluator

            def __call__(self, *args, **kwargs):
                return f(*args, **kwargs)

        return link_wrapper()

    return deco


def tiled_wrap(
    f: Callable[[dict], dict],
    tiled_client: Container,
    channels: list[str] | list[Readable],
    alt: dict[str, str] | None = None,
    static: dict[str, str] | None = None,
):
    """
    a possibly more pythonic interface for directly generating and returning a tiled linked evaluation function

    Parameters
    ----------
    f : Callable[[dict], dict]
        a single sample context only cost function that returns a dict containing all objective values for a sample.
        the provided function shall receive all the data relevent to its sample context described by channels, alt, static,
        and its suggestion as a dictionary keyed by the provided names.
    tiled_client : Container
        the uid servicing tiled container for which the runengine is to record to
    channels : list[str] | list[Readable]
        These are all of your standard detectors/ measures of which RunEngine is familiar with (implement read interface)
        RunEngine puts these into the 'primary' data stream by name. Implicit index by sample point required
    alt : dict[str, str] | None
        An alternate indexing for streams which are labeled/ seperated from the primary stream. Implicit sampling index
        like the primary stream is required (ie if sample x reports at primary index n then alt must also have x at index n)
    static : dict[str, str] | None
        this is for static channels that are sampled only once per run like a background stream or data like start time.
        data is accessed once per set of samples and passed to each evaluation as context.
        ( Can be used as a channel of last resort for when implicit indexing doesn't work with your data scheme.
        it will dump the entire sampled data structure so beware of data size)
    """
    return (tiled_link(tiled_client=tiled_client, channels=channels, alt=alt, static=static)(f)).linked
