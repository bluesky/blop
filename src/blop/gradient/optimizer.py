from collections import OrderedDict
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
from threading import Thread
from typing import Any, cast

import numpy as np
from scipy.optimize import OptimizeResult, dual_annealing, minimize

from blop.ax.dof import RangeDOF
from blop.ax.objective import Objective
from blop.protocols import ID_KEY, Optimizer


class SCP(StrEnum):
    Default = "Default"
    BFGS = "L-BFGS-B"
    Dual_Annealing = "dual annealing"


@dataclass
class ScipyCFG:
    dofs: Sequence[RangeDOF]
    objective: Objective
    # dof_constraints: Sequence[DOFConstraint] | None = None
    # outcome_constraints: Sequence[OutcomeConstraint] | None = None
    optimizer: SCP = SCP.Default
    initial: Sequence[float] | None = None
    rescale: Sequence[float] | float | None = None
    max_iter: int | None = 100
    eps: float | None = None
    threads: int | None = None


class ScipyOptimizer(Optimizer):
    """
    An optimizer object to supply an interactive interface for the scipy optimizers, with some caveats.
    """

    @dataclass
    class Request:
        args: tuple
        future: Future

    @dataclass
    class Result:
        x: list[float | int]
        fun: float
        nit: int
        status: int = 2

    def __init__(self, config: ScipyCFG, timeout: int | None = 200):
        self.session(config=config, timeout=timeout)

    def session(self, config: ScipyCFG, timeout: int | None = None):
        self._params: list[str] = []
        self._bounds: list[tuple[Any, Any]] = []
        self._increment: int = 0
        self._objective: Objective = config.objective
        self.force_resiliance = False  # kinda hidden for now
        self._scale = np.ones(len(config.dofs))
        self._active: dict[int, ScipyOptimizer.Request] = OrderedDict()
        self.intermediate: OptimizeResult | ScipyOptimizer.Result | None = None
        self.final: OptimizeResult | ScipyOptimizer.Result | None = None
        self.SUGGESTION_TIMEOUT = timeout

        if config.rescale is not None:
            if isinstance(config.rescale, list):
                self._scale = config.rescale
            else:
                self._scale *= config.rescale

        for ind, dof in enumerate(config.dofs):
            self._params.append(dof.parameter_name)
            self._bounds.append(tuple(np.array(dof.bounds) / self._scale[ind]))

        _x = np.mean(self._bounds, axis=1)
        if config.initial is not None:
            _x = np.array(config.initial) / self._scale

        def cost(x):  # thread safety needs timeout so there is not infinite hang on programs
            """
            simple cooperative thread that defers evaluation of cost call by scipy to the run engine
            """
            print("pushing to request queue")
            req = self.Request(args=x, future=Future())
            self._active[self._increment] = req
            self._increment += 1
            res = req.future.result(timeout=self.SUGGESTION_TIMEOUT)
            print(f"recovered result {res}")
            if res is None:
                raise ValueError("return value is not present")
            return res

        kw = {}
        self._thread_pool = None
        if config.optimizer in (SCP.Default, SCP.BFGS):
            if config.max_iter is not None:
                kw["max_iter"] = config.max_iter
            if config.eps is not None:
                kw["eps"] = config.eps

            def default_callback(intermediate_result: OptimizeResult):
                self.intermediate = intermediate_result

            def call(kws=None):
                self.final = minimize(
                    fun=cost,
                    x0=_x,
                    method=config.optimizer if config.optimizer != SCP.Default else None,
                    bounds=self._bounds,
                    callback=default_callback,
                    options=kws,
                )

        elif config.optimizer in (SCP.Dual_Annealing):

            def dual_callback(x, f, context):
                self.intermediate = self.Result(x, f, self._increment, context)

            def call(kws=None):
                self.final = dual_annealing(
                    func=cost,
                    x0=_x,
                    bounds=self._bounds,
                    callback=dual_callback,
                    minimizer_kwargs=kws,
                )

        else:
            raise NotImplementedError("")

        def mini_worker():
            try:
                if config.threads:
                    with ThreadPoolExecutor(max_workers=config.threads) as pool:
                        kw["workers"] = pool.map
                        print(f"creating {config.threads} workers with:{kw}")
                        call(kws=kw)
                else:
                    call(kws=kw)
            except (KeyboardInterrupt, TimeoutError):
                # have to have timeout, made it so that it can be restored to its state on agent auto reboot
                if self.final:
                    return
                if self.intermediate:
                    self.final = self.intermediate
                else:
                    self.final = self.Result(list(_x), np.nan, nit=self._increment)

        self._t = Thread(target=mini_worker, name="optimizer")
        self._t.start()
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def suggest(self, num_points: int | None = None) -> list[dict]:
        """
        Returns a set of points in the input space, to be evaulated next.

        The "_id" key is optional and can be used to identify suggested trials for later evaluation
        and ingestion.

        Parameters
        ----------
        num_points : int | None, optional
            The number of points to suggest. If not provided, will default to 1.

        Returns
        -------
        list[dict]
            A list of dictionaries, each containing a parameterization of a point to evaluate next.
            Each dictionary must contain a unique "_id" key to identify each parameterization.
        """
        if self.final is not None:
            vector = [x_n * s for s, x_n in zip(self._scale, self.final.x, strict=True)]
            suggestion = dict(zip(self._params, vector, strict=True))
            suggestion[ID_KEY] = self.final.nit
            return [suggestion]

        suggestions = []
        for id in list(self._active.keys())[: num_points if num_points is not None else 1]:
            x = self._active[id].args
            vector = [x_n * s for s, x_n in zip(self._scale, x, strict=True)]

            suggestion = dict(zip(self._params, vector, strict=True))
            suggestion[ID_KEY] = id
            suggestions.append(suggestion)
        print(f"returning {len(suggestions)} suggestions of {len(self._active.keys())} available")
        return suggestions

    def ingest(self, points: list[dict]) -> None:
        """
        Ingest a set of points into the experiment. Either from previously suggested points or from an external source.

        The "_id" key is optional.

        Parameters
        ----------
        points : list[dict]
            A list of dictionaries, each containing the outcomes of each suggested parameterization.
        """
        for res in points:
            y = res[self._objective.name]
            if res[ID_KEY] not in self._active:
                if not self.force_resiliance:
                    raise ValueError("optimizer did not expect to receive an update")
                continue
            self._active.pop(res[ID_KEY]).future.set_result(y)

    def get_best_points(self) -> list[tuple[Any, Mapping, Mapping]]:
        """
        Get a list of the optimal point found during optimization.

        Returns
        -------
        list[tuple[int, TParameterization, TOutcome]]
            Each element in the list is a tuple of:
              - trial index (int)
              - parameter values (dict)
              - metric values (dict, where values may be (value, sem) tuples)

        See Also
        --------
        navigate_to_best : Plan stub to move actuators to a best point.
        """
        result = self.intermediate
        if self.final is not None:
            result = self.final
        if (result is None) or (self._objective is None):
            raise ValueError("no optimization epoch has been recorded")

        vector = [x_n * s for s, x_n in zip(self._scale, result.x, strict=True)]
        cart = [
            result.nit - 1,
            cast(Mapping, dict(zip(self._params, vector, strict=True))),
            cast(Mapping, {self._objective: result.fun}),
        ]
        return cart

    def close(self):
        for fut in self._active.values():
            fut.future.set_exception(KeyboardInterrupt("Execution has been suspended"))
