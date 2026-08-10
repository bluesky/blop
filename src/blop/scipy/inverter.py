"""Core Scipy optimizer porting scipy algorithms."""

from collections import OrderedDict
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import Event, Thread
from typing import Any, cast

import numpy as np
from scipy.optimize import OptimizeResult

from blop.protocols import ID_KEY, Optimizer
from blop.scipy.configs import SCP, Objective, ScipyCFG
from blop.scipy.normalizers import InnerOptimizer, ScipyResult


@dataclass
class _Request:
    args: tuple
    future: Future


class InteractiveOptimizer(Optimizer):
    """An optimizer object to supply an interactive interface for the scipy optimizers, with some caveats."""

    def __init__(self, optimizer: InnerOptimizer, config: ScipyCFG | None = None, timeout: int | None = 200):
        self.optimizer = optimizer
        self.session(config=config if config else optimizer.config, timeout=timeout)

    def session(self, config: ScipyCFG, timeout: int | None = None):
        """
        Through path for initialization and stateful reinitialization of optimization.

        derived so that mutiple initializations and lifetimes can be used for optimization.
        Such as the standard ScipyOptimizer(...) call or a following "with"
        """
        self._params: list[str] = [dof.parameter_name for dof in config.dofs]
        self._increment: int = 0
        self._objective: Objective = config.objective
        self.force_resiliance = False  # kinda hidden for now
        self._scale = np.ones(len(config.dofs))
        self._active: dict[int, _Request] = OrderedDict()
        self.intermediate: OptimizeResult | ScipyResult | None = None
        self.final: OptimizeResult | ScipyResult | None = None
        self.SUGGESTION_TIMEOUT = timeout
        self.thread_monitor = Future()
        self.thread_start = Event()

        if config.rescale is not None:
            if isinstance(config.rescale, list):
                self._scale = config.rescale
            else:
                self._scale *= config.rescale

        def cost(x):  # thread safety needs timeout so there is not infinite hang on programs
            """Cooperative thread that defers evaluation of cost call by scipy to the run engine."""
            req = _Request(args=x, future=Future())
            self._active[self._increment] = req
            self.thread_start.set()
            self._increment += 1
            res = req.future.result(timeout=self.SUGGESTION_TIMEOUT)
            if res is None:
                raise ValueError("return value is not present")
            return res

        kw: dict = {}
        self._thread_pool = None
        if config.max_iter is not None:
            if config.optimizer is not SCP.TRUST_CONSTR:
                kw["max_iter"] = config.max_iter
            else:
                kw["maxiter"] = config.max_iter
        if config.eps is not None:
            kw["eps"] = config.eps

        def default_callback(intermediate_result: OptimizeResult):
            if self.intermediate and self.intermediate.fun < intermediate_result.fun:
                return
            self.intermediate = intermediate_result
            self.intermediate.nit = self._increment

        def mini_worker():
            try:
                if config.threads:
                    with ThreadPoolExecutor(max_workers=config.threads) as pool:
                        kw["workers"] = pool.map
                        res = self.optimizer.call(cost, default_callback, kws=kw)
                else:
                    res = self.optimizer.call(cost, default_callback, kws=kw)
                self.thread_monitor.set_result(res)

            except (KeyboardInterrupt, TimeoutError):
                # have to have timeout, so made it that it can be restored to its state on agent auto reboot
                if self.final:
                    ...
                if self.intermediate:
                    self.final = self.intermediate
                else:
                    self.final = ScipyResult(list(self.optimizer.x0), np.nan, nit=self._increment)
                # self.thread_monitor.set_result(self.final)
                return
            except Exception as e:
                self.thread_monitor.set_exception(e)

        self._t = Thread(target=mini_worker, name="optimizer")
        self._t.start()
        self.thread_start.wait(timeout=10)
        return self

    def __enter__(self):
        """Magic convenience to use "with" to better control thread lifetime."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Lifetime threads when using with."""
        self.close()

    def suggest(self, num_points: int | None = None) -> list[dict]:
        """
        Provide a set of points in the input space, to be evaulated next.

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
        try:
            self.final = self.thread_monitor.result(timeout=0.01)
            if not self.force_resiliance:
                print(self.final)
                raise RuntimeError("The optimizer has suspended or reached convergence")
        except TimeoutError:
            ...

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
            cast(Mapping, {self._objective.name: result.fun}),
        ]
        return cart

    def close(self):
        """Clear out futures to allow cleanup of threads."""
        for ind in list(self._active.keys()):
            self._active.pop(ind).future.set_exception(KeyboardInterrupt("Execution has been suspended"))
