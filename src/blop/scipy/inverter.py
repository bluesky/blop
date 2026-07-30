from dataclasses import asdict

from scipy.optimize import OptimizeResult, dual_annealing, minimize, shgo

from blop.scipy.configs import SCP, ScipyCFG
from blop.scipy.optimizer import ScipyOptimizer


class InnerOptimizer():
    def call(self, cost, callback, kws=None) -> ScipyOptimizer.Result | OptimizeResult:
        raise NotImplementedError("Optimizer spec not Provided")


class Optimize(InnerOptimizer):
    """
    Parameter Normalized implementation of scipy Optimize to be passed to a loop inversion.

    derives from inner optimizer protocol class
    """

    def __init__(self, optimizer: SCP, config: ScipyCFG) -> None:
        self.optimizer = optimizer
        self.config = config

    def call(self, cost, callback, kws=None) -> ScipyOptimizer.Result:
        bounds = kws.pop("bounds", self.config.dofs) if kws else self.config.dofs
        x0 = kws.pop("x0", self.config.initial) if kws else self.config.initial
        return minimize(
                    fun=cost,
                    x0=x0,
                    method=self.config.optimizer if self.config.optimizer != SCP.Default else None,
                    bounds=bounds,
                    callback=callback,
                    options=kws,
                )

class DualAnnealing(InnerOptimizer):

    def __init__(self, optimizer: SCP, config: ScipyCFG) -> None:
        self.optimizer = optimizer
        self.config = config

    def call(self, cost, callback, kws=None):
        self.final = dual_annealing(
            func=cost,
            x0=_x,
            bounds=self._bounds,
            callback=dual_callback,
            minimizer_kwargs={"callback": default_callback, "bounds": self._bounds, "options": kws},
        )
