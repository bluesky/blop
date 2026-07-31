"""Normalized SciPy optimizer wrappers used by the cooperative optimization loop."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import OptimizeResult, dual_annealing, minimize, shgo

from blop.scipy.configs import SCP, ScipyCFG


@dataclass
class ScipyResult:
    """Class to unify Optimize Result and other Scipy Results."""

    x: list[float | int]
    fun: float
    nit: int
    status: int = 2


class InnerOptimizer:
    """Protocol for SciPy optimizer wrappers used by the suggest/ingest loop.

    Subclasses adapt optimizer-specific call signatures into a shared
    ``call(cost, callback, kws)`` interface. This keeps optimizer internals
    decoupled from the cooperative optimization loop that requests suggestions,
    evaluates them externally, and ingests outcomes.
    """

    def __init__(self, config: ScipyCFG, base_args: dict | None = None) -> None:
        """Store normalized configuration varaibles.

        Parameters
        ----------
        config : ScipyCFG
            Normalized optimizer configuration, including default bounds,
            initial values, and selected method.
        base_args : dict, optional
            Extra keyword arguments always forwarded to wrapped optimizer.
        """
        self.config = config
        self.base_args = base_args
        self._bounds: list[tuple[Any, Any]] = []
        scale = np.ones(len(config.dofs))

        if config.rescale is not None:
            if isinstance(config.rescale, list):
                scale = config.rescale
            else:
                scale *= config.rescale

        for ind, dof in enumerate(config.dofs):
            self._bounds.append(tuple(np.array(dof.bounds) / scale[ind]))

        self.x0 = np.mean(self._bounds, axis=1)
        if config.initial is not None:
            self.x0 = np.array(config.initial) / scale

    def call(self, cost, callback, kws=None) -> ScipyResult | OptimizeResult:
        """Run the wrapped optimizer.

        Parameters
        ----------
        cost : callable
            Objective function evaluated by the optimizer.
        callback : callable
            Progress callback invoked by the underlying optimizer.
        kws : dict, optional
            Optimizer-specific options and temporary overrides.

        Returns
        -------
        ScipyResult | OptimizeResult
            Result object from the wrapped SciPy optimizer.

        Raises
        ------
        NotImplementedError
            Raised by the base protocol class when no implementation is
            provided.
        """
        raise NotImplementedError("Optimizer implementation not provided")


class Minimize(InnerOptimizer):
    """Normalized wrapper around ``scipy.optimize.minimize``.

    This adapter reads default bounds and initial conditions from ``ScipyCFG``
    and forwards them to ``minimize`` using the common ``InnerOptimizer``
    interface.
    """

    def call(self, cost, callback, kws=None) -> ScipyResult:
        """Execute ``scipy.optimize.minimize`` with normalized defaults.

        Parameters
        ----------
        cost : callable
            Objective function consumed by SciPy.
        callback : callable
            Callback passed directly to ``minimize``.
        kws : dict, optional
            Temporary call-time overrides. ``bounds`` and ``x0`` are extracted
            from this dictionary when present; remaining values are passed as
            ``options``.

        Returns
        -------
        ScipyResult
            SciPy optimization result (runtime type is ``OptimizeResult``).
        """
        bounds = kws.pop("bounds", self._bounds) if kws else self._bounds
        x0 = kws.pop("x0", self.x0) if kws else self.x0
        return minimize(
            fun=cost,
            x0=x0,
            method=self.config.optimizer if self.config.optimizer != SCP.Default else None,
            bounds=bounds,
            callback=callback,
            options=kws,
            **self.base_args if self.base_args else {},
        )


class DualAnnealing(InnerOptimizer):
    """Normalized wrapper around ``scipy.optimize.dual_annealing``.

    ``dual_annealing`` uses a callback signature different from
    ``scipy.optimize.minimize``. This adapter normalizes callback payloads so
    the outer loop can handle intermediate results consistently.
    """

    def __init__(self, config: ScipyCFG, base_args: dict | None = None, inner_args: dict | None = None) -> None:
        """Store normalized configuration for ``dual_annealing``.

        Parameters
        ----------
        config : ScipyCFG
            Normalized optimizer configuration, including bounds and initial
            values.
        base_args : dict, optional
            Extra keyword arguments forwarded directly to
            ``scipy.optimize.dual_annealing``.
        inner_args : dict, optional
            Additional values merged into ``minimizer_kwargs`` for the local
            minimizer stage.
        """
        self.inner_args = inner_args
        super().__init__(config=config, base_args=base_args)

    def dual_callback(self, x, f, context):
        """Convert dual-annealing callback values into a unified result type."""
        return ScipyResult(x, f, -1, context)

    def call(self, cost, callback, kws=None):
        """Execute ``dual_annealing`` with normalized bounds and callbacks.

        Parameters
        ----------
        cost : callable
            Objective function consumed by SciPy.
        callback : callable
            Outer-loop callback expecting a normalized result object.
        kws : dict, optional
            Temporary call-time overrides. ``bounds`` and ``x0`` are extracted
            when present; remaining keys are forwarded as local-minimizer
            ``options``.

        Returns
        -------
        OptimizeResult
            Final SciPy result from ``dual_annealing``.
        """
        bounds = kws.pop("bounds", self._bounds) if kws else self._bounds
        x0 = kws.pop("x0", self.x0) if kws else self.x0
        opt = self.inner_args["options"] if self.inner_args else {}
        return dual_annealing(
            func=cost,
            x0=x0,
            bounds=bounds,
            # Adapt SciPy's (x, f, context) callback to the outer callback
            # contract that expects a normalized result object.
            callback=lambda x, f, c: callback(self.dual_callback(x, f, c)),
            minimizer_kwargs=self.inner_args
            if self.inner_args
            else {} | {"callback": callback, "bounds": bounds, "options": opt | kws if kws else {}},
            **self.base_args if self.base_args else {},
        )


class SHGO(InnerOptimizer):
    """Normalized wrapper around ``scipy.optimize.shgo``.

    This adapter forwards globally optimized search settings while preserving
    the shared ``InnerOptimizer`` call contract.
    """

    def __init__(self, config: ScipyCFG, base_args: dict | None = None, inner_args: dict | None = None) -> None:
        """Store normalized configuration for ``scipy.optimize.shgo``.

        Parameters
        ----------
        config : ScipyCFG
            Normalized optimizer configuration, including bounds.
        base_args : dict, optional
            Extra keyword arguments forwarded directly to ``shgo``.
        inner_args : dict, optional
            Additional values merged into ``minimizer_kwargs`` for the local
            minimizer phase.
        """
        self.inner_args = inner_args
        super().__init__(config=config, base_args=base_args)

    def call(self, cost, callback, kws=None):
        """Execute ``shgo`` with normalized bounds and minimizer options.

        Parameters
        ----------
        cost : callable
            Objective function consumed by SciPy.
        callback : callable
            Callback forwarded to the local minimizer configuration.
        kws : dict, optional
            Temporary call-time overrides. ``bounds`` and ``workers`` are
            extracted when present; remaining keys are forwarded as
            local-minimizer ``options``.

        Returns
        -------
        OptimizeResult
            Final SciPy result from ``shgo``.
        """
        bounds = kws.pop("bounds", self._bounds) if kws else self._bounds
        workers = kws.pop("workers", 1) if kws else 1
        opt = self.inner_args["options"] if self.inner_args else {}
        return shgo(
            func=cost,
            bounds=bounds,
            minimizer_kwargs=self.inner_args
            if self.inner_args
            else {} | {"callback": callback, "bounds": bounds, "options": opt | kws if kws else {}},
            **self.base_args if self.base_args else {},
            workers=workers,
        )
