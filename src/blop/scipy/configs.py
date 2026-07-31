"""Collection of data and configuration objects used by scipy package."""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, cast

from scipy.optimize import Bounds

from blop.protocols import Actuator


@dataclass(frozen=True, kw_only=True, eq=False)
class RangeDOF:
    """
    A degree of freedom that is a continuous range.

    Use this class for continuous parameters that can take any value within
    specified bounds, such as motor positions, voltages, or temperatures.

    Attributes
    ----------
    bounds : tuple[float, float]
        The search domain of the DOF as (lower_bound, upper_bound).
    parameter_type : Literal["float", "int"]
        The data type of the DOF. Use "float" for continuous values or "int" for integer values.
    step_size : float | None, optional
        The step size of the DOF. If provided, the optimizer will only suggest values
        at multiples of this step size.
    scaling : Literal["linear", "log"] | None, optional
        The scaling of the DOF. Use "log" for parameters that span orders of magnitude.

    Examples
    --------
    Define a continuous DOF with a name (for non-actuator parameters):

    >>> from blop.scipy.configs import RangeDOF
    >>> dof = RangeDOF(name="voltage", bounds=(-10.0, 10.0), parameter_type="float")

    Define an integer DOF with a step size:

    >>> dof = RangeDOF(name="num_exposures", bounds=(1, 100), parameter_type="int", step_size=1)

    For examples with actuators, see :doc:`/tutorials/simple-experiment`.
    """

    name: str | None = None
    actuator: Actuator | str | None = None
    bounds: tuple[float, float]
    parameter_type: Literal["float", "int"]
    step_size: float | None = None
    scaling: Literal["linear", "log"] | None = None

    @property
    def parameter_name(self) -> str:
        """The parameter name used internally by Ax."""
        if isinstance(self.actuator, Actuator):
            param_name = self.actuator.name
        elif isinstance(self.actuator, str):
            param_name = self.actuator
        else:
            param_name = cast(str, self.name)
        return param_name

    def to_scipy_bounds(self) -> Bounds:
        """Convert DOF to the Scipy equivalent Bounds."""
        return Bounds(lb=self.bounds[0], ub=self.bounds[1])


@dataclass(frozen=True, kw_only=True)
class Objective:
    """
    An objective to optimize.

    An objective represents a measurable outcome that you want to optimize.
    The optimizer will try to minimize or maximize this outcome based on the
    acquired data and evaluation function.

    Attributes
    ----------
    name : str
        The name of the objective. This must match the key returned by the
        evaluation function for this outcome.
    minimize : bool
        Whether to minimize or maximize the objective. Set to True for minimization
        (e.g., reducing beam width) or False for maximization (e.g., increasing intensity).

    Examples
    --------
    Define an objective to maximize beam intensity:

    >>> from blop.scipy.objective import Objective
    >>> objective = Objective(name="beam_intensity", minimize=False)

    Define an objective to minimize beam width:

    >>> objective = Objective(name="beam_width", minimize=True)
    """

    name: str
    minimize: bool


class SCP(StrEnum):
    """Enumeration of all optimizers currently supported/tested.

    #TODO all commented optimizers require jacobian and are currently suspended in impl until its clear that external
    # gradient sampling is necesary and clearly cross implementable. likely necesary for noisy opts but this is clearly
    # a usage defined addition.
    """

    Default = "L-BFGS-B"

    NELDER_MEAD = "Nelder-Mead"
    PoWELL = "Powell"
    CG = "CG"
    BFGS = "BFGS"
    # Newton_CG = "Newton-CG"
    LBFGS = "L-BFGS-B"
    TNC = "TNC"
    COBYLA = "COBYLA"
    COBYQA = "COBYQA"
    SLSQP = "SLSQP"
    TRUST_CONSTR = "trust-constr"
    # Dogleg = "dogleg"
    # Trust_NCG = "trust-ncg"
    # Trust_Exact = "trust-exact"
    # Trust_Krylov = "trust-krylov"

    DUAL_ANNEALING = "dual annealing"
    SHGO = "SHGO"


@dataclass
class ScipyCFG:
    """
    Configuration dataclass that encompasses the core optimization problem and extra parameters within Scipy.

    Used as the optimizer/generation function is not injectable like in Ax
    """

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
