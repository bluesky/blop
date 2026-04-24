import re
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

from ax import ChoiceParameterConfig, RangeParameterConfig
from ax.api.types import TParameterValue

from ..protocols import Actuator


@dataclass(frozen=True, kw_only=True)
class DOF(ABC):
    """
    Base class for a degree of freedom (DOF) to optimize.

    A DOF represents a controllable input parameter in the optimization problem.
    DOFs define the search space that the optimizer explores to find optimal
    solutions. Use :class:`RangeDOF` for continuous parameters or :class:`ChoiceDOF`
    for discrete parameters.

    Attributes
    ----------
    name : str | None
        The name of the DOF. Provide a name if the DOF is not an actuator.
    actuator : Actuator | str | None
        The actuator or its name to use for the DOF. Provide an actuator if the DOF is controllable by Bluesky.

    Notes
    -----
    Either ``name`` or ``actuator`` must be provided, but not both. If ``actuator`` is
    provided, the DOF will be associated with a Bluesky-controllable device and will
    automatically move during acquisition. If only ``name`` is provided, the DOF
    represents a parameter that is controlled externally.

    See Also
    --------
    blop.protocols.Actuator : The protocol for actuators.
    RangeDOF : For continuous parameters with bounds.
    ChoiceDOF : For discrete parameters with specific values.
    """

    name: str | None = None
    actuator: Actuator | str | None = None

    def __post_init__(self) -> None:
        if not (bool(self.name) ^ bool(self.actuator)):
            raise ValueError("Either name or actuator must be provided, but not both or neither.")

    @property
    def parameter_name(self) -> str:
        if isinstance(self.actuator, Actuator):
            param_name = self.actuator.name
        elif isinstance(self.actuator, str):
            param_name = self.actuator
        else:
            param_name = cast(str, self.name)
        return param_name

    @abstractmethod
    def to_ax_parameter_config(self) -> RangeParameterConfig | ChoiceParameterConfig: ...


@dataclass(frozen=True, kw_only=True)
class RangeDOF(DOF):
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

    >>> from blop.ax.dof import RangeDOF
    >>> dof = RangeDOF(name="voltage", bounds=(-10.0, 10.0), parameter_type="float")

    Define an integer DOF with a step size:

    >>> dof = RangeDOF(name="num_exposures", bounds=(1, 100), parameter_type="int", step_size=1)

    For examples with actuators, see :doc:`/tutorials/simple-experiment`.
    """

    bounds: tuple[float, float]
    parameter_type: Literal["float", "int"]
    step_size: float | None = None
    scaling: Literal["linear", "log"] | None = None

    def to_ax_parameter_config(self) -> RangeParameterConfig:
        """
        Convert the DOF to the Ax parameter configuration equivalent.

        Returns
        -------
        RangeParameterConfig
            The Ax parameter configuration for this DOF.
        """
        return RangeParameterConfig(
            name=self.parameter_name,
            bounds=self.bounds,
            parameter_type=self.parameter_type,
            step_size=self.step_size,
            scaling=self.scaling,
        )


@dataclass(frozen=True, kw_only=True)
class ChoiceDOF(DOF):
    """
    A degree of freedom that is a discrete choice.

    Use this class for categorical or ordinal parameters that can only take on
    specific discrete values, such as filter selections, detector modes, or
    beam energies from a fixed set.

    Attributes
    ----------
    values : list[float] | list[int] | list[str] | list[bool]
        The possible discrete values of the DOF.
    parameter_type : Literal["float", "int", "str", "bool"]
        The data type of the DOF.
    is_ordered : bool | None, optional
        Whether the values are ordered. If not provided, it will be inferred from the values.
        Set to True for ordinal parameters (e.g., ["low", "medium", "high"]).
    dependent_parameters : Mapping[TParameterValue, Sequence[str]] | None, optional
        Specify which other DOFs are active dependent on specific values of this DOF.

    Examples
    --------
    Define a choice DOF for selecting a filter:

    >>> from blop.ax.dof import ChoiceDOF
    >>> dof = ChoiceDOF(name="filter", values=["none", "Al", "Si"], parameter_type="str")

    Define an ordered choice DOF:

    >>> dof = ChoiceDOF(name="power", values=[1, 2, 5, 10], parameter_type="int", is_ordered=True)
    """

    values: list[float] | list[int] | list[str] | list[bool]
    parameter_type: Literal["float", "int", "str", "bool"]
    is_ordered: bool | None = None
    dependent_parameters: Mapping[TParameterValue, Sequence[str]] | None = None

    def to_ax_parameter_config(self) -> ChoiceParameterConfig:
        """
        Convert the DOF to the Ax parameter configuration equivalent.

        Returns
        -------
        ChoiceParameterConfig
            The Ax parameter configuration for this DOF.
        """
        return ChoiceParameterConfig(
            name=self.parameter_name,
            values=self.values,
            parameter_type=self.parameter_type,
            is_ordered=self.is_ordered,
            dependent_parameters=self.dependent_parameters,
        )


class DOFConstraint:
    """
    A constraint on DOFs to refine the search space.

    Use DOF constraints to exclude regions of the search space that are invalid,
    unsafe, or uninteresting. Constraints are expressed as inequality expressions
    involving DOF parameters.

    Parameters
    ----------
    constraint : str
        The constraint expression to evaluate. Must be a valid inequality using
        operators like <=, >=, <, >. Variable names in the expression are mapped
        to DOFs via keyword arguments.
    **dofs : DOF
        Keyword arguments mapping variables in the constraint to DOFs.

    Examples
    --------
    Define a constraint that ensures the sum of two DOFs is less than a value:

    >>> from blop.ax.dof import DOFConstraint, RangeDOF
    >>> x_dof = RangeDOF(name="x", bounds=(0, 10), parameter_type="float")
    >>> y_dof = RangeDOF(name="y", bounds=(0, 10), parameter_type="float")
    >>> constraint = DOFConstraint("x + y <= 12", x=x_dof, y=y_dof)

    Notes
    -----
    The variable names used in the constraint expression (e.g., "x", "y") are
    arbitrary and do not need to match the DOF parameter names. They are mapped
    to DOFs via the keyword arguments for readability.
    """

    def __init__(self, constraint: str, **dofs: DOF) -> None:
        self._constraint = constraint
        self._dofs: dict[str, DOF] = dofs
        self._validate_dofs()

    def _validate_dofs(self) -> None:
        if not self._dofs:
            raise ValueError(
                "DOFConstraint requires at least one DOF to be specified.\n"
                "Use keyword arguments to map template variables to DOFs:\n"
                "  DOFConstraint('x + y <= 12', x=dof_x, y=dof_y)\n\n"
                "The variable names (x, y) are your choice and make the constraint readable."
            )
        invalidated: list[tuple[str, DOF]] = []
        for name, dof in self._dofs.items():
            if name not in self._constraint:
                invalidated.append((name, dof))

        if len(invalidated) > 0:
            msg = (
                "The following DOFs did not have matching names in the constraint "
                f"'{self._constraint}': {', '.join([f'{name}={dof.parameter_name}' for name, dof in invalidated])}"
            )
            raise ValueError(msg)

    @property
    def ax_constraint(self) -> str:
        """
        Convert the constraint to a string that can be used by Ax.

        Returns
        -------
        str
            The constraint expression with DOF names substituted.
        """
        template = self._constraint
        for key in self._dofs.keys():
            template = re.sub(f"\\b{key}\\b", f"{{{key}}}", template)
        return template.format(**{key: dof.parameter_name for key, dof in self._dofs.items()})

    def __str__(self) -> str:
        return self.ax_constraint

    def __repr__(self) -> str:
        dofs_str = ", ".join(f"{name}={dof.parameter_name}" for name, dof in self._dofs.items())
        return f"DOFConstraint('{self._constraint}', {dofs_str})"
