import re
from collections.abc import Sequence
from dataclasses import dataclass

from ax.api.protocols import IMetric


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

    >>> from blop.ax.objective import Objective
    >>> objective = Objective(name="beam_intensity", minimize=False)

    Define an objective to minimize beam width:

    >>> objective = Objective(name="beam_width", minimize=True)
    """

    name: str
    minimize: bool

    @property
    def ax_expression(self) -> str:
        """
        Convert the objective to a string that can be used by Ax.

        Returns
        -------
        str
            The expression with the objective name, negated if minimizing.
        """
        return f"-{self.name}" if self.minimize else self.name


class ScalarizedObjective:
    """
    A scalarized objective is a weighted sum of other objectives.

    Use this to combine multiple objectives into a single optimization target
    when you want to optimize a weighted combination of outcomes. This is useful
    for multi-objective optimization where you have clear trade-off preferences.

    Parameters
    ----------
    expression : str
        The expression to evaluate containing the objectives.
    minimize : bool
        Whether to minimize or maximize the expression.
    **objective_names : str
        Keyword arguments mapping variables in the expression to objective names.

    Examples
    --------
    Create a scalarized objective for minimizing a weighted sum:

    >>> from blop.ax.objective import ScalarizedObjective
    >>> scalarized_obj = ScalarizedObjective(
    ...     expression="x + 2 * y",
    ...     minimize=True,
    ...     x="objective1",
    ...     y="objective2",
    ... )
    >>> print(scalarized_obj)
    -(objective1 + 2 * objective2)

    Notes
    -----
    The variable names used in the expression (e.g., "intensity", "width") are
    arbitrary and do not need to match the objective names. They are mapped to
    objectives via the keyword arguments for readability.
    """

    def __init__(self, expression: str, *, minimize: bool, **objective_names: str):
        self._expression = expression
        self._objective_names = objective_names
        self._minimize = minimize

        if not self._objective_names:
            raise ValueError("ScalarizedObjective requires at least one objective.")

        invalidated = [key for key in self._objective_names if key not in self._expression]
        if invalidated:
            raise ValueError(f"Objectives {invalidated} not found in expression '{self._expression}'.")

    @property
    def ax_expression(self) -> str:
        """
        Convert the scalarized objective to a string that can be used by Ax.

        Returns
        -------
        str
            The expression with objective names substituted, negated if minimizing.
        """
        template = self._expression
        for key in self._objective_names:
            template = re.sub(f"\\b{key}\\b", f"{{{key}}}", template)

        objective_str = template.format(**self._objective_names)
        return f"-({objective_str})" if self._minimize else objective_str

    def __repr__(self) -> str:
        return (
            f"ScalarizedObjective('{self._expression}', minimize={self._minimize}, "
            f"{', '.join([f'{k}={v}' for k, v in self._objective_names.items()])})"
        )

    def __str__(self) -> str:
        return self.ax_expression


class OutcomeConstraint:
    """
    A constraint on an outcome of a suggestion.

    Outcome constraints guide the optimizer to prefer solutions that satisfy certain
    conditions on the measured outcomes. This is a *soft* constraint, meaning that
    the constraint may be violated during exploration but will be increasingly
    satisfied as optimization progresses.

    Parameters
    ----------
    constraint : str
        The constraint expression to evaluate. Must be a valid inequality using
        operators like <=, >=, <, >. Variable names in the expression are mapped
        to outcomes via keyword arguments.
    **outcomes : Objective | IMetric
        Keyword arguments mapping variables in the expression to objectives or metrics.

    Examples
    --------
    Constrain an objective to be below a threshold:

    >>> from blop.ax.objective import Objective, OutcomeConstraint
    >>> temp_obj = Objective(name="temperature", minimize=True)
    >>> constraint = OutcomeConstraint("temp <= 100", temp=temp_obj)
    >>> print(constraint)
    temperature <= 100

    For complete examples with multiple constraints, see :doc:`/how-to-guides/set-outcome-constraints`.

    Notes
    -----
    The variable names used in the constraint expression (e.g., "temp", "i", "w")
    are arbitrary and do not need to match the objective names. They are mapped
    to objectives via the keyword arguments for readability.

    Outcome constraints differ from DOF constraints in that they constrain the
    measured outcomes rather than the input parameters.
    """

    def __init__(self, constraint: str, **outcomes: Objective | IMetric):
        self._constraint = constraint
        self._outcomes = outcomes
        self._validate_outcomes()

    def _validate_outcomes(self) -> None:
        if not self._outcomes:
            raise ValueError("OutcomeConstraint requires at least one outcome.")

        invalidated = [name for name in self._outcomes if name not in self._constraint]
        if invalidated:
            raise ValueError(f"Outcomes {invalidated} not found in constraint '{self._constraint}'.")

    @property
    def ax_constraint(self) -> str:
        """
        Convert the constraint to a string that can be used by Ax.

        Returns
        -------
        str
            The constraint expression with objective names substituted.
        """
        template = self._constraint
        for key in self._outcomes:
            template = re.sub(f"\\b{key}\\b", f"{{{key}}}", template)

        return template.format(**{key: outcome.name for key, outcome in self._outcomes.items()})

    def __repr__(self) -> str:
        outcomes_str = ", ".join(f"{k}={v.name}" for k, v in self._outcomes.items())
        return f"OutcomeConstraint('{self._constraint}', {outcomes_str})"

    def __str__(self) -> str:
        return self.ax_constraint


def to_ax_objective_str(objectives: Sequence[Objective] | ScalarizedObjective) -> str:
    """
    Convert a list of objectives to a string that can be used by Ax.

    This is a utility function used internally to format objectives for Ax's API.
    Minimized objectives are prefixed with a minus sign.

    Parameters
    ----------
    objectives : Sequence[Objective] | ScalarizedObjective
        The objectives or single scalarized objective to convert to a string.

    Returns
    -------
    str
        The string representation of the objectives usable by Ax.

    Examples
    --------
    >>> from blop.ax.objective import Objective, to_ax_objective_str
    >>> objectives = [
    ...     Objective(name="intensity", minimize=False),
    ...     Objective(name="width", minimize=True)
    ... ]
    >>> to_ax_objective_str(objectives)
    'intensity, -width'

    >>> from blop.ax.objective import ScalarizedObjective, to_ax_objective_str
    >>> objective = ScalarizedObjective(expression="2 * x + 3 * y", minimize=False, x="obj1", y="obj2")
    >>> to_ax_objective_str(objective)
    2 * obj1 + 3 * obj2
    """
    return (
        objectives.ax_expression
        if isinstance(objectives, ScalarizedObjective)
        else ", ".join([o.ax_expression for o in objectives])
    )
