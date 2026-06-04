import re
from collections.abc import Sequence
from typing import Any

from xopt import VOCS

from ..ax.dof import DOF, ChoiceDOF, DOFConstraint, RangeDOF
from ..ax.objective import Objective, OutcomeConstraint, ScalarizedObjective
from ..protocols import Sensor

_INEQUALITY_RE = re.compile(r"^\s*(?P<left>.+?)\s*(?P<op><=|>=|<|>)\s*(?P<right>.+?)\s*$")
_SYMBOL_RE = re.compile(r"^[A-Za-z_]\w*$")


def _sensor_name(sensor: Sensor | str) -> str:
    return sensor if isinstance(sensor, str) else sensor.name


def _parse_single_symbol_inequality(expression: str) -> tuple[str, str, float] | None:
    # Accept expressions of the form "name <= value" or "value <= name".
    match = _INEQUALITY_RE.match(expression)
    if match is None:
        return None

    left = match.group("left").strip()
    op = match.group("op")
    right = match.group("right").strip()

    if _SYMBOL_RE.match(left):
        # Canonical case: metric/variable is on the left side.
        try:
            return left, op, float(right)
        except ValueError:
            return None

    if _SYMBOL_RE.match(right):
        # Reversed case: variable is on the right side, so flip inequality direction.
        try:
            value = float(left)
        except ValueError:
            return None

        flip_op = {"<=": ">=", ">=": "<=", "<": ">", ">": "<"}[op]
        return right, flip_op, value

    return None


def _apply_dof_constraints(
    variables: dict[str, list[float] | list[float | int | str | bool]], dof_constraints: Sequence[DOFConstraint]
) -> None:
    # Tighten the variable domains in-place using simple scalar inequality constraints.
    for constraint in dof_constraints:
        parsed = _parse_single_symbol_inequality(constraint.ax_constraint)
        if parsed is None:
            raise ValueError(
                f"Xopt mapping currently supports only single-variable DOF constraints, got: {constraint.ax_constraint!r}."
            )

        name, op, value = parsed
        if name not in variables:
            raise ValueError(f"Unknown variable {name!r} in DOF constraint {constraint.ax_constraint!r}.")

        current = variables[name]
        # Xopt currently receives range constraints only for numeric RangeDOF mappings.
        if not (isinstance(current, list) and len(current) == 2 and all(isinstance(v, (int, float)) for v in current)):
            raise ValueError(
                f"DOF constraint {constraint.ax_constraint!r} targets non-range variable {name!r}; "
                "only RangeDOF constraints are supported."
            )

        lower, upper = float(current[0]), float(current[1])
        # Intersect existing bounds with the new constraint.
        if op in ("<=", "<"):
            upper = min(upper, value)
        else:
            lower = max(lower, value)

        if lower > upper:
            raise ValueError(
                f"DOF constraint {constraint.ax_constraint!r} produces invalid bounds for {name!r}: [{lower}, {upper}]"
            )

        variables[name] = [lower, upper]


def _outcome_constraints_to_vocs_constraints(outcome_constraints: Sequence[OutcomeConstraint]) -> dict[str, list[Any]]:
    # Convert blop constraint expressions into VOCS-style [operator, threshold] constraints.
    constraints: dict[str, list[Any]] = {}
    for constraint in outcome_constraints:
        parsed = _parse_single_symbol_inequality(constraint.ax_constraint)
        if parsed is None:
            raise ValueError(
                f"Xopt mapping currently supports only single-metric outcome constraints, got: {constraint.ax_constraint!r}."
            )

        metric_name, op, value = parsed
        # Map inequality direction to Xopt constraint operator semantics.
        if op in ("<=", "<"):
            constraints[metric_name] = ["LESS_THAN", value]
        else:
            constraints[metric_name] = ["GREATER_THAN", value]

    return constraints


def _objectives_to_vocs_objectives(objectives: Sequence[Objective] | ScalarizedObjective) -> dict[str, str]:
    # ScalarizedObjective is intentionally rejected until a stable translation is defined.
    if isinstance(objectives, ScalarizedObjective):
        raise ValueError(
            "ScalarizedObjective cannot be auto-mapped to Xopt VOCS objectives. "
            "Provide an explicit scalarized metric from your evaluation function and use Objective."
        )

    if not objectives:
        raise ValueError("At least one objective is required to build VOCS.")

    return {objective.name: ("MINIMIZE" if objective.minimize else "MAXIMIZE") for objective in objectives}


def build_vocs(
    *,
    dofs: Sequence[DOF],
    objectives: Sequence[Objective] | ScalarizedObjective,
    sensors: Sequence[Sensor] | None = None,
    dof_constraints: Sequence[DOFConstraint] | None = None,
    outcome_constraints: Sequence[OutcomeConstraint] | None = None,
) -> VOCS:
    """Build an Xopt VOCS object from blop domain objects."""
    # First build raw variable definitions from blop DOFs.
    variables: dict[str, list[float] | list[float | int | str | bool]] = {}

    for dof in dofs:
        if isinstance(dof, RangeDOF):
            variables[dof.parameter_name] = [float(dof.bounds[0]), float(dof.bounds[1])]
        elif isinstance(dof, ChoiceDOF):
            # Keep scope explicit: only numeric choices map cleanly to current VOCS variable model.
            if any(not isinstance(value, (int, float)) for value in dof.values):
                raise ValueError(
                    "Xopt VOCS currently supports numeric variables only. "
                    f"ChoiceDOF {dof.parameter_name!r} has non-numeric values: {dof.values!r}"
                )
            variables[dof.parameter_name] = [float(value) for value in dof.values]
        else:
            raise TypeError(f"Unsupported DOF type for Xopt mapping: {type(dof).__name__}")

    # Apply optional search-space constraints after variables are materialized.
    if dof_constraints:
        _apply_dof_constraints(variables, dof_constraints)

    # Translate objective and outcome-constraint metadata.
    vocs_objectives = _objectives_to_vocs_objectives(objectives)
    vocs_constraints = _outcome_constraints_to_vocs_constraints(outcome_constraints or [])

    # Use sensor names as observables except when already used as optimization outputs.
    observables: list[str] = []
    if sensors:
        reserved_names = set(vocs_objectives) | set(vocs_constraints)
        observables = [name for name in (_sensor_name(sensor) for sensor in sensors) if name not in reserved_names]

    # Build a canonical VOCS object consumed by Xopt generators.
    return VOCS(
        variables=variables,
        objectives=vocs_objectives,
        constraints=vocs_constraints,
        constants={},
        observables=observables,
    )
