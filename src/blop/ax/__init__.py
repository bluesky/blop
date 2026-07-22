"""Ax backend for optimization."""

try:
    from .agent import Agent as Agent
    from .dof import DOF, ChoiceDOF, DOFConstraint, RangeDOF
    from .objective import Objective, OutcomeConstraint, ScalarizedObjective, to_ax_objective_str
    from .optimizer import AxOptimizer
except ImportError as e:
    raise ImportError("The ax integration requires additional dependencies. Install them with: pip install blop[ax]") from e

__all__ = [
    "Agent",
    "DOF",
    "RangeDOF",
    "ChoiceDOF",
    "DOFConstraint",
    "Objective",
    "OutcomeConstraint",
    "ScalarizedObjective",
    "to_ax_objective_str",
    "AxOptimizer",
]
