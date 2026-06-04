import pytest

from blop.ax.dof import ChoiceDOF, DOFConstraint, RangeDOF
from blop.ax.objective import Objective, OutcomeConstraint, ScalarizedObjective
from blop.tests.conftest import ReadableSignal
from blop.xopt.mapping import build_vocs


def test_build_vocs_maps_basic_objects():
    dof_x = RangeDOF(name="x", bounds=(0.0, 10.0), parameter_type="float")
    dof_mode = ChoiceDOF(name="mode", values=[0, 1], parameter_type="int")
    objective = Objective(name="score", minimize=True)
    outcome_constraint = OutcomeConstraint("s <= 2.5", s=objective)

    vocs = build_vocs(
        dofs=[dof_x, dof_mode],
        objectives=[objective],
        outcome_constraints=[outcome_constraint],
        sensors=[ReadableSignal(name="detector")],
    )

    assert vocs.variables["x"].domain == [0.0, 10.0]
    assert vocs.variables["mode"].domain == [0.0, 1.0]
    assert "score" in vocs.objectives
    assert "score" in vocs.constraints
    assert "detector" in vocs.observables


def test_build_vocs_applies_single_variable_dof_constraint():
    dof_x = RangeDOF(name="x", bounds=(0.0, 10.0), parameter_type="float")
    objective = Objective(name="score", minimize=True)
    dof_constraint = DOFConstraint("x >= 3.0", x=dof_x)

    vocs = build_vocs(
        dofs=[dof_x],
        objectives=[objective],
        dof_constraints=[dof_constraint],
    )

    assert vocs.variables["x"].domain == [3.0, 10.0]


def test_build_vocs_rejects_scalarized_objective_mapping():
    with pytest.raises(ValueError):
        build_vocs(
            dofs=[RangeDOF(name="x", bounds=(0.0, 1.0), parameter_type="float")],
            objectives=ScalarizedObjective("a + b", minimize=True, a="oa", b="ob"),
        )


def test_build_vocs_rejects_multivariable_dof_constraint():
    x = RangeDOF(name="x", bounds=(0.0, 10.0), parameter_type="float")
    y = RangeDOF(name="y", bounds=(0.0, 10.0), parameter_type="float")

    with pytest.raises(ValueError):
        build_vocs(
            dofs=[x, y],
            objectives=[Objective(name="score", minimize=True)],
            dof_constraints=[DOFConstraint("x + y <= 1", x=x, y=y)],
        )
