import pytest
from ax import ChoiceParameterConfig, RangeParameterConfig

from blop.ax.dof import DOF, ChoiceDOF, DOFConstraint, RangeDOF

from ..conftest import MovableSignal


def test_dof_movable():
    movable = MovableSignal(name="test_movable")
    range_dof = RangeDOF(actuator=movable, bounds=(0, 1), parameter_type="float", step_size=0.1, scaling="linear")
    assert range_dof.parameter_name == "test_movable"
    choice_dof = ChoiceDOF(
        actuator=movable, values=[0, 1, 2, 3, 4, 5], parameter_type="int", is_ordered=True, dependent_parameters=None
    )
    assert choice_dof.parameter_name == "test_movable"


def test_dof_name():
    range_dof = RangeDOF(name="test_name", bounds=(0, 1), parameter_type="float", step_size=0.1, scaling="linear")
    assert range_dof.parameter_name == "test_name"
    choice_dof = ChoiceDOF(
        name="test_name", values=[0, 1, 2, 3, 4, 5], parameter_type="int", is_ordered=True, dependent_parameters=None
    )
    assert choice_dof.parameter_name == "test_name"


def test_dof_invalid():
    movable = MovableSignal(name="test_movable")
    with pytest.raises(ValueError):
        RangeDOF(actuator=movable, name="test_movable", bounds=(0, 1), parameter_type="float")

    with pytest.raises(ValueError):
        ChoiceDOF(actuator=movable, name="test_name", values=[0, 1, 2, 3, 4, 5], parameter_type="int")

    with pytest.raises(TypeError):
        DOF(actuator=movable, name="test_movable")  # type: ignore[arg-type]


def test_range_dof():
    dof1 = RangeDOF(name="test_name", bounds=(0, 1), parameter_type="float", step_size=0.1, scaling="linear")

    assert dof1.to_ax_parameter_config() == RangeParameterConfig(
        name="test_name", bounds=(0, 1), parameter_type="float", step_size=0.1, scaling="linear"
    )

    movable = MovableSignal(name="test_movable")
    dof2 = RangeDOF(actuator=movable, bounds=(0, 1), parameter_type="float", step_size=0.1, scaling="linear")

    assert dof2.to_ax_parameter_config() == RangeParameterConfig(
        name="test_movable", bounds=(0, 1), parameter_type="float", step_size=0.1, scaling="linear"
    )


def test_choice_dof():
    dof1 = ChoiceDOF(
        name="test_name", values=[0, 1, 2, 3, 4, 5], parameter_type="int", is_ordered=True, dependent_parameters=None
    )

    assert dof1.to_ax_parameter_config() == ChoiceParameterConfig(
        name="test_name", values=[0, 1, 2, 3, 4, 5], parameter_type="int", is_ordered=True, dependent_parameters=None
    )

    movable = MovableSignal(name="test_movable")
    dof2 = ChoiceDOF(
        actuator=movable, values=[0, 1, 2, 3, 4, 5], parameter_type="int", is_ordered=True, dependent_parameters=None
    )

    assert dof2.to_ax_parameter_config() == ChoiceParameterConfig(
        name="test_movable", values=[0, 1, 2, 3, 4, 5], parameter_type="int", is_ordered=True, dependent_parameters=None
    )


def test_dof_constraint():
    dof1 = RangeDOF(name="test_dof1", bounds=(0, 1), parameter_type="float", step_size=0.1, scaling="linear")
    dof2 = RangeDOF(name="test_dof2", bounds=(0, 1), parameter_type="float", step_size=0.1, scaling="linear")
    constraint = DOFConstraint(constraint="x1 + x2 <= 10", x1=dof1, x2=dof2)
    assert constraint.ax_constraint == "test_dof1 + test_dof2 <= 10"
    assert str(constraint) == "test_dof1 + test_dof2 <= 10"


def test_invalid_dof_constraint():
    dof1 = RangeDOF(name="test_dof1", bounds=(0, 1), parameter_type="float", step_size=0.1, scaling="linear")
    dof2 = RangeDOF(name="test_dof2", bounds=(0, 1), parameter_type="float", step_size=0.1, scaling="linear")
    with pytest.raises(ValueError):
        DOFConstraint(constraint="x1 + x2 <= 10", x1=dof1, x3=dof2)
