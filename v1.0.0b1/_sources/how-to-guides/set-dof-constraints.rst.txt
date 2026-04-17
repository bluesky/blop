.. testsetup::
    
    from unittest.mock import MagicMock
    from typing import Any
    import time

    from bluesky.protocols import NamedMovable, Readable, Status, Hints, HasHints, HasParent
    from tiled.client.container import Container

    class AlwaysSuccessfulStatus(Status):
        def add_callback(self, callback) -> None:
            callback(self)

        def exception(self, timeout = 0.0):
            return None
        
        @property
        def done(self) -> bool:
            return True
        
        @property
        def success(self) -> bool:
            return True

    class ReadableSignal(Readable, HasHints, HasParent):
        def __init__(self, name: str) -> None:
            self._name = name
            self._value = 0.0

        @property
        def name(self) -> str:
            return self._name

        @property
        def hints(self) -> Hints:
            return { 
                "fields": [self._name],
                "dimensions": [],
                "gridding": "rectilinear",
            }
        
        @property
        def parent(self) -> Any | None:
            return None

        def read(self):
            return {
                self._name: { "value": self._value, "timestamp": time.time() }
            }

        def describe(self):
            return {
                self._name: { "source": self._name, "dtype": "number", "shape": [] }
            }

    class MovableSignal(ReadableSignal, NamedMovable):
        def __init__(self, name: str, initial_value: float = 0.0) -> None:
            super().__init__(name)
            self._value: float = initial_value

        def set(self, value: float) -> Status:
            self._value = value
            return AlwaysSuccessfulStatus()

    db = MagicMock(spec=Container)

Set DOF constraints
===================

This guide will show you how to set DOF constraints to refine the search space of your optimization.

These constraints are evaluated by the Ax backend. See the `Ax API documentation <https://ax.readthedocs.io/en/stable/api.html#ax.api.client.Client.configure_experiment>`_ for more information.


Create DOFs and an objective
----------------------------

.. testcode::

    from blop.ax import RangeDOF, Objective

    motor_x = MovableSignal(name="motor_x")
    motor_y = MovableSignal(name="motor_y")
    motor_z = MovableSignal(name="motor_z")

    dof1 = RangeDOF(actuator=motor_x, bounds=(0, 1000), parameter_type="float")
    dof2 = RangeDOF(actuator=motor_y, bounds=(0, 1000), parameter_type="float")
    dof3 = RangeDOF(actuator=motor_z, bounds=(0, 1000), parameter_type="float")

    objective = Objective(name="objective1", minimize=False)

    def evaluation_function(uid: str, suggestions: list[dict]) -> list[dict]:
        """Replace this with your own evaluation function."""
        outcomes = []
        for suggestion in suggestions:
            outcome = {
                "_id": suggestion["_id"],
                "objective1": 0.1,
            }
            outcomes.append(outcome)
        return outcomes


Set a linear constraint
-----------------------

Constraints are specified as strings that are templated and evaluated for you.

.. testcode::

    from blop.ax import DOFConstraint

    constraint = DOFConstraint("5 * x1 + 2 * x2 <= 4 * x3", x1=dof1, x2=dof2, x3=dof3)


Configure an agent with DOF constraints
---------------------------------------

.. testcode::

    from blop.ax import Agent

    agent = Agent(
        sensors=[],
        dofs=[dof1, dof2, dof3],
        objectives=[objective],
        evaluation_function=evaluation_function,
        dof_constraints=[constraint],
    )
