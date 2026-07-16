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

Set outcome constraints
=======================

This guide will show you how to set outcome constraints. These indicate preferences for specific objectives or metrics to satsify some condition during
optimization.

A surrogate model is built per-constraint to approximate violation of the constraint, so this is considered a *soft* constraint.

For guaranteed constraints, you will have to constrain the DOFs directly using :doc:`DOF constraints <set-dof-constraints>`.

For more information, check out these references:

* `Ax Outcome Constraints documentation <https://ax.dev/docs/recipes/outcome-constraints>`_
* `BoTorch Constraints documentation <https://botorch.org/docs/constraints/>`_

Create DOFs and multiple objectives
-----------------------------------

.. testcode::

    from blop.ax import RangeDOF, Objective

    motor_x = MovableSignal(name="motor_x")
    motor_y = MovableSignal(name="motor_y")
    motor_z = MovableSignal(name="motor_z")


    dofs = [
        RangeDOF(actuator=motor_x, bounds=(0, 1000), parameter_type="float"),
        RangeDOF(actuator=motor_y, bounds=(0, 1000), parameter_type="float"),
        RangeDOF(actuator=motor_z, bounds=(0, 1000), parameter_type="float"),
    ]

    objectives = [
        Objective(name="objective1", minimize=False),
        Objective(name="objective2", minimize=False),
    ]

    def evaluation_function(uid: str, suggestions: list[dict]) -> list[dict]:
        """Replace this with your own evaluation function."""
        outcomes = []
        for suggestion in suggestions:
            outcome = {
                "_id": suggestion["_id"],
                "objective1": 0.1,
                "objective2": 0.2,
            }
            outcomes.append(outcome)
        return outcomes


Set an objective threshold
--------------------------

Since this is a multi-objective optimization, we can set a preference for the first objective to be greater than or equal to 0.5.

.. testcode::

    from blop.ax import OutcomeConstraint

    constraint = OutcomeConstraint("x >= 0.5", x=objectives[0])


Configure an agent with outcome constraints
-------------------------------------------

.. testcode::

    from blop.ax import Agent

    agent = Agent(
        sensors=[],
        dofs=dofs,
        objectives=objectives,
        evaluation_function=evaluation_function,
        outcome_constraints=[constraint],
    )
