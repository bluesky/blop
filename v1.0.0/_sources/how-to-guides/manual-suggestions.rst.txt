.. testsetup::
    
    from unittest.mock import MagicMock
    from typing import Any
    import time

    from bluesky.protocols import NamedMovable, Readable, Status, Hints, HasHints, HasParent
    from bluesky.run_engine import RunEngine
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
    RE = RunEngine({})

    sensor = ReadableSignal("signal")
    motor_x = MovableSignal("motor_x")
    motor_y = MovableSignal("motor_y")

    # Mock evaluation function for examples
    def evaluation_function(uid: str, suggestions: list[dict]) -> list[dict]:
        """Mock evaluation function that returns constant outcomes."""
        outcomes = []
        for suggestion in suggestions:
            outcome = {
                "_id": suggestion["_id"],
                "signal": 0.5,
            }
            outcomes.append(outcome)
        return outcomes

Manual Point Injection
======================

This guide shows how to inject custom parameter combinations based on domain knowledge or external sources, alongside optimizer-driven suggestions.

Basic Usage
-----------

To evaluate manually-specified points, use the ``sample_suggestions`` method with parameter combinations (without ``"_id"`` keys). The optimizer will automatically register these trials and incorporate the results into the Bayesian model.

.. testcode::

    from blop.ax import Agent, RangeDOF, Objective
    
    # Configure agent
    agent = Agent(
        sensors=[sensor],
        dofs=[
            RangeDOF(actuator=motor_x, bounds=(-10, 10), parameter_type="float"),
            RangeDOF(actuator=motor_y, bounds=(-10, 10), parameter_type="float"),
        ],
        objectives=[Objective(name="signal", minimize=False)],
        evaluation_function=evaluation_function,
    )
    
    # Define points of interest
    manual_points = [
        {'motor_x': 0.5, 'motor_y': 1.0},  # Center region
        {'motor_x': 0.0, 'motor_y': 0.0},  # Origin
    ]
    
    # Evaluate them
    RE(agent.sample_suggestions(manual_points))

.. testoutput::
   :hide:

   ...

The manual points will be treated just like optimizer suggestions - they'll be tracked, evaluated, and used to improve the model.

Mixed Workflows
---------------

You can combine optimizer suggestions with manual points throughout your optimization:

.. testcode::

    from blop.ax import Agent, RangeDOF, Objective
    
    agent = Agent(
        sensors=[sensor],
        dofs=[
            RangeDOF(actuator=motor_x, bounds=(-10, 10), parameter_type="float"),
            RangeDOF(actuator=motor_y, bounds=(-10, 10), parameter_type="float"),
        ],
        objectives=[Objective(name="signal", minimize=False)],
        evaluation_function=evaluation_function,
    )
    
    # Run optimizer for initial exploration
    RE(agent.optimize(iterations=3))
    
    # Try a manual point based on domain insight
    manual_point = [{'motor_x': 0.75, 'motor_y': 0.25}]
    RE(agent.sample_suggestions(manual_point))
    
    # Continue optimization
    RE(agent.optimize(iterations=3))

.. testoutput::
   :hide:

   ...

The optimizer will incorporate your manual point into its model and use it to inform future suggestions.

Manual Approval Workflow
-------------------------

You can review optimizer suggestions before running them by using ``suggest()`` to get suggestions without acquiring data:

.. testcode::

    from blop.ax import Agent, RangeDOF, Objective
    
    agent = Agent(
        sensors=[sensor],
        dofs=[
            RangeDOF(actuator=motor_x, bounds=(-10, 10), parameter_type="float"),
            RangeDOF(actuator=motor_y, bounds=(-10, 10), parameter_type="float"),
        ],
        objectives=[Objective(name="signal", minimize=False)],
        evaluation_function=evaluation_function,
    )
    
    # Get suggestions without running
    suggestions = agent.suggest(num_points=5)
    
    # Review and filter
    print("Reviewing suggestions:")
    for s in suggestions:
        trial_id = s['_id']
        x = s['motor_x']
        y = s['motor_y']
        print(f"  Trial {trial_id}: x={x:.2f}, y={y:.2f}")
    
    # Only run approved suggestions
    approved = [s for s in suggestions if s['motor_x'] > -5.0]
    
    if approved:
        RE(agent.sample_suggestions(approved))
    else:
        print("No suggestions approved")

.. testoutput::

   Reviewing suggestions:
   ...

This workflow allows you to apply safety checks, domain constraints, or other validation before running trials.

Iterative Refinement
--------------------

A common pattern is to alternate between automated optimization and targeted manual exploration:

.. testcode::

    from blop.ax import Agent, RangeDOF, Objective
    
    agent = Agent(
        sensors=[sensor],
        dofs=[
            RangeDOF(actuator=motor_x, bounds=(-10, 10), parameter_type="float"),
            RangeDOF(actuator=motor_y, bounds=(-10, 10), parameter_type="float"),
        ],
        objectives=[Objective(name="signal", minimize=False)],
        evaluation_function=evaluation_function,
    )
    
    for cycle in range(3):
        # Automated exploration
        RE(agent.optimize(iterations=2, n_points=2))
        
        # Review results and manually probe interesting regions
        # (Look at plots, current best, etc.)
        
        # Try edge cases or special points
        if cycle == 1:
            # After first cycle, check boundaries
            boundary_points = [
                {'motor_x': -10.0, 'motor_y': 0.0},
                {'motor_x': 10.0, 'motor_y': 0.0},
            ]
            RE(agent.sample_suggestions(boundary_points))

.. testoutput::
   :hide:

   ...

See Also
--------

- :meth:`blop.ax.Agent.suggest` - Get optimizer suggestions without running
- :meth:`blop.ax.Agent.sample_suggestions` - Evaluate specific suggestions
- :meth:`blop.ax.Agent.optimize` - Run full optimization loop
- :class:`blop.protocols.CanRegisterSuggestions` - Protocol for manual trial support
