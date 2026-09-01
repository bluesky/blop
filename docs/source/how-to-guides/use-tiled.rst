.. testsetup::

    from bluesky.protocols import NamedMovable, Readable, Status, Hints, HasHints, HasParent, Any
    import time

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
    
    motor_x = MovableSignal(name="motor_x")


Tiled with Blop
================================
This guide explains how we can use Tiled for data storage and retrieval with Blop.

Setting Up Data Access
-----------------------

To access the data for optimization, you have to connect to a Tiled server instance:

**Tiled:**

.. testcode::

    from bluesky.run_engine import RunEngine
    from bluesky_tiled_plugins import TiledWriter
    from tiled.client import from_uri
    from tiled.server import SimpleTiledServer

    server = SimpleTiledServer()
    tiled_client = from_uri(server.uri)
    tiled_writer = TiledWriter(tiled_client)
    RE = RunEngine({})
    RE.subscribe(tiled_writer)


Data Storage with Blop's Default Plans
---------------------------------------

Blop provides a default acquisition plan (:func:`blop.plans.default_acquire`) that handles data acquisition. This plan:

- Uses the **"primary" stream** to store all acquired data
- Includes **blop_acquisition_order** metadata containing suggestion IDs in actual acquired-row order
- Includes **blop_suggestions** metadata containing the routed suggestions for backwards compatibility

The evaluator's ``suggestions`` sequence is optional context. Do not use its
positions to align primary rows or infer an optimizer generation order; use
``blop_acquisition_order`` instead.

When a custom acquisition plan is used, how the data is stored depends on the plan implementation.

Creating an Evaluation Function
--------------------------------

To access data from Tiled within your evaluation function, create a class that:

1. Accepts a client instance in its ``__init__`` method
2. Accepts the hashable acquisition identifier and validates that this default-acquisition example received a string run UID
3. Processes the data to compute optimization objectives

Evaluation Function with Tiled
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here's an example evaluation function that reads data from Tiled for where all data is stored in the "primary" stream:

.. testcode::

    from collections.abc import Hashable, Mapping, Sequence

    from tiled.client.container import Container

    class TiledEvaluation:
        def __init__(self, tiled_client: Container):
            self.tiled_client = tiled_client

        def __call__(self, uid: Hashable, suggestions: Sequence[Mapping]) -> Sequence[Mapping]:
            if not isinstance(uid, str):
                raise TypeError(f"TiledEvaluation requires a Bluesky run UID string, got {uid!r}")
            run = self.tiled_client[uid]
            acquisition_order = run.start["blop_acquisition_order"]

            # These IDs, not positions in suggestions, align the primary rows.
            # Extract data columns
            motor_x_data = run["primary/motor_x"].read()
            outcomes = []
            for index, suggestion_id in enumerate(acquisition_order):
                motor_x = motor_x_data[index]
                outcome = {
                    "_id": suggestion_id,
                    "objective1": 0.1 * motor_x,
                }
                outcomes.append(outcome)
            return outcomes
            
Configure an agent
------------------

.. testcode::

    from blop.ax import RangeDOF, Agent, Objective

    dof1 = RangeDOF(actuator=motor_x, bounds=(0, 1000), parameter_type="float")

    objective = Objective(name="objective1", minimize=False) 

    # Add motor_x as a sensor so it gets read and stored in Tiled
    agent = Agent(
        sensors=[motor_x],
        dofs=[dof1],
        objectives=[objective],
        evaluation_function=TiledEvaluation(tiled_client=tiled_client),
    )
    RE(agent.optimize())
    server.close()

.. testoutput::
   :hide:

   ...
