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

Blop provides a default acquisition plan (:func:`blop.plans.default_acquire`) that handle data acquisition. This plan:

- Uses the **"primary" stream** to store all acquired data
- Includes a default metadata key **blop_suggestions** which contains all of the suggestions (and their identifiers)

When a custom acquisition plan is used, how the data is stored depends on the plan implementation. 

Creating an Evaluation Function
--------------------------------

To access data from Tiled within your evaluation function, create a class that:

1. Accepts a client instance in its ``__init__`` method
2. Implements a ``__call__`` method that retrieves data using the latest run UID
3. Processes the data to compute optimization objectives

Evaluation Function with Tiled
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here's an example evaluation function that reads data from Tiled for where all data is stored in the "primary" stream:

.. testcode::

    from tiled.client.container import Container

    class TiledEvaluation:
        def __init__(self, tiled_client: Container):
            self.tiled_client = tiled_client

        def __call__(self, uid: str, suggestions: list[dict]) -> list[dict]:
            # Access the run data 
            run = self.tiled_client[uid]
            
            # Extract data columns
            motor_x_data = run["primary/motor_x"].read()
            outcomes = []
            for suggestion in suggestions:
                suggestion_id = suggestion["_id"]
                motor_x = motor_x_data[suggestion_id % len(motor_x_data)]
                outcome = {
                    "_id": suggestion["_id"],
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
