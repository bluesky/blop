.. testsetup::

    from typing import Any
    import time

    from bluesky.protocols import NamedMovable, Readable, Status, Hints, HasHints, HasParent
    from bluesky.run_engine import RunEngine

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
                self._name: {"value": self._value, "timestamp": time.time()}
            }

        def describe(self):
            return {
                self._name: {"source": self._name, "dtype": "number", "shape": []}
            }

    class MovableSignal(ReadableSignal, NamedMovable):
        def __init__(self, name: str, initial_value: float = 0.0) -> None:
            super().__init__(name)
            self._value: float = initial_value

        def set(self, value: float) -> Status:
            self._value = value
            return AlwaysSuccessfulStatus()

    RE = RunEngine({})
    motor_x = MovableSignal("motor_x")
    signal = ReadableSignal("signal")

    def evaluation_function(uid: str, suggestions: list[dict]) -> list[dict]:
        outcomes = []
        for suggestion in suggestions:
            outcomes.append({
                "_id": suggestion["_id"],
                "signal": 1.0 - abs(suggestion["motor_x"]),
            })
        return outcomes

Write a Custom Optimization Callback
====================================

This guide shows how to add a custom callback that records optimization
outcomes while an agent runs.

Use a callback when you want to observe optimization progress or send progress
somewhere else, such as a file, dashboard, log service, or operator display. If
you want background on how callbacks fit into Blop's optimization workflow, see
:doc:`/explanations/callbacks`.

Create a callback class
-----------------------

Subclass ``bluesky.callbacks.CallbackBase`` and implement the document methods
you need. A common pattern is to use ``descriptor`` to identify fields of
interest and ``event`` to record their values.

.. testcode::

    from bluesky.callbacks import CallbackBase


    class OutcomeHistory(CallbackBase):
        def __init__(self):
            self.outcome_keys = []
            self.rows = []

        def descriptor(self, doc):
            self.outcome_keys = [
                name
                for name, data_key in doc.get("data_keys", {}).items()
                if data_key.get("source") == "optimization-outcome"
            ]

        def event(self, doc):
            data = doc.get("data", {})
            row = {key: data[key] for key in self.outcome_keys if key in data}
            if row:
                self.rows.append(row)

Subscribe the callback
----------------------

Configure an agent, then create one callback instance and subscribe it before
running the optimization.

.. testcode::

    from blop.ax import Agent, Objective, RangeDOF


    agent = Agent(
        sensors=[signal],
        dofs=[RangeDOF(actuator=motor_x, bounds=(-1, 1), parameter_type="float")],
        objectives=[Objective(name="signal", minimize=False)],
        evaluation_function=evaluation_function,
    )

    outcome_history = OutcomeHistory()
    agent.subscribe(outcome_history)

    RE(agent.optimize(iterations=2))

.. testoutput::
   :hide:

   ...

After the run, inspect the data collected by the callback.

.. testcode::

    print(len(outcome_history.rows) == 2)

.. testoutput::

    True

Keep or remove the default logger
---------------------------------

Agents include the default console logger unless you remove it. Keep it if you
want normal console output alongside your custom callback.

To run only your callback, clear the existing callbacks before subscribing.

.. testcode::

    agent.callbacks.clear()
    agent.subscribe(outcome_history)

To remove a specific callback later, keep a reference to it and unsubscribe it.

.. testcode::

    agent.unsubscribe(outcome_history)

Use callback data carefully
---------------------------

Use callbacks for side effects: logging, notifications, dashboards, summaries,
or lightweight record keeping. Do not use a callback to compute objective
values, move hardware, or update the optimizer state. Put those tasks in the
evaluation function, acquisition plan, or optimizer instead.
