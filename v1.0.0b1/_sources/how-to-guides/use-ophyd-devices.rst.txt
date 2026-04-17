How to optimize with ophyd and ophyd-async devices
==================================================

This guide will walk you through the process of setting up Blop to optimize with `ophyd <https://blueskyproject.io/ophyd/>`_ and `ophyd-async <https://blueskyproject.io/ophyd-async/>`_ devices.

Ophyd devices
-------------

Ophyd's :class:`ophyd.Signal` class implements both the :class:`blop.protocols.Sensor` and :class:`blop.protocols.Actuator` protocols, so they can be used directly with Blop. You can also use the :class:`ophyd.SignalRO` class which only implements the :class:`blop.protocols.Sensor` protocol if you want to capture this data at each step of the experiment.

The ``name`` attribute of the signal will be used as the name of the :class:`blop.ax.DOF` on the backend.

.. testcode::

    from blop.ax import Agent, RangeDOF, Objective
    from ophyd import Signal, SignalRO

    some_control_signal = Signal(name="some_control_signal")
    some_readable_signal = SignalRO(name="some_readable_signal")
    dof = RangeDOF(actuator=some_control_signal, bounds=(0, 1000), parameter_type="float")

    agent = Agent(
        sensors=[some_readable_signal],
        dofs=[dof],
        objectives=[Objective(name="result", minimize=False)],
        evaluation_function=lambda uid, suggestions: [{"result": 0.1, "_id": suggestion["_id"]} for suggestion in suggestions],
    )

Ophyd-async devices
-------------------

Ophyd-async's :class:`ophyd_async.core.SignalW` class implements the :class:`blop.protocols.Actuator` protocol, so they can also be used directly with Blop. The :class:`ophyd_async.core.SignalR` class implements the :class:`blop.protocols.Sensor` protocol. And the :class:`ophyd_async.core.SignalRW` class implements both.

Below we create *soft* signals that return instances of the above classes.

Once again, the ``name`` attribute of the signal will be used as the name of the :class:`blop.ax.DOF` on the backend.

.. testcode::

    from blop.ax import Agent, RangeDOF, Objective
    from ophyd_async.core import soft_signal_rw, soft_signal_r_and_setter

    some_control_signal = soft_signal_rw(float, name="some_control_signal")
    some_readable_signal = soft_signal_r_and_setter(float, name="some_readable_signal")
    dof = RangeDOF(actuator=some_control_signal, bounds=(0, 1000), parameter_type="float")

    agent = Agent(
        sensors=[some_readable_signal],
        dofs=[dof],
        objectives=[Objective(name="result", minimize=False)],
        evaluation_function=lambda uid, suggestions: [{"result": 0.1, "_id": suggestion["_id"]} for suggestion in suggestions],
    )

Using your devices in custom acquisition plans
----------------------------------------------

If you use a custom acquisition plan by implementing the :class:`blop.protocols.AcquisitionPlan` protocol, you can use the ``actuators`` and/or ``sensors`` arguments to access the ophyd or ophyd-async devices you configured as DOFs.

.. testcode::

    import bluesky.plan_stubs as bps
    from bluesky.utils import MsgGenerator
    from bluesky.run_engine import RunEngine
    from ophyd_async.core import soft_signal_rw

    from blop.ax import Agent, RangeDOF, Objective
    from blop.protocols import AcquisitionPlan, Actuator, Sensor

    def custom_acquire(suggestions: list[dict], actuators: list[Actuator], sensors: list[Sensor]) -> MsgGenerator[str]:
        assert actuators[0].name == "signal1"
        assert sensors[0].name == "signal2"
        yield from bps.null()
        return "test-uid-123"

    RE = RunEngine({})

    signal1 = soft_signal_rw(float, name="signal1")
    signal2 = soft_signal_rw(float, name="signal2")

    dof = RangeDOF(actuator=signal1, bounds=(0, 1000), parameter_type="float")

    agent = Agent(
        sensors=[signal2],
        dofs=[dof],
        acquisition_plan=custom_acquire,
        objectives=[Objective(name="result", minimize=False)],
        evaluation_function=lambda uid, suggestions: [{"result": 0.1, "_id": suggestion["_id"]} for suggestion in suggestions],
    )

    RE(agent.optimize())

.. testoutput::
   :hide:

   ...
