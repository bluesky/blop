Optimization Callbacks
======================

Blop callbacks are for observing optimization progress. They let users attach
side effects such as console output, file logging, dashboards, or notifications
without putting reporting logic inside the optimizer, acquisition plan, or
evaluation function.

Why callbacks exist
-------------------

Optimization can be a long-running activity, and users often need to know which
points were suggested, which acquisition run evaluated them, what outcomes came
back, and whether the run finished cleanly. These concerns are related to the
workflow, but they do not define the workflow. A callback sits beside the
optimization and reacts to the Bluesky documents it emits.

The built-in :class:`blop.callbacks.logger.OptimizationLogger` follows this
model. It formats optimization progress for the console, but it does not choose
points, move hardware, or compute objectives.

How Blop routes documents
-------------------------

Blop builds on the standard Bluesky callback model: callbacks receive
``start``, ``descriptor``, ``event``, and ``stop`` documents. When
:meth:`blop.ax.Agent.optimize` runs, Blop wraps the plan with an optimization
callback router. The router forwards only runs whose ``run_key`` marks them as
optimization-related: ``"optimize"`` or ``"sample_suggestions"``.

The optimization run and the acquisition run are related but distinct. The
acquisition run records what happened at the beamline for a suggested point.
The optimization run records the optimization-facing summary of that step: the
suggestion identifiers, the acquisition UID, the parameter values, and the
outcomes returned by the evaluation function.

What a callback sees
--------------------

Optimization callbacks see the same document types as other Bluesky callbacks,
but the contents are optimization-focused.

``start``
    Run-level context such as optimizer description, actuator names, sensor
    names, number of iterations, and ``n_points``.

``descriptor``
    Field descriptions. Blop marks optimization fields with a ``source`` value,
    distinguishing parameters, outcomes, suggestion identifiers, acquisition
    UIDs, and other values.

``event``
    Per-step values such as ``suggestion_ids``, ``bluesky_uid``, parameter
    values, and outcome values. For batched optimization, these values may be
    arrays.

``stop``
    Completion status and reason, useful for summaries or cleanup.

What belongs in a callback
--------------------------

Callbacks are best suited to observation and side effects:

- Formatting progress for a console or notebook.
- Writing optimization summaries to a file or database.
- Updating a dashboard or live display.
- Sending a notification when an optimization starts, stops, or reaches a
  notable outcome.
- Collecting lightweight statistics from emitted parameter and outcome values.

Callbacks should generally avoid changing the optimization state, moving
hardware, or computing objective values. Those responsibilities belong
elsewhere. If a value changes what the optimizer learns, it belongs in the
evaluation function. If a step changes what happens at the beamline, it belongs
in the acquisition plan. If it changes what point should be sampled next, it
belongs in the optimizer.

Writing custom callbacks
------------------------

Custom callbacks are ordinary Bluesky callbacks. They should rely on the
optimization documents emitted by Blop, rather than on details of how the data
were acquired or how the outcomes were computed. For a runnable example, see
:doc:`/how-to-guides/custom-callbacks`.
