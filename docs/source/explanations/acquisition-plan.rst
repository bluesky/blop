Acquisition Plans
=================

An acquisition plan is a Bluesky plan responsible for executing an optimization
experiment. Given a collection of suggested points, actuators, and sensors, it
performs the necessary system configuration and data acquisition required to
evaluate each suggestion.

For many experiments, the optimizer's degrees of freedom map directly onto
physical actuators. In these cases, the provided ``default_acquire`` plan is
sufficient and requires no customization.

More complex experiments may do things such as optimize over a *virtual* parameter space rather
than directly controlling hardware. Examples include:

- Linear or nonlinear subspace projections
- Krylov subspace parameterizations
- Sequence-dependent configurations
- High-level beamline macros such as ``set_E`` or ``set_attenuation``

In these situations, users can implement a custom acquisition plan that
translates optimization variables into the corresponding physical operations.
The trade-off is that the custom plan becomes responsible for maintaining the
association between optimizer suggestions and the acquired data within the
chosen storage or event system (typically stashing the suggestions in proper 
order in storage or tagging the run markdown).

A simple example that optimizes in a subspace while executing measurements in
the full physical coordinate system is shown below. 

.. code-block:: python

    class SubspaceAcquisition(AcquisitionPlan):

        def __call__(
            self,
            suggestions: list[dict],
            actuators: Sequence[Actuator],
            sensors: Sequence[Sensor] | None = None,
            md: dict[str, Any] | None = None,
        ) -> MsgGenerator[str]:

            coords = [subspace_to_real(s) for s in suggestions]
            yield from default_acquire(
                coords,
                actuators,
                sensors=sensors,
                md=md,
            )

Wrapping the default acquire function is completely acceptable for coordinate to 
coordinate schemes to keep default acquire's convenience. Just remember that 
systems like Tiled will report your converted coordinates in their suggestion 
history during evaluation.

Virtual Coordinate Systems
--------------------------

The mapping between the optimizer's virtual coordinate system and the underlying
hardware does not need to be invertible. Instead, the only requirement is that
every point in the virtual parameter space maps deterministically to a single
physical configuration.

In other words, the mapping may be many-to-one from the physical space back to
the virtual space, but it must always produce a unique physical configuration
for every optimizer suggestion. This guarantees that each suggested point can
be executed unambiguously, even when multiple physical configurations would be
equivalent from the optimizer's perspective.
