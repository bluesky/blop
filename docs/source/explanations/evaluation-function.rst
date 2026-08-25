The Evaluation Function
=======================

The evaluation function is the primary interface between **blop** and your experimental data analysis pipeline. It is responsible for retrieving
experimental data, performing any required post-processing, and computing the objective values returned to the optimizer.

Rather than prescribing a particular processing framework or directly managing data, **blop** uses an identifier-driven workflow. The acquisition
plan returns a hashable identifier, and Blop passes that same identifier to the evaluation function. This mirrors event-based processing patterns
commonly used at beamlines while also supporting data acquired within a single Bluesky run.


Anatomy of an Evaluation Function
---------------------------------

An evaluation function is a callable that accepts a hashable acquisition identifier and a sequence of suggestion mappings, then returns a sequence of outcome mappings. The identifier may be a Bluesky run UID, suggestion IDs in executed order, a tuple of event UIDs, or another hashable key understood by the evaluator. The suggestion sequence is optimizer-provided and is not guaranteed to be in acquisition order; match data and outcomes by ``_id``.

A typical implementation is shown below:

.. code-block:: python

    from collections.abc import Hashable, Mapping, Sequence

    class GenericEvaluation(EvaluationFunction):
        """Inheriting from EvaluationFunction is optional but provides
        a useful typing protocol."""

        def __init__(self, **meta_parameters):
            # Perform one-time setup before passing the evaluator to
            # the optimizer.
            #
            # Typical responsibilities include:
            #   - stashing storage clients within self (e.g. Tiled)
            #   - Initializing analysis resources (dask distributed is considered but yet unexplored in our support)
            #   - Configuring optimization-specific parameters 
            #       - (perhaps varying of exponents in loss combinations, selecting between L1 and L2 norm...)

        def __call__(self, uid: Hashable, suggestions: Sequence[Mapping]) -> Sequence[Mapping]:
            # Invoked with the identifier returned by the acquisition plan.
            #
            # Typical responsibilities include:
            #   - Retrieving the data associated with the identifier
            #   - Iterating over individual suggestions in the acquisition
            #       - using the acquisition plan's documented order/correlation key
            #   - Constructing a per-suggestion analysis context
            #   - Calling a lower-level objective function for each sample or suggestion

Although the interface is intentionally minimal, separating setup from
execution is recommended.

``__init__``
    Perform one-time initialization such as constructing storage clients,
    loading analysis resources, and configuring reusable analysis parameters.

``__call__``
    Retrieve the data associated with the acquisition identifier, iterate over the
    individual suggestions or samples, and orchestrate the analysis workflow.

Where possible, keep the actual objective calculation in a separate function
that operates on a single sample or suggestion. This separation makes the
analysis logic easier to test, reuse, and maintain independently of the data
retrieval code.
