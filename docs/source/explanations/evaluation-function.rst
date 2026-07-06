The Evaluation Function
=======================

The evaluation function is the primary interface between **blop** and your experimental data analysis pipeline. It is responsible for retrieving
experimental data, performing any required post-processing, and computing the objective values returned to the optimizer.

Rather than prescribing a particular processing framework or directly managing data, **blop** adopts a UID-driven workflow that mirrors the event-based processing patterns
commonly used at beamlines. This design allows existing data handling and analysis code to be reused with minimal modification, reducing the need to
reimplement processing logic specifically for optimization.


Anatomy of an Evaluation Function
---------------------------------

An evaluation function is simply a callable object that accepts a run UID, and a suggestion list and returns the objective values computed from the corresponding experimental data.

A typical implementation is shown below:

.. code-block:: python

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

        def __call__(self, uid, suggestions):
            # Invoked by the RunEngine with the UID of an acquired run.
            #
            # Typical responsibilities include:
            #   - Retrieving data associated with the UID
            #   - Iterating over individual suggestions in the run 
            #       - found in a run's start document under "blop_suggestions" when using default_aquire aquisition plan
            #   - Constructing a per-suggestion analysis context
            #   - Calling a lower-level objective function for each
            #     sample or suggestion

Although the interface is intentionally minimal, separating setup from
execution is recommended.

``__init__``
    Perform one-time initialization such as constructing storage clients,
    loading analysis resources, and configuring reusable analysis parameters.

``__call__``
    Retrieve the data associated with a run UID, iterate over the individual
    suggestions or samples, and orchestrate the analysis workflow.

Where possible, keep the actual objective calculation in a separate function
that operates on a single sample or suggestion. This separation makes the
analysis logic easier to test, reuse, and maintain independently of the data
retrieval code.
