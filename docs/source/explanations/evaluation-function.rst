The Evaluation Function
=======================

The evaluation function is the primary interface between **blop** and your experimental data analysis pipeline. It is responsible for retrieving
experimental data, performing any required post-processing, and computing the objective values returned to the optimizer.

Rather than prescribing a particular processing framework or directly managing data, **blop** uses a uid-driven workflow. The acquisition
plan returns a uid, and Blop passes that same value to the evaluation function. The uid must uniquely identify the acquisition.


Anatomy of an Evaluation Function
---------------------------------

An evaluation function is a callable that accepts a uid and a sequence of suggestion mappings, then returns a sequence of outcome mappings. The uid may be a Bluesky run UID, suggestion IDs in executed order, a tuple of event UIDs, or a backend-specific type understood by the evaluator. The suggestion sequence is optimizer-provided and is not guaranteed to be in acquisition order; match data and outcomes by ``_id``.

A run-owning acquisition plan usually returns a string run UID:

.. code-block:: python

    from collections.abc import Mapping, Sequence

    from blop.protocols import EvaluationFunction

    class RunUidEvaluation(EvaluationFunction[str]):
        """Evaluator for acquisition plans that return Bluesky run UID strings."""

        def __call__(self, uid: str, suggestions: Sequence[Mapping]) -> Sequence[Mapping]:
            run = self.tiled_client[uid]
            return analyze_run(run, suggestions)

For a custom plan that returns a richer UID, use that concrete type in both the plan and evaluator:

.. code-block:: python

    from dataclasses import dataclass

    @dataclass(frozen=True)
    class QueueAcquisitionUID:
        correlation_uid: str
        item_uid: str | None

    class QueueEvaluation(EvaluationFunction[QueueAcquisitionUID]):
        def __call__(self, uid: QueueAcquisitionUID, suggestions: Sequence[Mapping]) -> Sequence[Mapping]:
            return analyze_queue_data(uid, suggestions)

Although the interface is intentionally minimal, separating setup from
execution is recommended.

``__init__``
    Perform one-time initialization such as constructing storage clients,
    loading analysis resources, and configuring reusable analysis parameters.

``__call__``
    Retrieve the data associated with the uid, iterate over the
    individual suggestions or samples, and orchestrate the analysis workflow.

Where possible, keep the actual objective calculation in a separate function
that operates on a single sample or suggestion. This separation makes the
analysis logic easier to test, reuse, and maintain independently of the data
retrieval code.
