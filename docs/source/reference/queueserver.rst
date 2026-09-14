Queueserver
===========

.. warning::

   The queueserver integration is **experimental**. The API is not yet stable
   and may change in future releases without a deprecation period. It is not
   recommended for production use.

These classes implement the distributed optimization backend that connects
Blop to a remote `Bluesky Queueserver <https://blueskyproject.io/bluesky-queueserver/>`_.
See the :doc:`/tutorials/queueserver` tutorial for a full worked example.

Evaluators receive a submission token immediately after a plan is queued and own
waiting for the data they need. No document dispatcher is required. Existing
run-UID evaluators can opt into ``DocumentStreamEvaluator`` with an
application-managed transport and explicit subscription cleanup.

QueueserverAcquisition
---------------------

.. autoclass:: blop.queueserver.QueueserverAcquisition
   :members:
   :undoc-members:

DocumentStreamEvaluator
-----------------------

.. autoclass:: blop.queueserver.DocumentStreamEvaluator
   :members:
   :undoc-members:

OptimizationResult
------------------

.. autoclass:: blop.queueserver.OptimizationResult
   :members:
   :undoc-members:

QueueserverClient
-----------------

.. autoclass:: blop.queueserver.QueueserverClient
   :members:
   :undoc-members:
   :show-inheritance:

QueueserverOptimizationRunner
------------------------------

.. autoclass:: blop.queueserver.QueueserverOptimizationRunner
   :members:
   :undoc-members:
   :show-inheritance:
