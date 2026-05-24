Queueserver
===========

.. warning::

   The queueserver integration is **experimental**. The API is not yet stable
   and may change in future releases without a deprecation period. It is not
   recommended for production use.

These classes implement the distributed optimization backend that connects
Blop to a remote `Bluesky Queueserver <https://blueskyproject.io/bluesky-queueserver/>`_.
See the :doc:`/tutorials/queueserver` tutorial for a full worked example.

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
