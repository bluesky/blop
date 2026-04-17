Ax Integration
==============

We use `Ax <https://ax.dev>`_ as the primary backend for optimization and experiment tracking. This enables us to offload the optimization and experiment tracking processes and
focus on delivering a great user experience for optimization using Bluesky.

.. note::

    The use of Ax is optional. You can implement your own optimizer using the :class:`blop.protocols.Optimizer` protocol. However,
    the :class:`blop.ax.Agent` class is designed to be a nicer interface for using Ax only.

Wrapping Ax
-----------

Generally, we want to avoid wrapping the Ax API and allow users to use the full power of Ax themselves. However, there are some cases where
an API wrapper is useful:

- **DOFs**: We wrap Ax parameters to link them with Bluesky :class:`~blop.protocols.Actuator`\s, enabling automatic movement during acquisition (see :class:`blop.ax.dof.DOF`).
- **Constraints**: We provide :class:`blop.ax.dof.DOFConstraint` and :class:`blop.ax.objective.OutcomeConstraint` with a readable syntax that maps to Ax constraints.
- **Agent**: The :class:`blop.ax.Agent` class provides a higher-level interface for common optimization workflows.

Blop handles the following aspects of the Ax API for you:

- Experiment creation and configuration
- Optimization configuration (objectives, constraints)
- Trial suggestion and ingestion

We chose this minimal approach because it encompasses the basic usage of Ax for optimization. For more complex setups (e.g. multi-fidelity optimization, custom model fitting), use the Ax API directly.

Using the Ax API directly
-------------------------

You can access the underlying :class:`ax.Client` instance via :attr:`blop.ax.Agent.ax_client`.

You can learn all about the various Ax features in the `Ax documentation <https://ax.dev/docs/tutorials/quickstart>`_. Some notable features that are not used by Blop by default are:

- Generation strategy configuration

  - Influencing Ax's default generation strategy: `<https://ax.dev/docs/recipes/influence-gs-choice>`_
  - Using custom BoTorch models: `<https://ax.dev/docs/tutorials/modular_botorch/>`_ (see :doc:`/how-to-guides/custom-generation-strategies` for an example in Blop)
  - Using external generators: `<https://ax.dev/docs/tutorials/external_generation_node/>`_

- Configuring early stopping: `<https://ax.dev/docs/tutorials/early_stopping/>`_ (first-class support for early stopping is coming soon, see `<https://github.com/NSLS-II/blop/issues/129>`_)
- Analyzing the optimization results and model fit: `<https://ax.dev/docs/tutorials/analyses/>`_ (see :func:`blop.ax.Agent.plot_objective` for an example in Blop)
- Configuring tracking metrics: `<https://ax.dev/docs/tutorials/tracking_metrics/>`_
- Saving and loading experiments

  - To JSON: `<https://ax.dev/docs/recipes/experiment-to-json>`_
  - To SQLite: `<https://ax.dev/docs/recipes/experiment-to-sqlite>`_

- Summarizing the experiment: `<https://ax.dev/docs/tutorials/quickstart/>`_
