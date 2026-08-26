Installation
============

For users
---------

Installation
^^^^^^^^^^^^

The package works with Python 3.11+ and can be installed from both PyPI and/or conda-forge.

To install the package using the ``pip`` package manager, run the following command:

.. include:: _includes/installation-code-snippets.rst
   :start-after: .. snippet-pip-standard-start
   :end-before: .. snippet-pip-standard-end

To install the package using the ``conda`` package manager, run the following command:

.. include:: _includes/installation-code-snippets.rst
   :start-after: .. snippet-conda-standard-start
   :end-before: .. snippet-conda-standard-end

Optional Extras
^^^^^^^^^^^^^^^

``blop`` is modular — install only what you need by appending one or more extras:

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Extra
     - Installs
     - Notes
   * - ``blop[ax]``
     - ``ax-platform``, ``botorch``, ``gpytorch``, ``torch``
     - GPU torch by default; pair with ``[cpu]`` for CPU-only
   * - ``blop[xopt]``
     - ``xopt``, ``botorch``, ``gpytorch``, ``torch``
     - GPU torch by default; pair with ``[cpu]`` for CPU-only
   * - ``blop[queueserver]``
     - ``bluesky-queueserver-api``
     - Transport layer only; pair with ``[ax]`` for ``QueueserverAgent``
   * - ``blop[cpu]``
     - *(uv index routing)*
     - Routes ``torch`` to the CPU-only PyTorch index; requires ``uv``
   * - ``blop[all]``
     - All backends + ``[queueserver]``
     - No dev tooling; will grow as new backends are added
   * - ``blop[dev]``
     - ``blop[all]`` + dev tooling
     - For contributors

PyTorch Acceleration Options
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, ``blop`` installs PyTorch with GPU support (~7GB). For environments without GPU support,
or to reduce installation size, you can install a CPU-only version (~900MB) using ``uv``:

.. include:: _includes/installation-code-snippets.rst
   :start-after: .. snippet-pip-cpu-start
   :end-before: .. snippet-pip-cpu-end

This is particularly useful for:

- Containerized deployments without GPU access
- CI/CD pipelines
- Development environments on laptops without NVIDIA GPUs
- Edge computing scenarios

.. note::

   The CPU-only installation requires `uv <https://docs.astral.sh/uv/>`_, a fast Python package installer.
   If you prefer to use standard ``pip``, the default installation will include GPU support.

For conda users who want CPU-only PyTorch:

.. include:: _includes/installation-code-snippets.rst
   :start-after: .. snippet-conda-cpu-start
   :end-before: .. snippet-conda-cpu-end
