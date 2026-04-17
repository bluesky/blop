Installation
============

For users
---------

Installation
^^^^^^^^^^^^

The package works with Python 3.10+ and can be installed from both PyPI and/or conda-forge.

To install the package using the ``pip`` package manager, run the following command:

.. code:: bash

   $ pip install blop

To install the package using the ``conda`` package manager, run the following command:

.. code:: bash

   $ conda install -c conda-forge blop

PyTorch Acceleration Options
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, ``blop`` installs PyTorch with GPU support (~7GB). For environments without GPU support,
or to reduce installation size, you can install a CPU-only version (~900MB) using ``uv``:

.. code:: bash

   $ pip install uv
   $ uv pip install blop[cpu]

This is particularly useful for:

- Containerized deployments without GPU access
- CI/CD pipelines
- Development environments on laptops without NVIDIA GPUs
- Edge computing scenarios

.. note::

   The CPU-only installation requires `uv <https://docs.astral.sh/uv/>`_, a fast Python package installer.
   If you prefer to use standard ``pip``, the default installation will include GPU support.

For conda users who want CPU-only PyTorch:

.. code:: bash

   $ conda install -c conda-forge blop pytorch cpuonly -c pytorch

Running the tutorials
^^^^^^^^^^^^^^^^^^^^^

You have the option of running the tutorials in `Jupyter Lab <https://jupyter.org/>`_ locally or in a browser using `Binder <https://mybinder.org/>`_.

`Binder Blop Tutorials <https://mybinder.org/v2/gh/NSLS-II/blop/HEAD>`_

If you are using Pixi (see :ref:`for-developers` below), you can do the following for a local Jupyter Lab instance: 

.. code:: bash

   $ pixi run start-jupyter

Your third option is to simply convert the tutorials to ipynb format and use whatever you prefer to run them.

.. code:: bash

   $ jupytext --to ipynb docs/source/tutorials/*.md

.. _for-developers:

For developers
--------------

We recommend using Pixi to manage your development environments. Go to https://pixi.sh/latest/installation/ to install it.

If you don't want to use Pixi, you can view the configuration in the ``pixi.toml`` file and create your own based on it.

Static checks
^^^^^^^^^^^^^

For linting, formatting, and static code analysis.

.. code:: bash

   $ pixi run check

Run tests
^^^^^^^^^

For running the tests.

.. code:: bash

   $ pixi run unit-tests
   $ pixi run test-docs


Build documentation
^^^^^^^^^^^^^^^^^^^

For building this documentation.

.. code:: bash

   $ pixi run build-docs
