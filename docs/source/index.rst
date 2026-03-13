Blop
====

.. toctree::
    :maxdepth: 2
    :hidden:

    installation
    how-to-guides
    explanation
    tutorials
    reference
    release-history


.. raw:: html

    <p class="index-subtitle">a BeamLine Optimization Package</p>
    <div class="index-about-grid">
    <div>

What is Blop?
-------------

Blop is a Python library for performing optimization for beamline
experiments. It is designed to integrate nicely with the Bluesky
ecosystem and primarily targets rapid beamline data acquisition
and control.

Our goal is to provide a simple and practical data-driven
optimization interface for beamline experiments.

.. raw:: html

    </div>
    <div class="about-viz">
      <div class="about-viz-inner">
         <span class="about-viz-placeholder">[Visualization Placeholder]</span>
      </div>
    </div>
    </div>
    <hr class="section-divider">

Installation
------------

.. raw:: html

    <div class="installation-grid">

.. container:: install-card

    .. container:: install-card-title

        Via PyPI

    .. container:: install-label

        Standard (GPU support):

    .. include:: _includes/installation-code-snippets.rst
       :start-after: .. snippet-pip-standard-start
       :end-before: .. snippet-pip-standard-end

    .. container:: install-label

        CPU-only (containers, CI/CD, laptops):

    .. include:: _includes/installation-code-snippets.rst
       :start-after: .. snippet-pip-cpu-start
       :end-before: .. snippet-pip-cpu-end

.. container:: install-card

    .. container:: install-card-title

        Via Conda-forge

    .. container:: install-label

        Standard:

    .. include:: _includes/installation-code-snippets.rst
       :start-after: .. snippet-conda-standard-start
       :end-before: .. snippet-conda-standard-end

    .. container:: install-label

        CPU-only:

    .. include:: _includes/installation-code-snippets.rst
       :start-after: .. snippet-conda-cpu-start
       :end-before: .. snippet-conda-cpu-end


.. raw:: html

    </div>

For additional installation instructions, refer to the :doc:`installation` guide.

.. raw:: html

    <hr class="section-divider">

Learn More!
-----------

.. raw:: html

    <div class="learn-more-grid">

.. container:: learn-more-item

    .. container:: learn-more-card-link

        :doc:`Tutorials <tutorials>`

    .. container:: learn-more-card-desc

        Step-by-step guides to get started with Blop fundamentals and basic workflows.

.. container:: learn-more-item

    .. container:: learn-more-card-link

        :doc:`How-To Guides <how-to-guides>`

    .. container:: learn-more-card-desc

        Practical recipes and solutions for specific beamline optimization tasks.

.. container:: learn-more-item

    .. container:: learn-more-card-link

        :doc:`References <reference>`

    .. container:: learn-more-card-desc

        Complete API documentation, class references, and technical specifications.

.. container:: learn-more-item

    .. container:: learn-more-card-link

        :doc:`Release History <release-history>`

    .. container:: learn-more-card-desc

        Version updates, new features, bug fixes, and changelog for the current release.

.. raw:: html

    </div>
    <hr class="section-divider">

References
----------

If you use this package in your work, please cite the following paper:

.. raw:: html

    <div class="index-reference-box">

Morris, T. W., Rakitin, M., Du, Y., Fedurin, M., Giles, A. C., Leshchev,
D., Li, W. H., Romasky, B., Stavitski, E., Walter, A. L., Moeller, P.,
Nash, B., & Islegen-Wojdyla, A. (2024). A general Bayesian algorithm for
the autonomous alignment of beamlines. *Journal of Synchrotron Radiation*,
31(6), 1446–1456. `https://doi.org/10.1107/S1600577524008993 <https://doi.org/10.1107/S1600577524008993>`_

.. raw:: html

    </div>

**BibTeX:**

.. code-block:: bibtex

    @Article{Morris2024,
         author   = {Morris, Thomas W. and Rakitin, Max and Du, Yonghua and Fedurin, Mikhail and Giles, Abigail C. and Leshchev, Denis and Li, William H. and Romasky, Brianna and Stavitski, Eli and Walter, Andrew L. and Moeller, Paul and Nash, Boaz and Islegen-Wojdyla, Antoine},
         journal  = {Journal of Synchrotron Radiation},
         title    = {A general Bayesian algorithm for the autonomous alignment of beamlines},
         year     = {2024},
         month    = {Nov},
         number   = {6},
         pages    = {1446--1456},
         volume   = {31},
         doi      = {10.1107/S1600577524008993},
         keywords = {Bayesian optimization, automated alignment, synchrotron radiation, digital twins, machine learning},
         url      = {https://doi.org/10.1107/S1600577524008993},
    }
