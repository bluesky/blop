Deployments
===========

This page highlights examples of where Blop is being used or developed. It is
not intended to be a complete deployment registry.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Facility
     - Beamline
     - Activity
   * - NSLS-II
     - `TES <https://www.bnl.gov/nsls2/beamlines/beamline.php?r=8-BM>`_
     - Autonomous beamline alignment using Bayesian optimization. Published as:
       Morris, T. W., Rakitin, M., Du, Y., Fedurin, M., Giles, A. C., Leshchev,
       D., Li, W. H., Romasky, B., Stavitski, E., Walter, A. L., Moeller, P.,
       Nash, B., & Islegen-Wojdyla, A. (2024). *A general Bayesian algorithm
       for the autonomous alignment of beamlines*. Journal of Synchrotron
       Radiation, 31(6), 1446-1456. https://doi.org/10.1107/S1600577524008993
   * - NSLS-II
     - `LiX <https://www.bnl.gov/nsls2/beamlines/beamline.php?r=16-ID>`_
     - Optics tuning, including CRL position searches and bimorph mirror
       optimization using image or intensity feedback from the beamline.
   * - NSLS-II
     - `XPD <https://www.bnl.gov/nsls2/beamlines/beamline.php?r=28-ID-2>`_
     - Autonomous synthesis workflows where syringe-pump infusion rates are
       optimized against measured material properties such as emission peak
       position, peak width, quantum yield, and PDF-derived metrics.
   * - NSLS-II
     - `BMM <https://www.bnl.gov/nsls2/beamlines/beamline.php?r=6-BM>`_
     - Beam-position and beam-quality recovery across energy changes, using
       selected optics settings as optimization variables and camera/intensity
       diagnostics as feedback.
   * - NSLS-II
     - `CHX <https://www.bnl.gov/nsls2/beamlines/beamline.php?r=11-ID>`_
     - Bragg-peak optimization and detector-centering workflows using sample
       and detector-stage positions as optimization variables with Timepix3 ROI
       intensity rocking curves and beam image centroid metrics as feedback.
