# Blop - Beamline Optimization Package


[![Testing](https://github.com/bluesky/blop/actions/workflows/ci.yml/badge.svg)](https://github.com/bluesky/blop/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/bluesky/blop/branch/main/graph/badge.svg)](https://codecov.io/gh/bluesky/blop)
[![PyPI](https://img.shields.io/pypi/v/blop.svg)](https://pypi.python.org/pypi/blop)
[![Conda](https://img.shields.io/conda/vn/conda-forge/blop.svg)](https://anaconda.org/conda-forge/blop)

* Free software: 3-clause BSD license
* Documentation: <https://NSLS-II.github.io/blop>.


## Installation

### For Users

```bash
pip install blop
```

or with conda:

```bash
conda install -c conda-forge blop
```

### For Development

Using Pixi (recommended):

```bash
git clone https://github.com/bluesky/blop.git
cd blop
pixi install -e dev-cpu
```

Or using pip:

```bash
git clone https://github.com/bluesky/blop.git
cd blop
pip install -e .
```

### For Running Tutorials

To run the tutorial notebooks and examples, you'll also need to install the simulation package:

Using Pixi:
```bash
pixi install -e docs  # Includes blop-sim with XRT support
```

Or using pip:
```bash
pip install -e sim/        # Basic simulations
```

Note: The `sim/` directory contains example simulations and is **not published to PyPI**. It's only available in the GitHub repository for running tutorials and benchmarks.


## Citation

If you use this package in your work, please cite the following paper:

```bibtex
@Article{Morris2024,
  author   = {Morris, Thomas W. and Rakitin, Max and Du, Yonghua and Fedurin, Mikhail and Giles, Abigail C. and Leshchev, Denis and Li, William H. and Romasky, Brianna and Stavitski, Eli and Walter, Andrew L. and Moeller, Paul and Nash, Boaz and Islegen-Wojdyla, Antoine},
  journal  = {Journal of Synchrotron Radiation},
  title    = {{A general Bayesian algorithm for the autonomous alignment of beamlines}},
  year     = {2024},
  month    = {Nov},
  number   = {6},
  pages    = {1446--1456},
  volume   = {31},
  doi      = {10.1107/S1600577524008993},
  keywords = {Bayesian optimization, automated alignment, synchrotron radiation, digital twins, machine learning},
  url      = {https://doi.org/10.1107/S1600577524008993},
}
```
