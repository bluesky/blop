# Blop Examples

This package contains example simulations and benchmarks for blop. It is **not published to PyPI** and is only meant for local development, testing, and running tutorials.

## Installation

To use the examples and tutorials, install this package in editable mode:

```bash
pip install -e .
```

For XRT-based simulations (KB mirrors tutorial), also install the XRT extra:

```bash
pip install -e ".[xrt]"
```

## Contents

- **simulations/**: Simulated beamlines and detectors for tutorials
  - `beamline.py`: Basic synthetic beamline
  - `handlers.py`: HDF5 handlers and utilities
  - `xrt_kb_pair/`: XRT-based KB mirror simulations
- **benchmarks/**: (Future) Benchmarking code for testing optimization algorithms

## Usage

Once installed, you can import from the examples package in notebooks and tests:

```python
from simulations.beamline import TiledBeamline
from simulations.xrt_kb_pair.xrt_beamline import TiledBeamline as XRTBeamline
```
