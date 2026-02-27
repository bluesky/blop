# Blop Simulations

This package provides ophyd-async simulation devices for BLOP documentation and tutorials. It is **not published to PyPI** and is only meant for local development, testing, and running tutorials.

## Installation

To use the examples and tutorials, install this package in editable mode from the repository root:

```bash
pip install -e sim/
```

## Architecture

The package uses a component-based architecture with individual devices:

- **Backends**: Global singletons that handle beam physics
  - `SimpleBackend`: Mathematical Gaussian beam simulation
  - `XRTBackend`: Full ray-tracing simulation using XRT
  
- **Devices**: Individual ophyd-async devices
  - `KBMirrorSimple`: KB mirror with jack positions (for SimpleBackend)
  - `KBMirrorXRT`: KB mirror with curvature radius (for XRTBackend)
  - `SlitDevice`: Four-blade aperture slit
  - `DetectorDevice`: Generates beam images (from the backend API)

