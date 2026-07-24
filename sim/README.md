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
  - `TESBackend`: Replay of a data-trained emulator of the NSLS-II TES beamline
    (28,561 measured KB-jack/detector frames at 3 keV). Reuses the
    `blop_sim.devices.simple.KBMirror` jack devices. The trained weights (16 MB,
    derived from measured data) are not in this repository: point the
    `BLOP_SIM_TES_WEIGHTS` environment variable at the `emulator_weights.npz`
    shipped with the [tes-emulator](https://github.com/FLlorente/tes-emulator)
    package, or pass `TESBackend(weights_path=...)`.

- **Devices**: Individual ophyd-async devices

  - Backend-agnostic:
    - `DetectorDevice`: Generates beam images (from the backend API)
    - `SlitDevice`: Four-blade aperture slit
  - Backend-specific (available in submodules):
    - `blop_sim.devices.simple.KBMirror`: KB mirror with jack positions (for SimpleBackend)
    - `blop_sim.devices.xrt.KBMirror`: KB mirror with curvature radius (for XRTBackend)
