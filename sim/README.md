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
  - `DetectorDevice`: Generates beam images (statistics computed in evaluation functions)

## Usage

### XRT Backend Example

```python
from blop_sim.backends.xrt import XRTBackend
from blop_sim.devices import KBMirrorXRT, DetectorDevice, SimplePathProvider

# Create backend (singleton)
backend = XRTBackend()

# Create devices
kbv = KBMirrorXRT(backend, mirror_index=0, initial_radius=38000, name="kbv")
kbh = KBMirrorXRT(backend, mirror_index=1, initial_radius=21000, name="kbh")
det = DetectorDevice(backend, SimplePathProvider("/tmp/blop/sim"), name="det")
```

### Simple Backend Example

```python
from blop_sim.backends.simple import SimpleBackend
from blop_sim.devices import KBMirrorSimple, SlitDevice, DetectorDevice, SimplePathProvider

# Create backend
backend = SimpleBackend()

# Create devices
kbh = KBMirrorSimple(backend, orientation="horizontal", name="kbh")
kbv = KBMirrorSimple(backend, orientation="vertical", name="kbv")
slit = SlitDevice(backend, name="ssa")
det = DetectorDevice(backend, SimplePathProvider("/tmp/blop/sim"), name="det")
```

## Design Philosophy

- **Realistic**: Each beamline component is a separate device, mirroring real hardware
- **Simple**: Detector only stores images; evaluation functions compute statistics
- **Flexible**: Users compose devices as needed rather than using monolithic beamline classes
- **Modern**: Uses ophyd-async throughout with proper async patterns
