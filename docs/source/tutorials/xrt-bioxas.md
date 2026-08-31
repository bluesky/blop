---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.3
kernelspec:
  display_name: blop:docs (3.13.13)
  language: python
  name: python3
---

# Multi-objective optimization for Complicated Beamline Optimization Problems

In this tutorial, you will learn how Blop can be used in more complicated realistic optimization problems, for example this problem of minimizing height and width of the beam while maximizing intensity with six degrees of freedom.

This example follows the same workflow as the earlier tutorials

## Enviornment Setup

```{code-cell} ipython3
import logging
from pathlib import PurePath 

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tiled.client.container import Container
from bluesky_tiled_plugins import TiledWriter
from bluesky.run_engine import RunEngine
from tiled.client import from_uri  # type: ignore[import-untyped]
from tiled.server import SimpleTiledServer
from ophyd_async.core import StaticPathProvider, UUIDFilenameProvider

from blop.ax import Agent, RangeDOF, Objective, OutcomeConstraint
from blop.protocols import EvaluationFunction

# Import simulation devices (requires: pip install -e sim/)
from blop_sim.backends.xrt_bioxas import XRTBIOXASBackend
from blop_sim.devices.xrt_bioxas import DBHR
from blop_sim.devices.xrt import KBMirror
from blop_sim.devices import DetectorDevice
import torch
logging.getLogger("httpx").setLevel(logging.WARNING)

# Enable interactive plotting
plt.ion()

DETECTOR_STORAGE = "/tmp/blop/sim"
```

## Start up SimpleTiledServer

```{code-cell} ipython3
tiled_server = SimpleTiledServer(readable_storage=[DETECTOR_STORAGE])
tiled_client = from_uri(tiled_server.uri)
tiled_writer = TiledWriter(tiled_client)

RE = RunEngine({})
RE.subscribe(tiled_writer)
```

## Writing the evaluation function

In this tutorial we compute three objectives:

- `intensity` (maximize)
- `width` (minimize)
- `height` (minimize)

```{code-cell} ipython3
class BioXASEvaluation(EvaluationFunction):
    """
    Extracts total flux and illuminated footprint area from SampleScreen_local.

    The xrt backend stores a 2-D inten# After the warm-up it is good practice to run Ax's built-in diagnostics before continuing:sity array (photons/s/mm² per bin) at
    each detector exposure.  We integrate over all bins to get total flux and
    count the bins above a threshold to estimate footprint area.
    """

    def __init__(self, tiled_client: Container):
        self.tiled_client = tiled_client

    def _compute_stats(self, image: np.ndarray) -> tuple[float, float, float]:
        """Compute integrated intensity and beam width/height from a beam image."""
        # Convert to grayscale
        gray = image.squeeze()
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        # Convert data type for numerical stability
        gray = gray.astype(np.float32)

        # Smooth w/ (5, 5) kernel and threshold
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        max_val = np.max(blurred)
        if max_val == 0:
            return 0.0, 0.0, 0.0

        thresh_value = 0.2 * max_val
        _, thresh = cv2.threshold(blurred, thresh_value, 255, cv2.THRESH_TOZERO)

        # Total integrated intensity
        total_intensity = np.sum(thresh)

        # Beam width/height from intensity-weighted second moment (sigma)
        total_weight = np.sum(thresh)
        if total_weight <= 0:
            return total_intensity, 0.0, 0.0

        h, w = thresh.shape
        y_coords = np.arange(h, dtype=np.float32)
        x_coords = np.arange(w, dtype=np.float32)

        x_bar = np.sum(x_coords * np.sum(thresh, axis=0)) / total_weight
        y_bar = np.sum(y_coords * np.sum(thresh, axis=1)) / total_weight

        x_var = np.sum((x_coords - x_bar) ** 2 * np.sum(thresh, axis=0)) / total_weight
        y_var = np.sum((y_coords - y_bar) ** 2 * np.sum(thresh, axis=1)) / total_weight

        width = 2 * np.sqrt(x_var)   # ~2 sigma width
        height = 2 * np.sqrt(y_var)  # ~2 sigma height

        return total_intensity, width, height

    def __call__(self, uid: str, suggestions: list[dict]) -> list[dict]:
        outcomes = []
        run1 = self.tiled_client[uid]

        images = run1["primary/det_image"].read()
        suggestion_ids = [
            s["_id"]
            for s in run1.metadata["start"]["blop_suggestions"]
        ]

        for idx, sid in enumerate(suggestion_ids):
            intensity, width, height = self._compute_stats(images[idx])
            outcomes.append(
                {
                    "_id": sid,
                    "intensity": intensity,
                    "width": width,
                    "height": height,
                }
            )

        return outcomes
```

## Define DOF and Objective Functions

For the DOFs, the mirrors’ meridional radii (R) were included as optimization variables because, in the BioXAS-Main beamline model, they are adjustable. `extraPitch` and `extraRoll` were also added to allow for fine local adjustments around otherwise fixed pitch and roll parameters, and demonstrating having multiple dofs for the same device.

The objective function was designed to produce the most tightly focused beam possible by minimizing both beam height and beam width while simultaneously maximizing beam intensity.

```{code-cell} ipython3
VERTICAL_BOUNDS_M = (2500000, 45000000)
HORIZONTAL_BOUNDS_M = (650000, 4000000)
DCM_BOUNDS_PITCH_1 = (0,0.0001) 
DCM_BOUNDS_PITCH_2 = (0,0.0001)    
DCM_BOUNDS_ROLL_1 = (0,0.001)
DCM_BOUNDS_ROLL_2 = (0,0.001)

backend = XRTBIOXASBackend()

det = DetectorDevice(backend, StaticPathProvider(UUIDFilenameProvider(), PurePath(DETECTOR_STORAGE)), name="det")

mirror1 = KBMirror(backend, mirror_index=0, initial_radius=7120000, name="mirror1")
mirror2 = KBMirror(backend, mirror_index=1, initial_radius=2500000, name="mirror2")

dbhr1 = DBHR(backend, optic_index=0, extraPitch=0, extraRoll=0, name="dbhr1")
dbhr2 = DBHR(backend, optic_index=1, extraPitch=0, extraRoll=0, name="dbhr2")

dofs = [      
    RangeDOF(actuator=dbhr1.extraPitch, bounds=DCM_BOUNDS_PITCH_1, parameter_type="float"),
    RangeDOF(actuator=dbhr1.extraRoll, bounds=DCM_BOUNDS_ROLL_1, parameter_type="float"),
    RangeDOF(actuator=dbhr2.extraPitch, bounds=DCM_BOUNDS_PITCH_2, parameter_type="float"),
    RangeDOF(actuator=dbhr2.extraRoll, bounds=DCM_BOUNDS_ROLL_2, parameter_type="float"),
    RangeDOF(actuator=mirror1.radius, bounds=VERTICAL_BOUNDS_M, parameter_type="float"),
    RangeDOF(actuator=mirror2.radius, bounds=HORIZONTAL_BOUNDS_M, parameter_type="float"),
]

objectives = [
    Objective(name="intensity", minimize=False),
    Objective(name="width", minimize=True),
    Objective(name="height", minimize=True),
]
```

## Defining the Agent

Start with 1 iteration with 15 sample points to get an idea of the sample region

```{code-cell} ipython3
agent = Agent(
    sensors=[det],
    dofs=dofs,
    objectives=objectives,
    evaluation_function=BioXASEvaluation(tiled_client),
    name="bioxas-blop-demo",
    description="test",
    experiment_type="demo",
)

# This allows for faster convergence for multi-objective problems but takes significantly longer than the deafult "fast". Requires botorch[fully_bayesian]
# agent.ax_client.configure_generation_strategy(method="quality")

RE(agent.optimize(1, n_points=15))
```

Do some optimization iterations. This simply shows proof of concept since the full iteration loop requires many more iterations before convergence

```{code-cell} ipython3
RE(agent.optimize(20))
```

## Selecting parameters from the Pareto frontier

Because this is a multi-objective problem, the optimizer returns a set of Pareto-optimal trade-offs.

```{code-cell} ipython3
optimal_parameters = next(iter(agent.ax_client.get_pareto_frontier()))[0]
optimal_parameters
```

We are using the first Pareto candidate and than using it with a direct scan.

```{code-cell} ipython3
from bluesky.plans import list_scan

uid = RE(list_scan(
    [det],
    dbhr1.extraPitch, [optimal_parameters[dbhr1.extraPitch.name]],
    dbhr1.extraRoll, [optimal_parameters[dbhr1.extraRoll.name]],
    dbhr2.extraPitch, [optimal_parameters[dbhr2.extraPitch.name]],
    dbhr2.extraRoll, [optimal_parameters[dbhr2.extraRoll.name]],
    mirror1.radius, [optimal_parameters[mirror1.radius.name]],
    mirror2.radius, [optimal_parameters[mirror2.radius.name]],
))
```

```{code-cell} ipython3
image = tiled_client[uid[0]]["primary/det_image"].read().squeeze()
plt.imshow(image, origin="lower")
plt.colorbar()
plt.show()
```

```{code-cell} ipython3
_ = agent.ax_client.compute_analyses()
```

```{code-cell} ipython3
agent.ax_client.summarize()
```
