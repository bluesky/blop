---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: dev
  language: python
  name: python3
---

# Optimizing KB Mirrors with Bayesian Optimization

In this tutorial, you will learn how to use Blop to optimize a Kirkpatrick-Baez (KB) mirror system. By the end, you will understand:

- How **degrees of freedom (DOFs)** represent the parameters you can adjust in an experiment
- How **objectives** define what you're trying to optimize
- How to write an **evaluation function** that extracts results from experimental data
- How the **Agent** coordinates the optimization loop
- How to **check optimization health** mid-run and continue

We'll work with a simulated KB mirror beamline, but the concepts apply directly to real experimental setups.

## What are KB Mirrors?

KB mirror systems use two curved mirrors to focus X-ray beams. Each mirror has adjustable curvature—getting both just right produces a tight, intense focal spot. This is a multi-objective optimization problem: we want to maximize beam intensity while minimizing the spot size in both X and Y directions.

The image below shows our simulated setup: a beam from a geometric source propagates through a pair of toroidal mirrors that focus it onto a screen.

![xrt_blop_layout_w.jpg](../_static/xrt_blop_layout_w.jpg)

## Setting Up the Environment

Before we can optimize, we need to set up the data infrastructure. Blop uses [Bluesky](https://blueskyproject.io/) to run experiments and [Tiled](https://blueskyproject.io/tiled/) to store and retrieve data.

```{code-cell} ipython3
import logging
from pathlib import PurePath

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tiled.client.container import Container
from bluesky.callbacks import best_effort
from bluesky_tiled_plugins import TiledWriter
from bluesky.run_engine import RunEngine
from tiled.client import from_uri  # type: ignore[import-untyped]
from tiled.server import SimpleTiledServer
from ophyd_async.core import StaticPathProvider, UUIDFilenameProvider

from blop.ax import Agent, RangeDOF, Objective
from blop.protocols import EvaluationFunction

# Import simulation devices (requires: pip install -e sim/)
from blop_sim.backends.xrt import XRTBackend
from blop_sim.devices.xrt import KBMirror
from blop_sim.devices import DetectorDevice

# Suppress noisy logs from httpx 
logging.getLogger("httpx").setLevel(logging.WARNING)

# Enable interactive plotting
plt.ion()

DETECTOR_STORAGE = "/tmp/blop/sim"
```

Next, we create a local Tiled server. The `TiledWriter` callback will save experimental data to this server, and our evaluation function will read from it. The `BestEffortCallback` provides live feedback during scans.

```{code-cell} ipython3
tiled_server = SimpleTiledServer(readable_storage=[DETECTOR_STORAGE])
tiled_client = from_uri(tiled_server.uri)
tiled_writer = TiledWriter(tiled_client)
bec = best_effort.BestEffortCallback()
bec.disable_plots()

RE = RunEngine({})
RE.subscribe(bec)
RE.subscribe(tiled_writer)
```

## Defining Degrees of Freedom

**Degrees of freedom (DOFs)** are the parameters the optimizer can adjust. In our KB system, we control the curvature radius of each mirror. Let's define the search space:

```{code-cell} ipython3
# Define search ranges for each mirror's curvature radius
# The optimal values (~38000 and ~21000) are intentionally placed
# away from the center to make the optimization more realistic
VERTICAL_BOUNDS = (25000, 45000)    # Optimal ~38000 is in upper portion
HORIZONTAL_BOUNDS = (15000, 35000)  # Optimal ~21000 is in lower portion
```

Now we create the simulation backend and individual devices. Each `RangeDOF` wraps an actuator (something we can move) with bounds that constrain the search space:

```{code-cell} ipython3
# Create XRT simulation backend
backend = XRTBackend()

# Create individual KB mirror devices
kbv = KBMirror(backend, mirror_index=0, initial_radius=38000, name="kbv")
kbh = KBMirror(backend, mirror_index=1, initial_radius=21000, name="kbh")

# Create detector device
det = DetectorDevice(backend, StaticPathProvider(UUIDFilenameProvider(), PurePath(DETECTOR_STORAGE)), name="det")

# Define DOFs using mirror radius signals
dofs = [
    RangeDOF(actuator=kbv.radius, bounds=VERTICAL_BOUNDS, parameter_type="float"),
    RangeDOF(actuator=kbh.radius, bounds=HORIZONTAL_BOUNDS, parameter_type="float"),
]
```

The `actuator` is the device that physically changes the parameter. The `bounds` tell the optimizer what range of values to explore. Think of DOFs as the "knobs" the optimizer can turn.

## Defining Objectives

**Objectives** specify what you want to optimize. Each objective has a name (matching a value your evaluation function will return) and a direction: `minimize=True` for things you want smaller, `minimize=False` for things you want larger.

For our KB mirrors, we have three objectives:
- **Intensity** (`intensity`): We want *more* signal → `minimize=False`
- **Spot width** (`width`): We want a *tighter* spot → `minimize=True`
- **Spot height** (`height`): We want a *tighter* spot → `minimize=True`

```{code-cell} ipython3
objectives = [
    Objective(name="intensity", minimize=False),
    Objective(name="width", minimize=True),
    Objective(name="height", minimize=True),
]
```

With multiple objectives that can conflict (maximizing intensity might increase spot size), the optimizer finds the *Pareto frontier*—the set of solutions where you can't improve one objective without sacrificing another.

## Writing an Evaluation Function

The **evaluation function** is the bridge between raw experimental data and the optimizer. After each measurement, the optimizer needs to know how well that configuration performed. Your evaluation function:

1. Receives a run UID and the suggestions that were tested
2. Reads the beam images from Tiled
3. Computes statistics (intensity, width, centroid, etc.) from the images
4. Returns outcome values for each suggestion

```{code-cell} ipython3
class DetectorEvaluation(EvaluationFunction):
    def __init__(self, tiled_client: Container):
        self.tiled_client = tiled_client

    def _compute_stats(self, image: np.array) -> dict[str, float]:
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

        # Beam width/height from intensity-weighted second moment (σ)
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

        width = 2 * np.sqrt(x_var)   # ~2σ width
        height = 2 * np.sqrt(y_var)   # ~2σ height

        return total_intensity, width, height

    def __call__(self, uid: str, suggestions: list[dict]) -> list[dict]:
        outcomes = []
        run = self.tiled_client[uid]
        
        # Read beam images from detector
        images = run["primary/det_image"].read()

        # Suggestions are stored in the start document's metadata when
        # using the `blop.plans.default_acquire` plan.
        # You may want to store them differently in your experiment when writing
        # a custom acquisition plan.
        suggestion_ids = [suggestion["_id"] for suggestion in run.metadata["start"]["blop_suggestions"]]

        # Compute statistics from each image
        for idx, sid in enumerate(suggestion_ids):
            image = images[idx]
            intensity, width, height = self._compute_stats(image)
            
            outcome = {
                "_id": sid,
                "intensity": intensity,
                "width": width,
                "height": height,
            }
            outcomes.append(outcome)
        return outcomes
```

Note how we:
1. Read the image data from the stored detector data
2. Use image processing techniques to compute beam metrics from the raw detector images
3. Link each outcome back to its suggestion via the `_id` field

## Creating and Running the Agent

The **Agent** brings everything together. It:
- Uses DOFs to know what parameters to adjust
- Uses objectives to know what to optimize
- Calls the evaluation function to assess each configuration
- Builds a surrogate model to predict outcomes across the parameter space
- Suggests the next configurations to try

```{code-cell} ipython3
agent = Agent(
    sensors=[det],
    dofs=dofs,
    objectives=objectives,
    evaluation_function=DetectorEvaluation(tiled_client),
    name="xrt-blop-demo",
    description="A demo of the Blop agent with XRT simulated beamline",
    experiment_type="demo",
)
```

The `sensors` list contains any devices that produce data during acquisition. Here, `det` is our detector device.

## Running the Optimization

Let's start the optimization. Rather than running all iterations at once, we'll pause partway through to check the optimization's health—a practical workflow you'll use in real experiments.

```{code-cell} ipython3
# Run first 10 iterations
RE(agent.optimize(10))
```

## Checking Optimization Health

After running some iterations, it's good practice to check how the optimization is progressing. Ax provides built-in health checks and diagnostics through `compute_analyses()`:

```{code-cell} ipython3
_ = agent.ax_client.compute_analyses()
```

This runs all applicable analyses for the current experiment state, including health checks that flag potential issues like model fit problems or exploration gaps. Review these before continuing.

## Continuing the Optimization

The optimization state is preserved, so we can simply run more iterations:

```{code-cell} ipython3
# Run remaining 20 iterations
RE(agent.optimize(20))
```

## Understanding the Results

After optimization, we can examine what the agent learned. Let's run the full suite of analyses again to see how things have improved:

```{code-cell} ipython3
_ = agent.ax_client.compute_analyses()
```

We can also get a tabular summary of the trials:

```{code-cell} ipython3
agent.ax_client.summarize()
```

### Visualizing the Surrogate Model

The `plot_objective` method shows how an objective varies across the DOF space, based on the surrogate model the agent built:

```{code-cell} ipython3
_ = agent.plot_objective(x_dof_name="kbh_radius", y_dof_name="kbv_radius", objective_name="intensity")
```

This plot reveals the landscape the optimizer explored. Peaks (for maximization) or valleys (for minimization) show where good configurations lie.

## Applying the Optimal Configuration

The Pareto frontier contains all optimal trade-off solutions. Let's retrieve one and apply it to see the resulting beam:

```{code-cell} ipython3
optimal_parameters = next(iter(agent.ax_client.get_pareto_frontier()))[0]
optimal_parameters
```

Now move the mirrors to these optimal positions and acquire an image:

```{code-cell} ipython3
from bluesky.plans import list_scan

uid = RE(list_scan(
    [det],
    kbv.radius, [optimal_parameters[kbv.radius.name]],
    kbh.radius, [optimal_parameters[kbh.radius.name]],
))
```

```{code-cell} ipython3
image = tiled_client[uid[0]]["primary/det_image"].read().squeeze()
plt.imshow(image)
plt.colorbar()
plt.show()
```

## What You've Learned

In this tutorial, you worked through a complete Bayesian optimization workflow:

1. **DOFs** define the search space—the parameters you can control and their allowed ranges
2. **Objectives** specify your goals and whether to minimize or maximize each one
3. **Evaluation functions** extract meaningful metrics from experimental data
4. **The Agent** coordinates everything, building a model of your system and intelligently exploring the parameter space
5. **Health checks** let you diagnose optimization progress and catch issues early

These same components apply to any optimization problem: swap the simulated devices for real hardware, adjust the DOFs and objectives for your system, and write an evaluation function that extracts your metrics.

## Next Steps

- Learn about [custom acquisition plans](../how-to-guides/acquire-baseline.rst) for more complex measurement sequences
- Explore [DOF constraints](../how-to-guides/set-dof-constraints.rst) to encode physical limitations
- See [outcome constraints](../how-to-guides/set-outcome-constraints.rst) to enforce requirements on your results

## See Also

- [`blop_sim` package](https://github.com/bluesky/blop/tree/main/sim/blop_sim) for XRT simulated beamline control
