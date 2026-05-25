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
- How **tracking metrics** let you monitor values without optimizing them
- How to write an **evaluation function** that extracts results from experimental data
- How the **Agent** coordinates the optimization loop
- How to **check optimization health** mid-run and continue

We'll work with a simulated KB mirror beamline, but the concepts apply directly to real experimental setups.

## What are KB Mirrors?

KB mirror systems use two curved mirrors to focus X-ray beams. Each mirror has adjustable curvature—getting both just right produces a tight, intense focal spot. We'll frame this as a single-objective optimization problem: minimize the beam's FWHM (full width at half maximum) on the detector, subject to a minimum intensity constraint.

The image below shows our simulated setup: a beam from a geometric source propagates through a pair of toroidal mirrors that focus it onto a screen.

![xrt_blop_layout_w.jpg](../_static/xrt_blop_layout_w.jpg)

## Setting Up the Environment

Before we can optimize, we need to set up the data infrastructure. Blop uses [Bluesky](https://blueskyproject.io/) to run experiments and [Tiled](https://blueskyproject.io/tiled/) to store and retrieve data.

```{code-cell} ipython3
import logging
import warnings
from pathlib import PurePath

import matplotlib.pyplot as plt
import numpy as np
from ax.api.protocols import IMetric
from bluesky.run_engine import RunEngine
from bluesky_tiled_plugins import TiledWriter
from ophyd_async.core import StaticPathProvider, UUIDFilenameProvider
from tiled.client import from_uri  # type: ignore[import-untyped]
from tiled.client.container import Container
from tiled.server import SimpleTiledServer

from blop.ax import Agent, Objective, RangeDOF
from blop.ax.objective import OutcomeConstraint
from blop.protocols import EvaluationFunction

# Import simulation devices (requires: pip install -e sim/)
from blop_sim.backends.xrt import XRTBackend
from blop_sim.devices import DetectorDevice
from blop_sim.devices.xrt import KBMirror

# Suppress noisy logs from httpx and dependency deprecation warnings
logging.getLogger("httpx").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)

# Enable interactive plotting
plt.ion()

DETECTOR_STORAGE = "/tmp/blop/sim"
```

Next, we create a local Tiled server. The `TiledWriter` callback will save experimental data to this server, and our evaluation function will read from it.

```{code-cell} ipython3
tiled_server = SimpleTiledServer(readable_storage=[DETECTOR_STORAGE])
tiled_client = from_uri(tiled_server.uri)
tiled_writer = TiledWriter(tiled_client)

RE = RunEngine({})
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

## Defining the Objective and Constraints

For beam focusing, we use a single objective: **minimize the beam FWHM** (full width at half maximum). This is more sample-efficient than multi-objective optimization because the optimizer only needs to model one response surface.

We also track **intensity** as a metric without optimizing it directly. An `OutcomeConstraint` ensures the optimizer avoids configurations where the beam misses the detector entirely:

```{code-cell} ipython3
# Single objective: minimize the geometric-mean FWHM
objectives = [
    Objective(name="fwhm", minimize=True),
]

# Track intensity without optimizing it
intensity_metric = IMetric(name="intensity")

# Soft constraint: reject configurations where most rays miss the screen
outcome_constraints = [
    OutcomeConstraint(constraint="i >= 10000", i=intensity_metric),
]
```

Using a single objective with an outcome constraint gives us the best of both worlds: focused optimization on spot size, with a safety net ensuring we don't "optimize" toward configurations where the beam is simply lost.

## Writing an Evaluation Function

The **evaluation function** is the bridge between raw experimental data and the optimizer. After each measurement, the optimizer needs to know how well that configuration performed. Our evaluation function:

1. Receives a run UID and the suggestions that were tested
1. Reads the beam images from Tiled
1. Computes FWHM from the marginal beam profiles
1. Returns outcome values for each suggestion

We compute FWHM using **marginal profiles** — projecting the 2D image onto each axis by summing, then finding where the 1D profile crosses half its peak value. This approach is robust to noise and dead pixels (they get averaged out in the projection) and doesn't require curve fitting.

```{code-cell} ipython3
class DetectorEvaluation(EvaluationFunction):
    def __init__(self, tiled_client: Container):
        self.tiled_client = tiled_client

    def _fwhm_from_profile(self, profile: np.ndarray) -> float:
        """Compute FWHM from a 1D marginal profile.

        Finds the half-maximum crossing points with sub-pixel interpolation.
        Returns a large value if the beam is too dim or fills the entire detector.
        """
        peak = profile.max()
        if peak == 0:
            return float(len(profile))  # No signal — return detector width as penalty

        half_max = peak / 2.0
        above = profile >= half_max
        if not above.any():
            return float(len(profile))

        indices = np.where(above)[0]
        left_idx = indices[0]
        right_idx = indices[-1]

        # Sub-pixel interpolation at left crossing
        if left_idx > 0:
            left = left_idx - 1 + (half_max - profile[left_idx - 1]) / (profile[left_idx] - profile[left_idx - 1])
        else:
            left = 0.0

        # Sub-pixel interpolation at right crossing
        if right_idx < len(profile) - 1:
            right = right_idx + (half_max - profile[right_idx]) / (profile[right_idx + 1] - profile[right_idx])
        else:
            right = float(len(profile) - 1)

        return right - left

    def _compute_stats(self, image: np.ndarray) -> tuple[float, float]:
        """Compute FWHM and integrated intensity from a beam image.

        Returns
        -------
        fwhm : float
            Geometric mean of the horizontal and vertical FWHM (in pixels).
        intensity : float
            Total integrated intensity (sum of all pixel values).
        """
        gray = image.squeeze().astype(np.float64)
        if gray.ndim == 3:
            gray = gray.mean(axis=-1)

        # Integrated intensity (total flux on detector)
        intensity = gray.sum()

        if intensity == 0:
            return 400.0, 0.0  # No beam — return max FWHM penalty

        # Marginal profiles: project onto each axis
        x_profile = gray.sum(axis=0)  # sum along Y rows -> X profile
        y_profile = gray.sum(axis=1)  # sum along X cols -> Y profile

        fwhm_x = self._fwhm_from_profile(x_profile)
        fwhm_y = self._fwhm_from_profile(y_profile)

        # Geometric mean FWHM — targets a small, round spot
        fwhm = np.sqrt(fwhm_x * fwhm_y)

        return float(fwhm), float(intensity)

    def __call__(self, uid: str, suggestions: list[dict]) -> list[dict]:
        outcomes = []
        run = self.tiled_client[uid]

        # Read beam images from detector
        images = run["primary/det_image"].read()

        # Suggestion IDs stored in start document metadata
        suggestion_ids = [suggestion["_id"] for suggestion in run.metadata["start"]["blop_suggestions"]]

        # Compute statistics from each image
        for idx, sid in enumerate(suggestion_ids):
            image = images[idx]
            fwhm, intensity = self._compute_stats(image)

            outcome = {
                "_id": sid,
                "fwhm": fwhm,
                "intensity": intensity,
            }
            outcomes.append(outcome)
        return outcomes
```

Note how we:

1. Project the 2D image onto each axis to get 1D profiles
1. Find the FWHM of each profile using half-maximum crossings
1. Combine them into a single geometric-mean FWHM metric
1. Track integrated intensity for the outcome constraint
1. Link each outcome back to its suggestion via the `_id` field

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
    outcome_constraints=outcome_constraints,
    name="xrt-blop-demo",
    description="A demo of the Blop agent with XRT simulated beamline",
    experiment_type="demo",
)

# Register intensity as a tracking metric (monitored but not optimized)
agent.ax_client.configure_metrics([intensity_metric])
```

The `sensors` list contains any devices that produce data during acquisition. The `outcome_constraints` tell the optimizer to prefer configurations satisfying the intensity constraint. The `configure_metrics` call registers intensity as a tracking metric so it appears in analyses and summaries.

## Running the Optimization

Let's start the optimization. We'll begin with a batch of 10 points to build an initial model of the parameter space—this includes a center-of-space sample plus quasi-random exploration points.

```{code-cell} ipython3
# Run 1 iteration with a batch of 10 points for initial exploration
RE(agent.optimize(1, n_points=10))
```

## Continuing the Optimization

The optimization state is preserved, so we can simply run more iterations:

```{code-cell} ipython3
# Run remaining 10 iterations
RE(agent.optimize(10))
```

## Understanding the Results

After optimization, we can examine what the agent learned. Ax's `compute_analyses()` runs diagnostics including cross-validation of the surrogate model and optimization trace plots:

```{code-cell} ipython3
_ = agent.ax_client.compute_analyses()
```

We can also get a tabular summary of the trials:

```{code-cell} ipython3
agent.ax_client.summarize()
```

### Visualizing the Surrogate Model

The `plot_objective` method shows how the FWHM varies across the DOF space, based on the surrogate model the agent built:

```{code-cell} ipython3
_ = agent.plot_objective(x_dof_name="kbh-radius", y_dof_name="kbv-radius", objective_name="fwhm")
```

This plot reveals the landscape the optimizer explored. The valley (minimum) shows where the optimal mirror curvatures lie.

## Applying the Optimal Configuration

Let's retrieve the best configuration found during optimization and apply it to see the resulting beam:

```{code-cell} ipython3
optimal_parameters, metrics, _, _ = agent.ax_client.get_best_parameterization(use_model_predictions=False)
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
plt.title("Optimized KB Mirror Beam")
plt.show()
```

```{code-cell} ipython3
tiled_server.close()
```

## What You've Learned

In this tutorial, you worked through a complete Bayesian optimization workflow:

1. **DOFs** define the search space — the parameters you can control and their allowed ranges
1. **Objectives** specify your optimization goal (here: minimize FWHM for a tight focal spot)
1. **Tracking metrics** (`IMetric`) let you monitor values like intensity without optimizing them directly
1. **Outcome constraints** enforce safety bounds on tracked metrics (e.g., minimum beam intensity)
1. **Evaluation functions** extract meaningful metrics from experimental data using robust techniques like marginal-profile FWHM
1. **The Agent** coordinates everything, building a surrogate model of your system and intelligently exploring the parameter space
1. **Health checks** let you diagnose optimization progress and catch issues early

These same components apply to any optimization problem: swap the simulated devices for real hardware, adjust the DOFs and objectives for your system, and write an evaluation function that extracts your metrics.

## Next Steps

- Learn about [custom acquisition plans](../how-to-guides/acquire-baseline.rst) for more complex measurement sequences
- Explore [DOF constraints](../how-to-guides/set-dof-constraints.rst) to encode physical limitations
- See [outcome constraints](../how-to-guides/set-outcome-constraints.rst) to enforce requirements on your results

## See Also

- [`blop_sim` package](https://github.com/bluesky/blop/tree/main/sim/blop_sim) for XRT simulated beamline control
