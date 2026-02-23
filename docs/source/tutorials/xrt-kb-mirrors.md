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

import matplotlib.pyplot as plt
from tiled.client.container import Container
from bluesky.callbacks import best_effort
from bluesky.callbacks.tiled_writer import TiledWriter
from bluesky.run_engine import RunEngine
from tiled.client import from_uri  # type: ignore[import-untyped]
from tiled.server import SimpleTiledServer

from blop.ax import Agent, RangeDOF, Objective
from blop.protocols import EvaluationFunction


# From https://github.com/bluesky/blop/blob/main/examples
from simulations.xrt_kb_pair.xrt_beamline import TiledBeamline

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

Now we create the beamline and define our DOFs. Each `RangeDOF` wraps an actuator (something we can move) with bounds that constrain the search space:

```{code-cell} ipython3
beamline = TiledBeamline(name="bl")

dofs = [
    RangeDOF(actuator=beamline.kbv_dsv, bounds=VERTICAL_BOUNDS, parameter_type="float"),
    RangeDOF(actuator=beamline.kbh_dsh, bounds=HORIZONTAL_BOUNDS, parameter_type="float"),
]
```

The `actuator` is the device that physically changes the parameter. The `bounds` tell the optimizer what range of values to explore. Think of DOFs as the "knobs" the optimizer can turn.

## Defining Objectives

**Objectives** specify what you want to optimize. Each objective has a name (matching a value your evaluation function will return) and a direction: `minimize=True` for things you want smaller, `minimize=False` for things you want larger.

For our KB mirrors, we have three objectives:
- **Intensity** (`bl_det_sum`): We want *more* signal → `minimize=False`
- **Spot width X** (`bl_det_wid_x`): We want a *tighter* spot → `minimize=True`
- **Spot width Y** (`bl_det_wid_y`): We want a *tighter* spot → `minimize=True`

```{code-cell} ipython3
objectives = [
    Objective(name="bl_det_sum", minimize=False),
    Objective(name="bl_det_wid_x", minimize=True),
    Objective(name="bl_det_wid_y", minimize=True),
]
```

With multiple objectives that can conflict (maximizing intensity might increase spot size), the optimizer finds the *Pareto frontier*—the set of solutions where you can't improve one objective without sacrificing another.

## Writing an Evaluation Function

The **evaluation function** is the bridge between raw experimental data and the optimizer. After each measurement, the optimizer needs to know how well that configuration performed. Your evaluation function:

1. Receives a run UID and the suggestions that were tested
2. Reads the relevant data from Tiled
3. Returns outcome values for each suggestion

```{code-cell} ipython3
class DetectorEvaluation(EvaluationFunction):
    def __init__(self, tiled_client: Container):
        self.tiled_client = tiled_client
    
    def __call__(self, uid: str, suggestions: list[dict]) -> list[dict]:
        outcomes = []
        run = self.tiled_client[uid]
        bl_det_sum = run["primary/bl_det_sum"].read()
        bl_det_wid_x = run["primary/bl_det_wid_x"].read()
        bl_det_wid_y = run["primary/bl_det_wid_y"].read()

        # Suggestions are stored in the start document's metadata when
        # using the `blop.plans.default_acquire` plan.
        # You may want to store them differently in your experiment when writing
        # a custom acquisition plan.
        suggestion_ids = [suggestion["_id"] for suggestion in run.metadata["start"]["blop_suggestions"]]

        for idx, sid in enumerate(suggestion_ids):
            outcome = {
                "_id": sid,
                "bl_det_sum": bl_det_sum[idx],
                "bl_det_wid_x": bl_det_wid_x[idx],
                "bl_det_wid_y": bl_det_wid_y[idx],
            }
            outcomes.append(outcome)
        return outcomes
```

Note the `_id` field—this links each outcome back to the suggestion that produced it. This is essential when multiple configurations are tested in a single run.

## Creating and Running the Agent

The **Agent** brings everything together. It:
- Uses DOFs to know what parameters to adjust
- Uses objectives to know what to optimize
- Calls the evaluation function to assess each configuration
- Builds a surrogate model to predict outcomes across the parameter space
- Suggests the next configurations to try

```{code-cell} ipython3
agent = Agent(
    sensors=[beamline.det],
    dofs=dofs,
    objectives=objectives,
    evaluation_function=DetectorEvaluation(tiled_client),
    name="xrt-blop-demo",
    description="A demo of the Blop agent with XRT simulated beamline",
    experiment_type="demo",
)
```

The `sensors` list contains any devices that produce data during acquisition. Here, `beamline.det` is our detector.

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
_ = agent.plot_objective(x_dof_name="bl_kbh_dsh", y_dof_name="bl_kbv_dsv", objective_name="bl_det_sum")
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
    [beamline.det],
    beamline.kbv_dsv, [optimal_parameters[beamline.kbv_dsv.name]],
    beamline.kbh_dsh, [optimal_parameters[beamline.kbh_dsh.name]],
))
```

```{code-cell} ipython3
image = tiled_client[uid[0]]["primary/bl_det_image"].read().squeeze()
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

These same components apply to any optimization problem: swap the simulated beamline for real hardware, adjust the DOFs and objectives for your system, and write an evaluation function that extracts your metrics.

## Next Steps

- Learn about [custom acquisition plans](../how-to-guides/acquire-baseline.rst) for more complex measurement sequences
- Explore [DOF constraints](../how-to-guides/set-dof-constraints.rst) to encode physical limitations
- See [outcome constraints](../how-to-guides/set-outcome-constraints.rst) to enforce requirements on your results

For the beamline setup code used in this tutorial, see:
- [xrt_beamline.py](https://github.com/NSLS-II/blop/blob/main/src/blop/sim/xrt_beamline.py)
- [xrt_kb_model.py](https://github.com/NSLS-II/blop/blob/main/src/blop/sim/xrt_kb_model.py)
