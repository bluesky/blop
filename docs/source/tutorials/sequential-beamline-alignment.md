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

# Sequential Optimization

In this tutorial, you will learn how to optimize a complex system with multiple interdependent sections by first optimizing each section individually, then combining these results to find a globally optimal solution.

We'll demonstrate this using two simple mathematical functions, Himmableau and Booth, that share some control parameters.

+++

## Setup
Import the required libraries and start the data infrastructure

```{code-cell} ipython3
import logging
import time
from typing import Any

import numpy as np

from bluesky.run_engine import RunEngine
import bluesky.plans as bp
import bluesky.plan_stubs as bps
from bluesky.protocols import HasHints, HasParent, Hints, NamedMovable, Readable, Status
from bluesky.callbacks.best_effort import BestEffortCallback
from bluesky.callbacks.tiled_writer import TiledWriter

from tiled.client import from_uri
from tiled.server import SimpleTiledServer
from blop.ax import Agent, Objective, RangeDOF
from blop.protocols import EvaluationFunction

# Suppress noisy logs from httpx
logging.getLogger("httpx").setLevel(logging.WARNING)
```

```{code-cell} ipython3
# start a Tiled server and client for data storage
tiled_server = SimpleTiledServer()

# Set up the Bluesky RunEngine and connect it to Tiled
tiled_client = from_uri(tiled_server.uri)
tiled_writer = TiledWriter(tiled_client)
bec = BestEffortCallback()
bec.disable_plots()
RE = RunEngine({})
RE.subscribe(bec)
RE.subscribe(tiled_writer)
```

## Creating simulated devices

Bluesky controls devices through protocols. For this tutorial, we create simple simulated "movable" devices. 

```{code-cell} ipython3
class AlwaysSuccessfulStatus(Status):
    def add_callback(self, callback) -> None:
        callback(self)
    def exception(self, timeout = 0.0):
        return None
    @property
    def done(self) -> bool:
        return True
    @property
    def success(self) -> bool:
        return True

class ReadableSignal(Readable, HasHints, HasParent):
    def __init__(self, name: str) -> None:
        self._name = name
        self._value = 0.0
    @property
    def name(self) -> str:
        return self._name
    @property
    def hints(self) -> Hints:
        return {"fields": [self._name], "dimensions": [], "gridding": "rectilinear"}
    @property
    def parent(self) -> Any | None:
        return None
    def read(self):
        return {self._name: {"value": self._value, "timestamp": time.time()}}
    def describe(self):
        return {self._name: {"source": self._name, "dtype": "number", "shape": []}}

class MovableSignal(ReadableSignal, NamedMovable):
    def __init__(self, name: str, initial_value: float = 0.0) -> None:
        super().__init__(name)
        self._value: float = initial_value
    def set(self, value: float) -> Status:
        self._value = value
        return AlwaysSuccessfulStatus()
```

## Define Evaluation Functions

We need a evaluation functions for each stage. For this tutorial the upstream stage will be evaluated using the Himmelblau function and the downstream stage will use the Booth function.

### Upstream Stage Evaluation

```{code-cell} ipython3
from tiled.client.container import Container
class UpstreamEvaluation(EvaluationFunction):
    def __init__(self, tiled_client: Container):
        self.tiled_client = tiled_client

    def __call__(self, uid: str, suggestions: list[dict]) -> list[dict]:
        run = self.tiled_client[uid]
        outcomes = []
        x1_data = run["primary/x1"].read()
        x2_data = run["primary/x2"].read()

        for suggestion in suggestions:
            suggestion_id = suggestion["_id"]
            x1 = x1_data[suggestion_id % len(x1_data)]
            x2 = x2_data[suggestion_id % len(x2_data)]
            # Himmelblau function
            outcomes.append({
                "upstream_quality": (x1 ** 2 + x2 - 11) ** 2 + (x1 + x2 ** 2 - 7) ** 2,
                "_id": suggestion_id
            })
        
        return outcomes
```

### Downstream Stage Evaluation

```{code-cell} ipython3
class DownstreamEvaluation(EvaluationFunction):
    def __init__(self, tiled_client: Container):
        self.tiled_client = tiled_client

    def __call__(self, uid: str, suggestions: list[dict]) -> list[dict]:
        run = self.tiled_client[uid]
        outcomes = []
        x1_data = run["primary/x1"].read()
        x2_data = run["primary/x2"].read()

        for suggestion in suggestions:
            suggestion_id = suggestion["_id"]
            x1 = x1_data[suggestion_id % len(x1_data)]
            x2 = x2_data[suggestion_id % len(x2_data)]
            # Booth function
            outcomes.append({
                "downstream_quality": (x1 + 2 * x2 - 7) ** 2 + (2* x1 + x2 - 5) ** 2, 
                "_id": suggestion_id
            })
        
        return outcomes
```

## Create the Motors/Actuators

For this tutorial we are assuming that there are only two motors of which they are in both sections. This may not always be the case 

```{code-cell} ipython3
x1 = MovableSignal("x1", initial_value=0.1)
x2 = MovableSignal("x2", initial_value=0.2)
```

## Create Agents for Each Stage

Now we'll create agents for each stage.

```{code-cell} ipython3
# Upstream stage agent
upstream_agent = Agent(
    sensors=[],
    dofs = [
        RangeDOF(actuator=x1, bounds=(-5, 5), parameter_type="float"),
        RangeDOF(actuator=x2, bounds=(-5, 5), parameter_type="float"),
    ],
    objectives=[
        Objective(name="upstream_quality", minimize=True),
    ],
    evaluation=UpstreamEvaluation(tiled_client),
    name="upstream_stage",
    description="Optimize upstream parameters",
)

# Downstream stage agent  
downstream_agent = Agent(
    sensors=[],
    dofs = [
        RangeDOF(actuator=x1, bounds=(-5, 5), parameter_type="float"),
        RangeDOF(actuator=x2, bounds=(-5, 5), parameter_type="float"),
    ],
    objectives=[
        Objective(name="downstream_quality", minimize=True),
    ],
    evaluation=DownstreamEvaluation(tiled_client),
    name="downstream_stage",
    description="Optimize downstream parameters",
)
```

This function sequentially optimizes each section by iteratively running trials until either the stopping strategy determines sufficient improvement has been achieved or the maximum number of iterations is reached.

```{code-cell} ipython3
def sequential_beamline_alignment(stopping_strategy, sections, num_trials_per_section_iteration = 40, max_iterations=10):
    for section_agent in sections:
        section_agent.stopping_strategy = stopping_strategy
        for i in range(max_iterations):
            RE(section_agent.optimize(num_trials_per_section_iteration))
            should_stop, message = stopping_strategy.should_stop_optimization(section_agent.ax_client._experiment)
            if should_stop:
                print(f"Stopping optimization for {section_agent}: {message}")
                break
            else:
                print(f"Unable to stop optimization for {section_agent}")
                if i == max_iterations - 1:
                    print(f"Maximum iterations reached for {section_agent}, moving to next section.")
```

If you want manual approval before continuing to the next section, replace lines 6-13 with:

```python
# Show current results
section_agent.plot_objective()

# Ask user if they want to continue
if input("Continue to next section? (y/n): ").lower() == "y":
    break
```

This allows you to visually inspect the optimization results before proceeding. This should be changed based on what you determine is an appropriate determination of what a "solved" optimization should look like

+++

## Stopping Strategy (based off threshold)

If instead, you want to move automatically between sections, you can use the `ImprovementGlobalStoppingStrategy`, which will look at the previous `window_size` trials to see if there has been improvement less than `improvement_bar`. If so, than the optimization continues to the next section.

```{code-cell} ipython3
from ax.global_stopping.strategies.improvement import ImprovementGlobalStoppingStrategy

stopping_strategy = ImprovementGlobalStoppingStrategy(
    min_trials=30,           # Number of completed trials before checking for stopping
    window_size=20,          # Number of recent trials to consider in the window
    improvement_bar=0.01,   # Minimum relative improvement (as fraction of IQR) required
)
```

Now we can start the optimization for all sections

```{code-cell} ipython3
sequential_beamline_alignment(
    stopping_strategy=stopping_strategy,
    sections=[downstream_agent, upstream_agent],
    num_trials_per_section_iteration= 10, # the number of trials per iteration
    max_iterations=5, # the maximum number of iterations per section
)
```

## Global Optimization
There are three main cases for which global optimization may occur. We are assuming here that global optimzation means that all sections must be as close to optimal as possible at the same time

### Case 1: All motors used in all sections
If this is the case than when we globally optimze, each motor is already at its optimal value so the global optimal is just the optimal value of each combined.

```{code-cell} ipython3
from tiled.client.container import Container
class Case1Evaluation(EvaluationFunction):
    def __init__(self, tiled_client: Container):
        self.tiled_client = tiled_client

    def __call__(self, uid: str, suggestions: list[dict]) -> list[dict]:
        run = self.tiled_client[uid]
        outcomes = []
        x1_data = run["primary/x1"].read()
        x2_data = run["primary/x2"].read()

        for suggestion in suggestions:
            # Special key to identify a suggestion
            suggestion_id = suggestion["_id"]
            x1 = x1_data[suggestion_id % len(x1_data)]
            x2 = x2_data[suggestion_id % len(x2_data)]
            outcomes.append({
                "upstream_quality": self.upstream_value(x1, x2),
                "downstream_quality": self.downstream_value(x1, x2),
                "_id": suggestion_id
            })
        return outcomes
    
    def upstream_value(self,x1,x2):
        return (x1 ** 2 + x2 - 11) ** 2 + (x1 + x2 ** 2 - 7) ** 2
    def downstream_value(self,x1,x2): 
        return (x1 + 2 * x2 - 7) ** 2 + (2* x1 + x2 - 5) ** 2
```

```{code-cell} ipython3
# Upstream stage agent
case1_agent = Agent(
    sensors=[],
    dofs = [
        RangeDOF(actuator=x1, bounds=(-5, 5), parameter_type="float"),
        RangeDOF(actuator=x2, bounds=(-5, 5), parameter_type="float"),
    ],
    objectives=[
        Objective(name="upstream_quality", minimize=True),
        Objective(name="downstream_quality", minimize=True),
    ],
    evaluation=Case1Evaluation(tiled_client),
    name="upstream_stage",
    description="Optimize upstream parameters",
)
```

# Prepare data from sequential optimization results

We take the best parameters from each section's optimization and evaluate
ALL objectives at those parameter values. This gives the global optimizer
two good starting points:
1. Parameters that were optimal for upstream (with downstream quality computed)
2. Parameters that were optimal for downstream (with upstream quality computed)

In order to properly injest data, there must be a value (or `np.nan`) for each DOF and objective value 

```{code-cell} ipython3
case1_eval = Case1Evaluation(tiled_client)

# Get best parameters from each section agent
upstream_params, _, _, _ = upstream_agent.ax_client.get_best_parameterization()
downstream_params, _, _, _ = downstream_agent.ax_client.get_best_parameterization()

# Evaluate BOTH objectives at the upstream optimal point
upstream_best = upstream_params.copy()
upstream_best['upstream_quality'] = case1_eval.upstream_value(upstream_params['x1'], upstream_params['x2'])
upstream_best['downstream_quality'] = case1_eval.downstream_value(upstream_params['x1'], upstream_params['x2'])

# Evaluate BOTH objectives at the downstream optimal point
downstream_best = downstream_params.copy()
downstream_best['upstream_quality'] = case1_eval.upstream_value(downstream_params['x1'], downstream_params['x2'])
downstream_best['downstream_quality'] = case1_eval.downstream_value(downstream_params['x1'], downstream_params['x2'])

# Create list of seed data points
dicts = [
    upstream_best,    # Best for upstream, may not be great for downstream
    downstream_best,  # Best for downstream, may not be great for upstream
]
```

Now since the data is in the appropriate format, we can give the agent this data as a starting off point

```{code-cell} ipython3
case1_agent.ingest(dicts)
```

Since we have not yet ran any trials for this new agent, we must use `configure_generation_strategy`

```{code-cell} ipython3
case1_agent.ax_client.configure_generation_strategy()
```

Now we can optimze

```{code-cell} ipython3
RE(case1_agent.optimize(40))
```

This is one way to check to see how well the global optimization has performed.

```{code-cell} ipython3
print(case1_agent.ax_client.compute_analyses())
```

### Case 2: All motors are only used in one section
This is to say any one motor is only used in one section. This is the simplist out of all of the cases as since the motors are inly used in one section, there is no interaction with the other sections

```{code-cell} ipython3
def independent_motors():
    """Use when motors don't interact across sections"""
    upstream_best = upstream_agent.ax_client.get_best_parameterization()[0]
    downstream_best = downstream_agent.ax_client.get_best_parameterization()[0]
    
    global_best = {
        **upstream_best,
        **downstream_best
    }
    return global_best
```

### Case 3: Mixed - Some motors shared, others section-specific

This case combines aspects of Cases 1 and 2. When preparing seed data, you need to handle motors and objectives that don't exist in all sections.

**Key principle:** Use `np.nan` for unknown values.

**Example scenario:**
- `objective_1` uses motors `x1` and `x2`
- `objective_2` uses motors `x1`, `x2`, and `x3`
- You have results from optimizing `objective_1` (which never moved `x3`)

**How to prepare the data:**

```python
import pandas as pd
df = pd.DataFrame({
    "x1": [1.0],              # Known from objective_1 optimization
    "x2": [2.0],              # Known from objective_1 optimization
    "x3": [np.nan],           # Was not used in objective_1, so unknown
    "objective_1": [2.0],     # Measured value
    "objective_2": [np.nan],  # Not measured, will be evaluated later
})
data = df.to_dict(orient="records")
```

Use `np.nan` for:
- **Motors** that weren't used in a section (e.g., `x3` in upstream optimization)
- **Objectives** that weren't measured in a section (e.g., `downstream_quality` in upstream optimization)

The global agent will handle these appropriately during ingestion and optimization.

+++

You can than ingest the data than continue on as in Case 1.
