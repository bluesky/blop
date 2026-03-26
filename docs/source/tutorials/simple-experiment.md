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

# Your first Bayesian optimization with Blop

In this tutorial, you will learn the three core concepts of Blop: **DOFs** (the parameters you can adjust), **objectives** (what you want to optimize), and the **Agent** (which coordinates the optimization). We'll optimize a simple mathematical function using simulated devices—the same patterns apply to real hardware.

## Setup

First, let's import what we need and start the data infrastructure:

```{code-cell} ipython3
import logging
import time
from typing import Any

from blop.ax import Agent, RangeDOF, Objective

from bluesky.protocols import NamedMovable, Readable, Status, Hints, HasHints, HasParent
from bluesky.run_engine import RunEngine
from bluesky_tiled_plugins import TiledWriter
from tiled.client import from_uri
from tiled.client.container import Container
from tiled.server import SimpleTiledServer

# Suppress noisy logs from httpx 
logging.getLogger("httpx").setLevel(logging.WARNING)
```

```{code-cell} ipython3
# Start a local Tiled server for data storage
tiled_server = SimpleTiledServer()

# Set up the Bluesky RunEngine and connect it to Tiled
RE = RunEngine({})
tiled_client = from_uri(tiled_server.uri)
tiled_writer = TiledWriter(tiled_client)
RE.subscribe(tiled_writer)
```

## Creating simulated devices

Bluesky controls devices through protocols. For this tutorial, we create simple simulated "movable" devices. In real experiments, you would use [Ophyd](https://blueskyproject.io/ophyd-async) devices or similar—the code below is just boilerplate to simulate hardware:

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

## Defining DOFs and objectives

**DOFs** (degrees of freedom) are the parameters the optimizer can adjust. **Objectives** are what you want to optimize. Here we define two DOFs (`x1` and `x2`) that can range from -5 to 5, and one objective (the Himmelblau function) that we want to minimize:

```{code-cell} ipython3
x1 = MovableSignal("x1", initial_value=0.1)
x2 = MovableSignal("x2", initial_value=0.23)

dofs = [
    RangeDOF(actuator=x1, bounds=(-5, 5), parameter_type="float"),
    RangeDOF(actuator=x2, bounds=(-5, 5), parameter_type="float"),
]
objectives = [
    Objective(name="himmelblau_2d", minimize=True),
]
sensors = []
```

## Writing the evaluation function

The **evaluation function** computes objective values from experimental data. After each run, Blop calls this function with the run's unique ID and the suggestions that were tried. It returns the computed objective values:

```{code-cell} ipython3
class Himmelblau2DEvaluation():
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
            # Himmelblau function: has four global minima where value = 0
            outcomes.append({
                "himmelblau_2d": (x1 ** 2 + x2 - 11) ** 2 + (x1 + x2 ** 2 - 7) ** 2,
                "_id": suggestion_id
            })
        
        return outcomes
```

## Running the optimization

The **Agent** brings everything together. Create one with your DOFs, objectives, and evaluation function, then run the optimization:

```{code-cell} ipython3
agent = Agent(
    sensors=sensors,
    dofs=dofs,
    objectives=objectives,
    evaluation_function=Himmelblau2DEvaluation(tiled_client=tiled_client),
    name="simple-experiment",
    description="A simple experiment optimizing the Himmelblau function",
)

RE(agent.optimize(30))
```

## Viewing the results

After optimization, visualize what the Agent learned and see the best parameters found:

```{code-cell} ipython3
agent.plot_objective("x1", "x2", "himmelblau_2d")
agent.ax_client.summarize()
```

The Himmelblau function has four global minima (all with value 0). The `summarize` output shows which one(s) the optimizer found.

## What you learned

You now understand the three core concepts of Blop:

- **DOFs**: The parameters the optimizer adjusts (here, `x1` and `x2` with bounds)
- **Objectives**: What you're optimizing (here, minimizing the Himmelblau function)
- **Agent**: Coordinates the optimization loop between Bluesky and the evaluation function

## Next steps

For a more comprehensive tutorial with multiple objectives and diagnostic tools, see [Optimizing KB Mirrors](./xrt-kb-mirrors.md).
