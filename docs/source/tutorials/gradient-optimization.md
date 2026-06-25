---
jupyter:
  jupytext:
    default_lexer: ipython3
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Your first Scipy optimization with Blop

In this tutorial, you will learn the three core concepts of Blop: **DOFs** (the parameters you can adjust), **objectives** (what you want to optimize), and the **Agent** (which coordinates the optimization). We'll optimize a simple mathematical function using simulated devices—the same patterns apply to real hardware.

## Setup

First, let's import what we need and start the data infrastructure:

```python
import logging
import time
import warnings
from typing import Any

from bluesky.protocols import HasHints, HasParent, Hints, NamedMovable, Readable, Status
from bluesky.run_engine import RunEngine
from bluesky_tiled_plugins import TiledWriter
from tiled.client import from_uri
from tiled.client.container import Container
from tiled.server import SimpleTiledServer

from blop.ax import Objective, RangeDOF
from blop.gradient import SCP, Scipy, ScipyCFG

# Suppress noisy logs from httpx 
logging.getLogger("httpx").setLevel(logging.WARNING)
```

```python
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

```python
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

```python
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

```python
class Himmelblau2DEvaluation():
    def __init__(self, tiled_client: Container):
        self.tiled_client = tiled_client

    def __call__(self, uid: str, suggestions: list[dict]) -> list[dict]:
        run = self.tiled_client[uid]
        outcomes = []
        reordered_suggestions = run.start["blop_suggestions"]
        x1_data = run["primary/x1"].read()
        x2_data = run["primary/x2"].read()

        print("[Himmelblau] evaluating suggestions: ", [s["_id"] for s in suggestions], " reordered to: ", [s["_id"] for s in reordered_suggestions])
        for index, suggestion in enumerate(reordered_suggestions):
            # Special key to identify a suggestion
            suggestion_id = suggestion["_id"]
            x1 = x1_data[index]
            x2 = x2_data[index]
            # Himmelblau function: has four global minima where value = 0
            outcomes.append({
                "himmelblau_2d": (x1 ** 2 + x2 - 11) ** 2 + (x1 + x2 ** 2 - 7) ** 2,
                "_id": suggestion_id
            })
        
        return outcomes
```

## Running the optimization

The **Agent** brings everything together. Create one with your DOFs, objectives, and evaluation function, then run the optimization:

```python
agent = Scipy.Agent(
    sensors=sensors,
    dofs=dofs,
    objectives=objectives,
    evaluation_function=Himmelblau2DEvaluation(tiled_client=tiled_client),
    name="simple-experiment",
    description="A simple experiment optimizing the Himmelblau function",
)

RE(agent.optimize(10))
```

## Configuring the optimization

Sometimes a default **Agent** optimization may not do all that you'd like. We expose a configuration object called ScipyCFG and a pure scipy interface so that the classic parameters of scipy minimize can be tweaked (and some multipoint sampling can be used).

```python
config = ScipyCFG(
    dofs=dofs,
    objective=objectives[0],
    optimizer=SCP.Default,
    threads=4,
    eps=.1
)
agent = Scipy(
    sensors=sensors,
    config=config,
    evaluation_function=Himmelblau2DEvaluation(tiled_client=tiled_client),
    name="test_experiment",
)
res_uid = RE(agent.optimize(20, n_points=2))
```

## Viewing the results

Scipy is a local optimizer so it doesn't have internal point tracking, but we can to grab it from our datastore.

```python
import numpy as np
import matplotlib.pyplot as plt

res_client = tiled_client[res_uid[0]]
data = res_client["primary/internal"].read()
vec = data[["suggestion_ids", "x1", "x2", "himmelblau_2d"]]
res = []
for _, row in vec.iterrows():
    vic = [row.suggestion_ids, row.x1, row.x2, row.himmelblau_2d]
    vic = [x.strip("[]").split() for x in vic]
    for id, x, y, obj in zip(*vic,strict=True):
        if id != "''":
            res.append([int(id.strip("'")), float(x), float(y), float(obj)])
res = np.array(res)

fig, ax = plt.subplots(figsize=(12, 8))

xb, yb = np.random.uniform(-5, 5, (2, 1000))
ax.tripcolor(xb, yb, (xb**2 + yb - 11)**2 + (xb + yb**2 - 7)**2, shading="gouraud")

i, x, y, z = res.T
ps = ax.scatter(x, y, c=range(len(x)), cmap='plasma', s=50)
plt.colorbar(ps).set_label('sample index')
plt.title("Visualizing Scipy's traversal of Himmelblau")
```

The Himmelblau function has four global minima (all with value 0). The `summarize` output shows which one(s) the optimizer found.

## What you learned

You now understand the three core concepts of Blop:

- **DOFs**: The parameters the optimizer adjusts (here, `x1` and `x2` with bounds)
- **Objectives**: What you're optimizing (here, minimizing the Himmelblau function)
- **Agent**: Coordinates the optimization loop between Bluesky and the evaluation function

## Next steps

For a more comprehensive tutorial with multiple objectives and diagnostic tools, see [Optimizing KB Mirrors](./xrt-kb-mirrors.md).
