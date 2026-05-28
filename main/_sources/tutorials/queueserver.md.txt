---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Asynchronous Optimization with Bluesky Queueserver

```{warning}
The queueserver integration is **experimental**. The API is not yet stable and
may change in future releases without a deprecation period. It is not
recommended for production use.
```

In this tutorial, you will learn how to run Blop optimization against a remote [Bluesky Queueserver](https://blueskyproject.io/bluesky-queueserver/). This architecture is used when:

- The experiment hardware is controlled by a shared instrument server
- You want the optimizer to run in a separate process from the RunEngine
- You need asynchronous, non-blocking optimization (the agent submits plans and reacts to completions)

We will optimize the same Himmelblau function from the [simple experiment tutorial](./simple-experiment.md), but now the devices live inside a remote queueserver process rather than in the same Python session as the agent.

## Architecture

The distributed system has five components:

```{mermaid}
flowchart
    subgraph docker["Docker Compose Stack"]
        redis["Redis"]
        rem["RE Manager<br/>(devices + plans)"]
        zmqp["ZMQ Proxy<br/>(pub/sub)"]
        bridge["ZMQ-Tiled Bridge<br/>(persists docs to Tiled)"]
        tiled["Tiled Server<br/>(data storage)"]

        redis <-->|state| rem
        rem -->|publishes documents| zmqp
        zmqp -->|document stream| bridge
        bridge -->|writes| tiled
    end

    agent["Blop QueueserverAgent<br/>(suggests points, listens for completions, evaluates data)"]

    agent -->|submit plans via REManagerAPI| rem
    zmqp -->|document stream| agent
    tiled -->|read data| agent
```

**Data flow:**

1. The agent suggests parameter values and submits an acquisition plan to the RE Manager
1. The RE Manager executes the plan (moves motors, reads detectors)
1. Bluesky documents are published via ZMQ to the proxy
1. The ZMQ-Tiled bridge persists documents to the Tiled server (using [bluesky-tiled-plugins](https://blueskyproject.io/bluesky-tiled-plugins)'s `TiledWriter` callback)
1. The agent's ZMQ listener detects plan completion
1. The agent's evaluation function reads results from Tiled and computes objectives
1. The optimizer ingests outcomes and suggests the next point

## Prerequisites

- Docker and Docker Compose installed
- The `blop` Python package installed (with `bluesky-queueserver-api` and `tiled`)
- The `blop` GitHub repository cloned (for the service definitions under `docs/source/tutorials/queueserver/`)

## Starting the Infrastructure

All services are defined in a `docker-compose.yml` in the `docs/source/tutorials/queueserver/` directory. Before running this tutorial, start the stack in a separate terminal:

```bash
cd docs/source/tutorials/queueserver
docker compose up -d --build
```

Wait until all containers are healthy:

```bash
docker compose ps
```

You should see all services in a "healthy" or "running" state. The services expose the following ports on `localhost`:

| Service | Port | Purpose |
|---------|------|---------|
| RE Manager | 60615 | ZMQ control channel (REManagerAPI connects here) |
| ZMQ Proxy (out) | 5578 | Document stream (agent listens for plan completions) |
| Tiled | 8000 | Data access (evaluation function reads results) |
| Redis | 6379 | Internal message broker for queueserver |

Once the containers are up, proceed with the tutorial below.

```{code-cell} ipython3
import time

from bluesky.callbacks.zmq import RemoteDispatcher
from bluesky_queueserver_api.zmq import REManagerAPI

RM = REManagerAPI(zmq_control_addr="tcp://localhost:60615")
RM.environment_open()
RM.wait_for_idle(timeout=30)
status = RM.status()
print(f"RE Manager state: {status['manager_state']}")
print(f"Worker environment exists: {status['worker_environment_exists']}")
assert status["worker_environment_exists"], "Open the RE environment before continuing (see instructions above)"
```

## The Queueserver Environment

The queueserver startup script (shown below for reference) defines the devices and plans available in the remote environment. This script runs inside the RE Manager process — **not** in your notebook:

```python
# startup.py (runs inside the RE Manager)
from bluesky import RunEngine
from bluesky_queueserver import is_re_worker_active
from ophyd.sim import SynAxis, SynSignal

RE = RunEngine({})

# Publish documents to ZMQ so external subscribers can react
if is_re_worker_active():
    import os
    from bluesky.callbacks.zmq import Publisher as ZmqPublisher

    addr = os.environ.get("BLUESKY_ZMQ_PROXY_IN_ADDR", "tcp://zmq-proxy:5577")
    host, port = addr.replace("tcp://", "").split(":")
    publisher = ZmqPublisher((host, int(port)))
    RE.subscribe(publisher)

# Simulated motors
motor1 = SynAxis(name="motor1", labels={"motors"})
motor2 = SynAxis(name="motor2", labels={"motors"})


# Simulated detector that computes the Himmelblau function
def _compute_himmelblau():
    x = motor1.read()["motor1"]["value"]
    y = motor2.read()["motor2"]["value"]
    return float((x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2)


himmel_det = SynSignal(name="himmel_det", func=_compute_himmelblau, labels={"detectors"})

# Acquisition plan — moves actuators to suggested positions and reads sensors
import bluesky.plans as bp


def default_acquire(suggestions, actuators, sensors, *, md=None):
    plan_args = []
    for actuator in actuators:
        values = [s[actuator.name] for s in suggestions]
        plan_args.append(actuator)
        plan_args.append(values)

    _md = {"blop_suggestions": suggestions}
    if md:
        _md.update(md)

    yield from bp.list_scan(list(sensors), *plan_args, md=_md)
```

Key points:

- `is_re_worker_active()` gates code that should only run inside the queueserver worker
- The ZMQ publisher sends all run documents to the proxy for external consumption
- `default_acquire` is a simple plan that wraps Bluesky's `list_scan` — it moves actuators to each suggested position and reads sensors

## Connecting to Tiled

The evaluation function will read experimental data from the Tiled server:

```{code-cell} ipython3
from tiled.client import from_uri

tiled_client = from_uri("http://localhost:8000", api_key="tutorialkey")
```

## Defining the Optimization Problem

Just as in the simple experiment tutorial, we define **DOFs** and **objectives**. The key difference: since devices exist only in the remote queueserver environment, DOFs reference device names as strings (no `actuator` objects).

```{code-cell} ipython3
from blop.ax import QueueserverAgent, RangeDOF, Objective

dofs = [
    RangeDOF(actuator="motor1", bounds=(-5.0, 5.0), parameter_type="float"),
    RangeDOF(actuator="motor2", bounds=(-5.0, 5.0), parameter_type="float"),
]

objectives = [
    Objective(name="himmelblau", minimize=True),
]

# Sensors are referenced by name — these are the detectors in the queueserver environment
sensors = ["himmel_det"]
```

## Writing the Evaluation Function

The evaluation function is called each time a plan completes. It reads data from Tiled and computes the objective values. The function receives:

- `uid`: the Bluesky run UID (use this to look up data in Tiled)
- `suggestions`: the list of parameter dicts that were evaluated

It must return a list of outcome dicts, each containing the objective value(s) and an `_id` matching the suggestion.

Because the agent and the ZMQ-Tiled bridge are separate subscribers to the same ZMQ stream, there is a race condition: the agent may receive the stop document before the bridge has finished writing data to Tiled. The evaluation function should poll Tiled until both the run and the detector data are available.

```{code-cell} ipython3
import numpy as np
from tiled.client.container import Container


class HimmelblauEvaluation:
    """Reads detector data from Tiled and computes the Himmelblau objective."""

    def __init__(self, tiled_client: Container, timeout: float = 30.0, poll_interval: float = 0.5):
        self.tiled_client = tiled_client
        self.timeout = timeout
        self.poll_interval = poll_interval

    def _wait_for_run(self, uid: str):
        """Poll Tiled until the run with the given UID is available."""
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            try:
                return self.tiled_client[uid]
            except KeyError:
                time.sleep(self.poll_interval)
        raise TimeoutError(
            f"Run '{uid}' not found in Tiled after {self.timeout}s. "
            "The ZMQ-Tiled bridge may not be running."
        )

    def _wait_for_detector_data(self, run, path: str):
        """Poll Tiled until the detector data path is readable."""
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            try:
                return run[path].read()
            except KeyError:
                time.sleep(self.poll_interval)
        raise TimeoutError(
            f"Data path '{path}' for run '{run.metadata['start']['uid']}' was not readable after {self.timeout}s. "
            "The ZMQ-Tiled bridge may still be writing the run."
        )

    def __call__(self, uid: str, suggestions: list[dict]) -> list[dict]:
        run = self._wait_for_run(uid)

        # Read the detector values from the primary data stream
        himmel_values = self._wait_for_detector_data(run, "primary/himmel_det")

        outcomes = []
        for idx, suggestion in enumerate(suggestions):
            outcomes.append({
                "_id": suggestion["_id"],
                "himmelblau": float(himmel_values[idx]),
            })

        return outcomes
```

## Creating the Queueserver Agent

Now we bring everything together. The `QueueserverAgent` needs:

- `re_manager_api`: how to communicate with the queueserver (submit plans, check status)
- `document_dispatcher`: a `RemoteDispatcher` that subscribes to the Bluesky document stream
- The DOFs, objectives, sensors, and evaluation function

```{code-cell} ipython3
document_dispatcher = RemoteDispatcher(("localhost", 5578))

agent = QueueserverAgent(
    re_manager_api=RM,
    document_dispatcher=document_dispatcher,
    sensors=sensors,
    dofs=dofs,
    objectives=objectives,
    evaluation_function=HimmelblauEvaluation(tiled_client),
    acquisition_plan="default_acquire",
)
```

```{note}
The `re_manager_api` argument also accepts an HTTP-based client
(`bluesky_queueserver_api.http.REManagerAPI`) for deployments that expose the
queueserver over HTTP rather than ZMQ. Construct the `RemoteDispatcher` with
whatever arguments your document stream requires, such as a ZMQ address or
message prefix.
```

## Running the Optimization

The `run()` method is **non-blocking** — it submits the first plan and returns immediately. The agent reacts asynchronously to plan completions via ZMQ callbacks.

```{code-cell} ipython3
future = agent.run(iterations=10, n_points=1)
```

Wait for the optimization to complete and inspect the result:

```{code-cell} ipython3
result = future.result()

print(f"Iterations completed : {result.iterations_completed}")
print(f"Points per iteration : {result.num_points}")
print(f"Total acquisitions   : {len(result.uids)}")
print()
print("Run UIDs:")
for uid in result.uids:
    print(f"  {uid}")
```

## Viewing Results

Since `QueueserverAgent` uses the same Ax optimizer backend as the local `Agent`, all the familiar analysis methods are available:

```{code-cell} ipython3
agent.ax_client.summarize()
```

The Himmelblau function has four global minima (all with value 0). The optimizer should have made some progress toward these optima.

- (3.0, 2.0)
- (-2.805, 3.131)
- (-3.779, -3.283)
- (3.584, -1.848)

## Cleanup

When you're done, close the RE environment and stop the Docker services:

```{code-cell} ipython3
RM.environment_close()
RM.wait_for_idle(timeout=30)
RM.close()
```

```bash
cd docs/source/tutorials/queueserver
docker compose down
```

## What You Learned

- **Distributed architecture**: The queueserver separates experiment execution from optimization logic, connected via ZMQ and Tiled
- **String-based device references**: Since devices live in the remote process, DOFs, sensors, and plans are referenced by name
- **Asynchronous operation**: `agent.run()` is non-blocking; the agent reacts to events via ZMQ callbacks
- **Evaluation function**: Reads from Tiled (not direct device access) to compute objectives after each plan completes

## Next Steps

- Add multiple objectives for multi-objective optimization (see [KB Mirrors tutorial](./xrt-kb-mirrors.md))
- Use `agent.submit_suggestions()` to manually evaluate specific parameter combinations (see [](../how-to-guides/manual-suggestions.rst))
- Implement `outcome_constraints` to constrain the optimization (see [](../how-to-guides/set-outcome-constraints.rst))
- Add a `checkpoint_path` to persist optimizer state across restarts (see [](../reference/agent.rst))
