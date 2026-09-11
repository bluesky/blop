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
- You need asynchronous, non-blocking optimization (the agent submits plans and delegates data readiness to an evaluator)

We will optimize the same Himmelblau function from the [simple experiment tutorial](./simple-experiment.md), but now the devices live inside a remote queueserver process rather than in the same Python session as the agent.

## Architecture

The application uses five infrastructure components:

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

    agent["Blop QueueserverAgent<br/>(suggests points, submits plans, evaluates data)"]

    agent -->|submit plans via REManagerAPI| rem
    tiled -->|correlation lookup and data readiness| agent
```

**Data flow:**

1. The agent suggests parameter values and submits an acquisition plan with a Blop correlation UID to the RE Manager.
1. As soon as submission succeeds, the agent calls its evaluator with a `QueueserverAcquisition` token and the suggestions. The token identifies the submission; data may not exist yet.
1. Independently, the RE Manager executes the plan (moves motors, reads detectors) and publishes Bluesky documents via ZMQ to the proxy.
1. The ZMQ-Tiled bridge persists documents to Tiled using [bluesky-tiled-plugins](https://blueskyproject.io/bluesky-tiled-plugins)'s `TiledWriter` callback.
1. The evaluator searches Tiled by the token's correlation UID and waits for the expected detector rows before computing objectives. It does not wait for a stop document.
1. The optimizer ingests the outcomes, finishes any checkpoint, and suggests the next batch.

The agent needs no document-stream subscriber. The independent ZMQ-Tiled bridge is still required to store the data that this evaluator reads.

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
| ZMQ Proxy (out) | 5578 | Document stream (the ZMQ-Tiled bridge subscribes for persistence) |
| Tiled | 8000 | Data access (evaluation function reads results) |
| Redis | 6379 | Internal message broker for queueserver |

Once the containers are up, proceed with the tutorial below.

```{code-cell} ipython3
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

    # This plan scans the input sequence in order. Plans that reorder points must
    # construct this ID list after reordering.
    _md = {"blop_suggestions": suggestions, "blop_acquisition_order": [s["_id"] for s in suggestions]}
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
from blop.ax import RangeDOF, Objective
from blop.ax.queueserver_agent import QueueserverAgent

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

The evaluation function is called immediately after a successful plan submission, before data necessarily exists. It accepts a `QueueserverAcquisition` token and a sequence of suggestion mappings, and returns a sequence of outcome mappings. The token's `correlation_uid` is injected into the plan's `blop_correlation_uid` metadata; its `item_uid` identifies the Queue Server item, and its `plan_name` names the submitted plan. None of these fields is a Bluesky run UID. Tokens are immutable and hashable, so they can also key evaluator-side caches.

This evaluator owns readiness: it polls Tiled for exactly one run matching the correlation UID, then waits for the detector array to contain exactly the expected number of rows. The tutorial's plan emits one run per acquisition; multiple matching runs are an error rather than an arbitrary choice. Once the run is available, its `blop_acquisition_order` provides the expected row count and maps detector values to suggestion IDs. Suggestions alone do not determine acquisition order. Each outcome must contain the objective value(s) and an `_id` from that acquisition order.

Both waits are bounded by the evaluator's timeout. Missing or incomplete data may become ready while the plan is still running, so no stop document is required. Other backends can implement their own readiness policy without constructing a dispatcher.

```{code-cell} ipython3
from collections.abc import Mapping, Sequence
import time

import numpy as np
from tiled.client.container import Container
from tiled.queries import Eq

from blop.queueserver import QueueserverAcquisition


class HimmelblauEvaluation:
    """Reads detector data from Tiled and computes the Himmelblau objective."""

    def __init__(self, tiled_client: Container, timeout: float = 30.0, poll_interval: float = 0.5):
        self.tiled_client = tiled_client
        self.timeout = timeout
        self.poll_interval = poll_interval

    def _wait_for_run(self, uid: QueueserverAcquisition) -> Container:
        """Poll Tiled until exactly one run matches the acquisition correlation."""
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            matches = self.tiled_client.search(Eq("start.blop_correlation_uid", uid.correlation_uid))
            count = len(matches)
            if count == 1:
                return next(iter(matches.values()))
            if count > 1:
                raise RuntimeError(f"Expected one run for acquisition {uid.correlation_uid!r}, found {count}.")
            time.sleep(self.poll_interval)
        raise TimeoutError(
            f"Acquisition {uid.correlation_uid!r} was not found in Tiled after {self.timeout}s."
        )

    def _wait_for_detector_data(self, run: Container, path: str, expected_rows: int) -> np.ndarray:
        """Poll Tiled until the detector array contains all expected rows."""
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            try:
                data = run[path].read()
            except KeyError:
                pass
            else:
                if len(data) == expected_rows:
                    return data
                if len(data) > expected_rows:
                    raise ValueError(f"Expected {expected_rows} rows at {path!r}, got {len(data)}.")
            time.sleep(self.poll_interval)
        raise TimeoutError(
            f"Data path '{path}' for run '{run.metadata['start']['uid']}' was not readable after {self.timeout}s. "
            "The ZMQ-Tiled bridge may still be writing the run."
        )

    def __call__(self, uid: QueueserverAcquisition, suggestions: Sequence[Mapping]) -> Sequence[Mapping]:
        run = self._wait_for_run(uid)

        acquisition_order = run.metadata["start"]["blop_acquisition_order"]

        # Wait for every detector row before associating values with acquired IDs.
        himmel_values = self._wait_for_detector_data(run, "primary/himmel_det", len(acquisition_order))
        outcomes = []
        for idx, suggestion_id in enumerate(acquisition_order):
            outcomes.append({
                "_id": suggestion_id,
                "himmelblau": float(himmel_values[idx]),
            })

        return outcomes
```

## Creating the Queueserver Agent

Now we bring everything together. The `QueueserverAgent` needs:

- `re_manager_api`: how to communicate with the queueserver (submit plans, check status)
- The DOFs, objectives, sensors, and evaluation function

```{code-cell} ipython3
agent = QueueserverAgent(
    re_manager_api=RM,
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
queueserver over HTTP rather than ZMQ. This native-token evaluator needs only
Tiled access, not a document dispatcher. Queue Server enforces plan and device
permissions; the agent does not maintain a separate local allowlist.
```

### Optional migration for run-UID evaluators

If an existing evaluator expects a Bluesky run UID string, explicitly wrap it in `DocumentStreamEvaluator`. This optional adapter waits for a correlated successful start/stop pair and passes the real run UID to the wrapped evaluator. It is a single-run adapter, not a multi-run aggregation policy. Failed or aborted runs raise an error instead of calling the wrapped evaluator; the configured timeout bounds the document wait, not the wrapped evaluation.

In the illustrative example below, `document_dispatcher` is an application-owned Bluesky `Dispatcher` (such as a `RemoteDispatcher`) whose transport is already running, and `existing_run_uid_evaluator` is your existing callable. Construct the adapter before submitting any acquisitions so it can buffer early documents. Neither the agent nor the adapter starts or stops the transport, and there is no implicit evaluator adaptation.

```python
from contextlib import closing

from blop.queueserver import DocumentStreamEvaluator

with closing(DocumentStreamEvaluator(document_dispatcher, existing_run_uid_evaluator, timeout=30.0)) as evaluation_function:
    stream_agent = QueueserverAgent(
        re_manager_api=RM,
        sensors=sensors,
        dofs=dofs,
        objectives=objectives,
        evaluation_function=evaluation_function,
        acquisition_plan="default_acquire",
    )
    stream_future = stream_agent.run(iterations=10, n_points=1)
    stream_result = stream_future.result()
```

Keep `future.result()` inside the `closing` scope so completion or an exception releases the adapter's owned subscription and buffered documents. Other dispatcher subscribers are unaffected. `stream_agent.stop()` does not close the adapter: wait for the in-flight evaluation to finish before leaving its scope. The application remains responsible for the dispatcher transport's lifecycle. The primary tutorial below continues to use the native Tiled evaluator, without this adapter.

## Running the Optimization

The `run()` method is **non-blocking**: after synchronous preflight checks, it reserves the run and returns a running `Future`. A worker performs even the first suggestion and submission, then calls the token evaluator, ingests its outcomes, and finishes any checkpoint before continuing. `submit_suggestions()` uses the same worker for a single manual batch.

Preflight failures, such as an unavailable worker environment or an optimization already in progress, are raised by `run()` or `submit_suggestions()` itself. After launch, suggestion, registration, submission, evaluation, ingestion, and checkpoint errors are raised by `future.result()`. Queue Server request and transport errors are not replaced by local permission checks.

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
print("Acquisition tokens:")
for uid in result.uids:
    print(f"  {uid}")
```

`result.uids` is an ordered tuple of `QueueserverAcquisition` tokens for successfully evaluated and ingested acquisitions, not a list of run UIDs for direct Tiled lookup. Use the correlation UID to locate a run as shown above. The future remains running through evaluation, ingestion, checkpointing, and any pending failure notification.

To prevent later acquisitions, call `agent.stop()`. It may wait for an already-started submission to return, but does not wait for the evaluator, interrupt the current acquisition, or issue a Queue Server stop/abort. Once submitted, that acquisition still evaluates and ingests, or its error propagates through the future. The future stays pending until this work finishes; `future.cancel()` cannot cancel a running optimization. New `run()` and `submit_suggestions()` calls are rejected during this interval. After completion the agent can be reused, and calling `stop()` again does not change the completed result.

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

When you're done, wait for the RE Manager to become idle before closing its environment: the optimization future may finish as soon as the final detector data is ready, while the acquisition plan is still cleaning up. Then stop the Docker services:

```{code-cell} ipython3
RM.wait_for_idle(timeout=30)
RM.environment_close()
RM.wait_for_idle(timeout=30)
RM.close()
```

```bash
cd docs/source/tutorials/queueserver
docker compose down
```

## What You Learned

- **Distributed architecture**: Queue Server separates experiment execution from optimization logic; the independent ZMQ-Tiled bridge persists data without an agent-side document subscriber
- **String-based device references**: Since devices live in the remote process, DOFs, sensors, and plans are referenced by name
- **Asynchronous operation**: `agent.run()` returns a future; its worker submits plans and waits for evaluator-defined readiness, ingestion, and checkpoint completion
- **Evaluation function**: Receives an acquisition token immediately after submission and polls Tiled for complete, ID-associated data without requiring a stop document
- **Optional document streams**: Existing run-UID evaluators can opt into `DocumentStreamEvaluator`, with application-managed transport and explicit subscription cleanup

## Next Steps

- Add multiple objectives for multi-objective optimization (see [KB Mirrors tutorial](./xrt-kb-mirrors.md))
- Use `agent.submit_suggestions()` to manually evaluate specific parameter combinations (see [](../how-to-guides/manual-suggestions.rst))
- Implement `outcome_constraints` to constrain the optimization (see [](../how-to-guides/set-outcome-constraints.rst))
- Add a `checkpoint_path` to persist optimizer state across restarts (see [](../reference/ax/agent.rst))
