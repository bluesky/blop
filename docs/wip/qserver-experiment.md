---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.18.1
kernelspec:
  display_name: .venv
  language: python
  name: python3
---

# Himmelblau Queueserver Demo

In this demo we will make the himmelblau example, but with ophyd devices inside a remote Queueserver. The Queueserver allows us to run the agent in another process, seperated from the experiment environment. The devices in the Queueserver include:

- A detector whose output as a function of x,y is the himmelblau function
- a pair sim motors which the detector uses as inputs

## Queueserver Configuration

The Queueserver should have a RE which sends documents to a ZMQ buffer and then a Tiled server. In this example the Queueserver is communicated with over ZMQ.

We will:

- Use the ZMQ buffer to find a stop document that will tell us a trial point has been measured
- Use the Tiled client to access the data

You will need:

- The Queueserver IP and control port e.g. `tcp://localhost:60615`
- The Queueserver IP and info port e.g. `tcp://localhost:60625`
- The Tiled Server IP, port and password e.g. `http://localhost:8000`
- The ZMQ Buffer IP and port e.g. `localhost:5578`

+++

### The Ophyd devices in the Queueserver Environment

The following devices should be made available in the Queueserver Environment.

```python
from ophyd import Device, Component as Cpt, Signal
from blop.utils.functions import himmelblau
from ophyd.sim import motor1, motor2
from ophyd.sim import SynGauss, SynSignal, EnumSignal
import numpy as np


class SynHimmelblauDetector(Device):
    """
    Evaluate a point on a Gaussian based on the value of a motor.

    Parameters
    ----------
    name : str
        The name of the detector
    motor0 : SynAxis
        The 'x' coordinate of the 2-D gaussian blob
    motor_field0 : str
        The name field of the motor. Should be the key in motor0.describe()
    motor1 : SynAxis
        The 'y' coordinate of the 2-D gaussian blob
    motor_field1 : str
        The name field of the motor. Should be the key in motor1.describe()
    noise : {'poisson', 'uniform', None}, optional
        Add noise to the gaussian peak..
        Defaults to None
    noise_multiplier : float, optional
        Only relevant for 'uniform' noise. Multiply the random amount of
        noise by 'noise_multiplier'
        Defaults to 1
    random_state : numpy random state object, optional
        np.random.RandomState(0), to generate random number with given seed

    Example
    -------
    motor1 = SynAxis(name='motor1')
    motor2 = SynAxis(name='motor2')
    det = SynHimmelblauDetector('det', motor1, 'motor1', motor2, 'motor2)
    """

    val = Cpt(SynSignal, kind="hinted")
    noise = Cpt(
        EnumSignal,
        value="none",
        kind="config",
        enum_strings=("none", "poisson", "uniform"),
    )
    noise_multiplier = Cpt(Signal, value=1, kind="config")

    def _compute(self):

        # Get the current values of the motors
        x = self._motor0.read()[self._motor_field0]["value"]
        y = self._motor1.read()[self._motor_field1]["value"]
        m = np.array([x, y])

        noise = self.noise.get()
        noise_multiplier = self.noise_multiplier.get()

        v = himmelblau(x, y)
        if noise == "poisson":
            v = int(self.random_state.poisson(np.round(v), 1))
        elif noise == "uniform":
            v += self.random_state.uniform(-1, 1) * noise_multiplier
        return v

    def __init__(
        self,
        name,
        motor0,
        motor_field0,
        motor1,
        motor_field1,
        noise="none",
        noise_multiplier=1,
        random_state=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self._motor0 = motor0
        self._motor1 = motor1
        self._motor_field0 = motor_field0
        self._motor_field1 = motor_field1
        self.noise.put(noise)
        self.noise_multiplier.put(noise_multiplier)

        if random_state is None:
            random_state = np.random
        self.random_state = random_state
        self.val.name = self.name
        self.val.sim_set_func(self._compute)

        self.trigger()

    def trigger(self, *args, **kwargs):
        return self.val.trigger(*args, **kwargs)


himmel_det = SynHimmelblauDetector(
    "himmel_det", motor1, "motor1", motor2, "motor2", labels={"detectors"}, noise="uniform", noise_multiplier=0.01
)
```

+++

### Plans in the Queueserver Environment

The Queueserver environment has the plan `acquire` which wraps the `list_scan` plan like this:

```python
from blop.plans import TParameterization, Movable, TParameterValue, defaultdict


def _unpack_parameters(dofs: list[Movable], parameterizations: list[TParameterization]) -> list[Movable | TParameterValue]:
    """Unpack the parameterizations into Bluesky plan arguments."""
    unpacked_dict = defaultdict(list)
    for parameterization in parameterizations:
        for dof in dofs:
            if dof.name in parameterization:
                unpacked_dict[dof.name].append(parameterization[dof.name])
            else:
                raise ValueError(f"Parameter {dof.name} not found in parameterization. Parameterization: {parameterization}")

    """ create a dict of dofs"""

    dofs_dict = {}
    for dof in dofs:
        dofs_dict[dof.name] = dof

    """ Finally, create a list of dofs and setpoints """
    unpacked_list = []
    for dof_name, values in unpacked_dict.items():
        unpacked_list.append(dofs_dict[dof_name])
        unpacked_list.append(values)

    return unpacked_list


def acquire(readables, dofs, trials: dict, md=None):

    plan_args = _unpack_parameters(dofs, trials.values())

    yield from list_scan(readables, *plan_args, per_step=None, md=md)
```

+++

## Remote Blop Agent

The remote blop agent needs to know:

- The details of the Queueserver (to send instructions to)
- The details of the tiled server (to get data from)
- The details of the ZMQ buffer (to know when the instructions are complete)

Before you continue with the tutorial, make sure each of these are set up and running

```{code-cell} ipython3
# Connect to our Tiled Server

from tiled.client import from_uri
tiled_client = from_uri("http://localhost:8000", api_key='secret')
```

### Configuration of the Blop Agent

Just as in the other tutorials, we have to configure the DOFS, Objectives and Sensors.

Unlike the other tutorials, all of these are now just strings because the objects to the real devices exist only in the Queueserver environment.

```{code-cell} ipython3
from blop.ax.dof import RangeDOF
from blop.ax.objective import Objective

dofs = [
    RangeDOF(name='motor1', bounds=(-6.0, 6.0), parameter_type="float"),
    RangeDOF(name='motor2', bounds=(-6.0, 6.0), parameter_type="float"),
]

# This is the objective that our evaluation function will produce
objectives = [
    Objective(name="himmel_det_objective",minimize=True),
  
]

# This is the list of devices we want to read from in the Queueserver env
sensors = ['himmel_det']
```

### Making an Evaluation Function

After the agent has suggested points and run them on the Queueserver, we want to update our model with the results. The EvaluationFunction defines how data is read from a bluesky run and the objective values are created. Unlike the blop agent which produces messages to be consumed by a run engine, the blop Queueserver agent is operating with a process which it doesn't, or at least might not, solely control. It's possible that the agent will submit plans to a queue that is not empty. We therefore have to search for a specific key value pair in a stop document.

The EvaluationFunction is called every time a stop document is recieved. It must include some check to see if this particular stop document is from the plan that was submitted previously by the agent. The `BlopQserverAgent` adds a key `agent_suggestion_uid` to the start document `md` dict. This `agent_suggestion_uid` is passed to the `EvaluationFunction` as the argument `uid`.

```{code-cell} ipython3
from blop.protocols import EvaluationFunction
from tiled.client.container import Container
from tiled.queries import Eq
from blop.ax import QueueserverAgent 
import numpy as np

class DetectorEvaluation(EvaluationFunction):
    def __init__(self, tiled_client: Container):
        self.tiled_client = tiled_client

    def __call__(self, uid: str, suggestions: list[dict]) -> list[dict]:
        
        """
        The uid is the suggestion id, we have to search for it since it might not be the last one run on the queue
        """
        outcomes = []
        
        # Search the database for the required run
        results_db = self.tiled_client.search(Eq("agent_suggestion_uid", uid))
        
        if len(results_db) == 1:
            run =  results_db[-1]
               
            # Read the data from the detector in the Queueserver environment
            himmel_det = run["primary/data/himmel_det"].read()

            # Read the suggestion ID's from the metadata
            suggestion_ids = [suggestion["_id"] for suggestion in run.metadata["start"]["blop_suggestions"]]

            # Create a list of dicts containg the results. Note that the name is the name of the Objective defined earlier
            for idx, sid in enumerate(suggestion_ids):
                outcome = {
                    "_id": sid,
                    "himmel_det_objective": np.log(himmel_det[idx])
                }
                outcomes.append(outcome)
            
        elif len(results_db) >1:
            raise ValueError(f"There are {len(results_db)} runs matching the required suggestion id: {uid} ")
        else: 
            print('This stop document was from a different run')
            
        return outcomes
```

### Create the agent

Finally we put everything together, instantiate the agent and start an optimization.

```{code-cell} ipython3

agent = QueueserverAgent(
    sensors=sensors,                                # The list of sensors to read from
    dofs=dofs,                                      # The list of DOFs to search over 
    objectives=objectives,                          # The list of objectives to be optimized
    evaluation_function= DetectorEvaluation(tiled_client),   # The function to create objective function values
    acquisition_plan= "acquire",                    # The name of the plan in the Queueserver environment
    Queueserver_control_addr="tcp://localhost:60615",
    Queueserver_info_addr="tcp://localhost:60625",
    zmq_consumer_ip= "localhost",
    zmq_consumer_port= "5578", 
)
```

```{code-cell} ipython3
# Configure the agent as required
agent.ax_client.configure_generation_strategy(initialization_budget=8)
```

### Make sure the RE environment is open

```{code-cell} ipython3
status = agent.RM.status()
print(status)
```

```{code-cell} ipython3
if not status["worker_environment_exists"]:
    agent.RM.environment_open()
```

```{code-cell} ipython3
# Wait a bit for it to start:
status = agent.RM.status()
print(f"The environment status is: {status['worker_environment_exists']}")
```

### Run the optimization task

```{code-cell} ipython3
# Start an optimization run. Note that this is not blocking because it is interacting with the remote Queueserver. 
agent.optimize(iterations=15, n_points=1)
```

## Data Evaluation

Since the `QueueserverAgent` class is just a child of the `Agent` class, all of the useful methods are still available

```{code-cell} ipython3
agent.plot_objective("motor1", "motor2", "himmel_det_objective")
```

```{code-cell} ipython3
agent.ax_client.summarize()
```
