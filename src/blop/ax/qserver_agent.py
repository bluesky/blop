import logging

###
import threading
import uuid
from collections.abc import Sequence
from typing import Any

from ax.api.types import TParameterization
from bluesky.callbacks import CallbackBase
from bluesky.callbacks.zmq import RemoteDispatcher
from bluesky_queueserver_api import BPlan
from bluesky_queueserver_api.zmq import REManagerAPI

from ..protocols import EvaluationFunction, Sensor
from .agent import Agent as BlopAxAgent  # type: ignore[import-untyped]
from .dof import DOF, DOFConstraint
from .objective import Objective

logger = logging.getLogger("blop")


class ConsumerCallback(CallbackBase):
    """
    A child of Callback base which caches the start document and calls a callback function on the stop document

    """

    def __init__(self, callback: callable = None, enable=True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.start_doc_cache = None
        self.callback = callback  # A function that is called when stop is called
        self.enable = enable

    def start(self, doc):
        if self.enable:
            self.start_doc_cache = doc

    def stop(self, doc):
        if self.enable:
            self.callback(self.start_doc_cache, doc)
            self._clear_cache()

    def _clear_cache(self):
        self.start_doc_cache = None


class ZMQConsumer:
    """
    Allows us to start a thread which will listen to docs and call the callback in CallbackBase
    """

    def __init__(self, zmq_consumer_ip_address, zmq_consumer_port, callback: callable = None):
        self.zmq_consumer_ip_address = zmq_consumer_ip_address
        self.zmq_consumer_port = zmq_consumer_port

        self.zmq_consumer = RemoteDispatcher(f"{self.zmq_consumer_ip_address}:{self.zmq_consumer_port}")
        self.zmq_consumer_callback = ConsumerCallback(callback=callback, enable=True)
        self.zmq_consumer.subscribe(self.zmq_consumer_callback)
        self._zmq_thread = None

    def start_zmq_listener_thread(self):
        logger.info("Starting ZMQ Callback Thread")

        self._zmq_thread = threading.Thread(target=self.zmq_consumer.start, name="zmq-consumer", daemon=True)
        self._zmq_thread.start()


class BlopQserverAgent(BlopAxAgent):
    """
    An interface that uses Ax as the backend for optimization and experiment tracking.

    The Agent is the main entry point for setting up and running Bayesian optimization
    using Blop. It coordinates the DOFs, objectives, evaluation function, and optimizer
    to perform intelligent exploration of the parameter space.

    This class sends JSON strings to a queueserver, rather than emmitting messages to be
    consumed directly by a RE.

    Parameters
    ----------
    sensors : Sequence[Sensor]
        The sensors to use for acquisition. These should be the minimal set
        of sensors that are needed to compute the objectives.
    dofs : Sequence[DOF]
        The degrees of freedom that the agent can control, which determine the search space.
    objectives : Sequence[Objective]
        The objectives which the agent will try to optimize.
    evaluation_function : EvaluationFunction
        The function to evaluate acquired data and produce outcomes.
    acquisition_plan : str, optional
        The name of the plan on the queueserver
    dof_constraints : Sequence[DOFConstraint] | None, optional
        Constraints on DOFs to refine the search space.
    outcome_constraints : Sequence[OutcomeConstraint] | None, optional
        Constraints on outcomes to be satisfied during optimization.
    qserver_control_addr : str, default="tcp://localhost:60615"
        Queueserver Control Address
    qserver_info_addr : str, default="tcp://localhost:60625"
        Queueserver Info Address
    zmq_consumer_ip : str, default= "localhost"
        The IP address of the ZMQ proxy to listen for stop document
    zmq_consumer_port : str, default= "5578"
        The PORT of the ZMQ proxy to listen for stop document
    **kwargs : Any
        Additional keyword arguments to configure the Ax experiment.

    Notes
    -----
    For more complex setups, you can configure the Ax client directly via ``self.ax_client``.

    For complete working examples of creating and using an Agent, see the tutorial
    documentation, particularly :doc:`/tutorials/qserver-experiment`.


    """

    def __init__(
        self,
        sensors: Sequence[Sensor],
        dofs: Sequence[DOF],
        objectives: Sequence[Objective],
        evaluation_function: EvaluationFunction = None,
        acquisition_plan: str = "acquire",
        dof_constraints: Sequence[DOFConstraint] = None,
        qserver_control_addr: str = "tcp://localhost:60615",
        qserver_info_addr: str = "tcp://localhost:60625",
        zmq_consumer_ip: str = "localhost",
        zmq_consumer_port: str = "5578",
        **kwargs: Any,
    ):
        super().__init__(
            sensors=sensors,
            dofs=dofs,
            objectives=objectives,
            evaluation_function=evaluation_function,
            acquisition_plan=acquisition_plan,
            dof_constraints=dof_constraints,
            **kwargs,
        )

        # Store dofs for qserver-specific operations (parent class only stores actuators)
        self._dofs = list(dofs)

        # Instantiate an object that can communicate with the queueserver
        self.RM = REManagerAPI(
            zmq_control_addr=qserver_control_addr, zmq_info_addr=qserver_info_addr
        )  # To Do, Add arguements to class init

        # Should plans be submitted and automatically started, or not?
        self._queue_autostart = True

        # Instantiate an object that will listen for start and stop documents and call a method on the stop document
        self.zmq_consumer = ZMQConsumer(zmq_consumer_ip, zmq_consumer_port, callback=self._stop_doc_callback)
        self.zmq_consumer.start_zmq_listener_thread()

        # Should we do something when there is a new event?
        self._listen_to_events = True

        # Should new suggestions be made automatically until all of the trials are complete?
        self.continuous_suggestion = True

        # Learning parameters
        self.num_itterations = 30
        self.n_points = 1

        # Variables used to keep track of the current optimization
        self.current_itteration = 0
        self.agent_suggestion_uid = None
        self.trials = None
        self.acquisition_finished = False
        self.optimization_problem = None

    @property
    def dofs(self) -> Sequence[DOF]:
        """The degrees of freedom for this agent."""
        return self._dofs

    def _stop_doc_callback(self, start_doc, stop_doc):
        """
        In here we can decide whether our experiment requested has completed

        If it has completed, we can digest the data from it and move on to the next point.
        """

        if self._listen_to_events:
            # Mark the current acquisition as finished

            logger.info("A stop document has been received, evaluating")

            # Evaluate it with the evaluation function
            outcomes = self.optimization_problem.evaluation_function(uid=self.agent_suggestion_uid, suggestions=self.trials)

            logger.info(f"successfully evaluated id: {self.agent_suggestion_uid}")

            self.acquisition_finished = True
            # ingest the data, updating the model of the optimizer
            self.optimization_problem.optimizer.ingest(outcomes)

            # After this is complete, call gen_next_trials again if required
            if self.continuous_suggestion:
                if self.current_itteration < self.num_itterations:
                    logger.info("making another suggestion")
                    self.suggest()
                else:
                    self.current_itteration = 0
                    logger.info("made all required suggestions")

    def optimize(self, iterations=1, n_points=1):
        """
        This method will create the optimization problem, suggest points and execute them in the QS
        """

        # Before we do anything check the connection to the Queueserver
        status = self.RM.status()
        if not status["worker_environment_exists"]:
            raise ValueError("The queueserver environment is not open")

        # Check that the devices we want to interact with are in the queueserver environment
        res = self.RM.devices_allowed()
        for dof in self.dofs:
            if dof.parameter_name not in res["devices_allowed"]:
                raise ValueError(f"The device {dof.parameter_name} is not in the Queueserver Environment")

        for sensor in self.sensors:
            # Handle both sensor objects (with .name) and string sensor names
            sensor_name = sensor.name if hasattr(sensor, "name") else sensor
            if sensor_name not in res["devices_allowed"]:
                raise ValueError(f"The device {sensor_name} is not in the Queueserver Environment")

        # Check that the plan we want to call is in the queueserver environment
        res = self.RM.plans_allowed()
        if self._acquisition_plan not in res["plans_allowed"]:
            raise ValueError(f"The plan {self._acquisition_plan} is not in the Queueserver Environment")

        # Form the problem and start suggesting points to measure at
        self.optimization_problem = self.to_optimization_problem()
        self.num_itterations = iterations
        self.n_points = n_points

        self.suggest()

    def suggest(self):
        """
        get suggestions from the optimizer, then send them to the plan on the queueserver

        """

        # record this itteration
        self.current_itteration = self.current_itteration + 1

        # Get the trials to perform
        self.trials = self.optimization_problem.optimizer._client.get_next_trials(self.n_points)

        # acquire the values from those trials
        self.agent_suggestion_uid = self.acquire(self.trials)
        logger.info(
            f"sending suggestion {self.current_itteration} to queueserver with suggestion id: {self.agent_suggestion_uid}"
        )

    def acquire_baseline(self, parameterization: dict[str, Any] | None = None):
        logger.info("This is not implemented for the qserver agent")

    def acquire(self, trials: dict[int, TParameterization] | None = None):
        """
        Acquire the new data from the system by submitting the suggested
        points to the queueserver. This method does not block while the
        queueserver is running.

        Parameters
        ----------
        trials : dict[int, TParameterization]
            A dictionary mapping trial indices to their suggested parameterizations.
        """

        try:
            self.acquisition_finished = False

            # Create a unique identifier which will connect the children to the parent batch
            # This batch ID will be used by all runs from this request by the agent.
            # It will be used by the EvaluationFunction later to work out what happened.

            agent_suggestion_uid = str(uuid.uuid4())
            kwargs = {}
            kwargs.setdefault("md", {})

            # Add the unique suggestion ID so we can find this run later
            kwargs["md"]["agent_suggestion_uid"] = agent_suggestion_uid

            # Add the suggestion _id key so we can work out which number we are on later
            suggestions = [
                {
                    "_id": trial_index,
                    **parameterization,
                }
                for trial_index, parameterization in trials.items()
            ]
            kwargs["md"]["blop_suggestions"] = suggestions

            # Create the BPlan object to send to the queue. Convert dofs to strings
            item = BPlan(
                self.acquisition_plan,
                readables=self.sensors,
                dofs=[dof.parameter_name for dof in self.dofs],
                trials=trials,
                md=kwargs["md"],
            )

            # Send the plan to the Run Engine Manager
            r = self.RM.item_add(item)
            logger.debug(
                f"Sent http-server request for trials {trials}"
                f"with agent_suggestion_uid= {agent_suggestion_uid}\n.Received reponse: {r}"
            )

            # If the queue should start automatically, then start the queue.
            if self._queue_autostart:
                logger.debug("Waiting for Queue to be idle or paused")
                self.RM.wait_for_idle_or_paused(timeout=600)
                r = self.RM.queue_start()
                logger.debug(f"Sent http-server request to start the queue\n.Received reponse: {r}")

        except KeyboardInterrupt as interrupt:
            raise interrupt

        return agent_suggestion_uid

    def stop(self):
        """
        Stop the agent if it is running
        """
        self._queue_autostart = False
        self._listen_to_events = False
