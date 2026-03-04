.. testsetup::
    
    from typing import Any
    import time
    import logging

    from bluesky.protocols import NamedMovable, Readable, Status, Hints, HasHints, HasParent
    from bluesky.run_engine import RunEngine
    from bluesky_tiled_plugins import TiledWriter
    from tiled.client import from_uri
    from tiled.server import SimpleTiledServer

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
            return { 
                "fields": [self._name],
                "dimensions": [],
                "gridding": "rectilinear",
            }
        
        @property
        def parent(self) -> Any | None:
            return None

        def read(self):
            return {
                self._name: { "value": self._value, "timestamp": time.time() }
            }

        def describe(self):
            return {
                self._name: { "source": self._name, "dtype": "number", "shape": [] }
            }

    class MovableSignal(ReadableSignal, NamedMovable):
        def __init__(self, name: str, initial_value: float = 0.0) -> None:
            super().__init__(name)
            self._value: float = initial_value

        def set(self, value: float) -> Status:
            self._value = value
            return AlwaysSuccessfulStatus()

    server = SimpleTiledServer()
    logging.getLogger("httpx").setLevel(logging.WARNING)
    db = from_uri(server.uri)
    tiled_writer = TiledWriter(db)
    RE = RunEngine({})
    RE.subscribe(tiled_writer)

    movable1 = MovableSignal("movable1")
    movable2 = MovableSignal("movable2")
    movable3 = MovableSignal("movable3")
    readable1 = ReadableSignal("objective1")
    readable2 = ReadableSignal("objective2")

.. testcleanup::

    # Suppress stdout from server.close() otherwise the doctest will fail
    import os
    import contextlib

    with contextlib.redirect_stdout(open(os.devnull, "w")):
        server.close()

Using custom generation strategies
==================================

This guide will show you how to use custom generation strategies with BoTorch, Blop, and Ax.

Configure an agent
------------------

.. testcode::

    from blop.ax import Agent, RangeDOF, Objective

    dofs = [
        RangeDOF(actuator=movable1, bounds=(-5.0, 5.0), parameter_type="float"),
        RangeDOF(actuator=movable2, bounds=(-5.0, 5.0), parameter_type="float"),
    ]

    objectives = [
        Objective(name="objective1", minimize=False),
    ]

    def evaluation_function(uid: str, suggestions: list[dict]) -> list[dict]:
        """Replace this with your own evaluation function."""
        outcomes = []
        for suggestion in suggestions:
            outcome = {
                "_id": suggestion["_id"],
                "objective1": 0.1,
            }
            outcomes.append(outcome)
        return outcomes

    agent = Agent(
        sensors=[readable1, readable2],
        dofs=dofs,
        objectives=objectives,
        evaluation_function=evaluation_function,
    )

Configure a generation strategy
-------------------------------

The following example shows a generation strategy that uses the Sobol generator for the first 10 trials, and then uses the ``LatentGP`` model for the remaining trials.

For more information on generation strategies, see the `Ax documentation <https://ax.dev/docs/generation_strategy>`_. This is essentially a thin wrapper around the Ax generation strategy API. There are many different options to configure which we will not cover in depth here.

.. note::
    
    The is not part of Ax's backward compatibile API. The ``GenerationStrategy`` may be subject to breaking changes in future versions of Ax.

.. testcode::

    from ax.generation_strategy.generation_node import GenerationNode
    from ax.generation_strategy.generation_strategy import GenerationStrategy
    from ax.generation_strategy.generator_spec import GeneratorSpec
    from ax.generation_strategy.transition_criterion import MinTrials
    from ax.adapter.registry import Generators
    from ax.generators.torch.botorch_modular.surrogate import SurrogateSpec
    from ax.generators.torch.botorch_modular.utils import ModelConfig
    from botorch.acquisition.logei import qLogNoisyExpectedImprovement

    from blop.bayesian.models import LatentGP


    generation_strategy = GenerationStrategy(
        name="Custom Generation Strategy",
        nodes=[
            GenerationNode(
                name="Sobol",
                generator_specs=[
                    GeneratorSpec(generator_enum=Generators.SOBOL, model_kwargs={"seed": 0}),
                ],
                transition_criteria=[
                    MinTrials(
                        threshold=10,
                        transition_to="LatentGP",
                        use_all_trials_in_exp=True,
                    ),
                ],
            ),
            GenerationNode(
                name="LatentGP",
                generator_specs=[
                    GeneratorSpec(
                        generator_enum=Generators.BOTORCH_MODULAR,
                        model_kwargs={
                            "surrogate_spec": SurrogateSpec(
                                model_configs=[
                                    ModelConfig(
                                        botorch_model_class=LatentGP,
                                        input_transform_classes=None,
                                        model_options={},
                                    ),
                                ],
                            ),
                            "botorch_acqf_class": qLogNoisyExpectedImprovement,
                            "acquisition_options": {},
                        },
                        model_gen_kwargs={
                            "optimizer_kwargs": {
                                "num_restarts": 10,
                                "sequential": True,
                            },
                        },
                    ),
                ],
            ),
        ],
    )

Configure the experiment and set the generation strategy
--------------------------------------------------------

.. testcode::

    agent.ax_client.set_generation_strategy(generation_strategy)

Run the experiment with Bluesky
-------------------------------

.. testcode::

    RE(agent.optimize(iterations=12, n_points=1))


Verify the generation strategy was used
---------------------------------------

.. testcode::

    df = agent.ax_client.summarize()
    assert "LatentGP" in df["generation_node"].values
