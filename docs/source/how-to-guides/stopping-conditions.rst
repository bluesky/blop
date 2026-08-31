Set Stopping Conditions
=======================

This guide shows how to set stopping conditions for an optimization run for the Ax optimizer backend. 
These conditions can be a variety of methods such as meeting a certain toleration, reaching a maximum number of iterations, or exceeding a time limit

Defining Stopping Conditions
----------------------------

Ax has built-in stopping conditions, `found here <https://github.com/facebook/Ax/blob/6e711c86a162f8fc5173b0d5e18f0326c329bac7/ax/global_stopping/strategies/__init__.py>`_.

However, if you need a more customizable stopping condition, you can configure one following this format.

.. code-block:: python

    from ax.global_stopping.strategies.base import BaseGlobalStoppingStrategy
    from ax.core.experiment import Experiment

    class CustomStoppingStrategy(BaseGlobalStoppingStrategy):
        def __init__(
            self,
            min_trials: int = 1,
            inactive_when_pending_trials: bool = True,
            # Add any additional parameters you want to customize here
        ) -> None:
            super().__init__(
                min_trials=min_trials, 
                inactive_when_pending_trials=inactive_when_pending_trials
            )
            # Initialize your custom parameters here

        def _should_stop_optimization(self, experiment: Experiment, **kwargs) -> tuple[bool, str]:
            # Implement your custom stopping logic here
            if your_condition_is_met:
                return True, "stopping criteria met"
            return False, "stopping criteria not met"


Adding Stopping Conditions to the Agent
---------------------------------------

The Agent can then be configured with an additional parameter, `stopping_strategy`, to use the custom stopping condition.

To run until the stopping condition is met, pass ``iterations=None`` to
:meth:`blop.ax.Agent.optimize`. Only use an unbounded run when the agent has a
stopping strategy configured; otherwise the optimization has no termination
condition.

.. code-block:: python

    agent = Agent(
        ...,
        stopping_strategy=CustomStoppingStrategy(),
    )
    RE(agent.optimize(iterations=None))

A finite ``iterations`` value remains a maximum: the stopping strategy may end
the run before that limit is reached.
