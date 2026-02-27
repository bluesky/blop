---
name: blop-agent
description: Help creating and configuring blop Bayesian optimization agents with DOFs, objectives, and constraints
compatibility: opencode
---

## What I do

Help create blop optimization agents for beamline optimization.

## Basic usage

```python
from blop import Agent, RangeDOF, Objective

agent = Agent(
    dofs=[RangeDOF(name="x", lower=0, upper=10)],
    objectives=[Objective(name="intensity", minimize=False)],
    evaluation_function=lambda: {"intensity": read_sensor()},
)
```

## Key classes

**RangeDOF** - continuous parameter:
```python
RangeDOF(name="x", lower=0, upper=10, actuator=device)
```

**ChoiceDOF** - discrete parameter:
```python
ChoiceDOF(name="filter", values=["red", "green", "blue"])
```

**Objective** - what to optimize:
```python
Objective(name="intensity", minimize=False)  # maximize
Objective(name="width", minimize=True)       # minimize
```

**OutcomeConstraint** - constrain outputs:
```python
OutcomeConstraint("width", "<=", 10)
```

**DOFConstraint** - constrain inputs:
```python
DOFConstraint("x**2 + y**2", "<=", 100)
```

## When to use me

Use me when setting up optimization agents or debugging blop code.
