---
name: blop-overview
description: Overview of the Blop project structure, workflow, and use-cases
compatibility: opencode
---

## What is Blop?

Blop (Beamline Optimization Package) is a Python library for Bayesian
optimization of experimental systems, particularly beamline experiments at
synchrotron facilities. It bridges optimization algorithms (Ax platform,
BoTorch, GPyTorch) with the Bluesky ecosystem for data acquisition and device
control, enabling efficient exploration of expensive-to-evaluate parameter
spaces.

## Project structure

- `src/blop/ax/` — primary user-facing API. Built on the Ax platform.
  - `agent.py` — `Agent` is the central optimization driver;
    `_AxAgentMixin` holds shared logic; `QueueserverAgent` is the
    bluesky-queueserver variant.
  - `dof.py` — degrees of freedom: abstract base `DOF` plus concrete
    `RangeDOF` and `ChoiceDOF`, and `DOFConstraint`.
  - `objective.py` — `Objective`, `ScalarizedObjective`, and
    `OutcomeConstraint`.
  - `optimizer.py` — wraps the Ax `Client` / generation strategy.
- `src/blop/bayesian/` — older/lower-level Bayesian-optimization components
  and GP models. Excluded from `pyright` — treat as legacy; prefer `ax/` for
  new work.
- `src/blop/plans.py` — Bluesky plans. Key entry points: `optimize`,
  `optimize_step`, `sample_suggestions`, `acquire_baseline`, and
  `default_acquire` for the standard "move actuators, read detectors"
  cycle.
- `src/blop/plan_stubs.py` — reusable plan fragments.
- `src/blop/protocols.py` — `Actuator`, `OptimizationProblem`, `ID_KEY`, and
  other typing protocols. Use these in new public APIs instead of `Any`.
- `src/blop/queueserver.py` — bluesky-queueserver integration helpers.
- `src/blop/callbacks/` — RunEngine callbacks (plotting, logging, etc.).
- `sim/` — separate `blop-sim` package providing simulated hardware for
  tests and tutorials.

When you need an exact location, grep for the symbol name (e.g.
`rg "^class Agent" src/blop/ax/`) rather than relying on line numbers.

## Basic workflow

1. **Define DOFs** with `RangeDOF(name=..., lower=..., upper=...)` or
   `ChoiceDOF(name=..., values=[...])`.
1. **Define objectives** with `Objective(name=..., minimize=False)` or
   `ScalarizedObjective(...)` for weighted multi-objective problems.
1. **(Optional) Add constraints** — `DOFConstraint` for parameter-space
   bounds, `OutcomeConstraint` for measurement-space bounds.
1. **Construct an `Agent`** with the DOFs, objectives, actuators,
   detectors, and a `dofs_to_actuators` mapping.
1. **Run optimization** by handing a plan from `src/blop/plans.py`
   (typically `optimize`) to the Bluesky `RunEngine`. Each step asks the
   agent for suggestions, moves actuators, reads detectors, and tells the
   agent the result.
1. **Analyze** via the agent's introspection helpers and standard Ax /
   tiled tooling.

## Minimal sketch

```python
from bluesky.run_engine import RunEngine
from blop.ax import Agent, RangeDOF, Objective
from blop.plans import optimize

dofs = [RangeDOF(name="x", lower=-1.0, upper=1.0)]
objectives = [Objective(name="y", minimize=False)]
agent = Agent(
    dofs=dofs,
    objectives=objectives,
    actuators=[x_motor],
    detectors=[detector],
    dofs_to_actuators={"x": x_motor},
)
RunEngine()(optimize(agent, num_iterations=20))
```

## Common use-cases

- **Beamline alignment** — mirror angles, curvatures, slit positions.
- **Multi-objective optimization** — e.g. maximize flux while minimizing
  spot size (use `ScalarizedObjective` or rely on the Pareto frontier).
- **Parameter tuning** when measurements are expensive or slow.
- **Automated calibration** of device parameter sets.

## When in doubt

- Prefer the `ax/` API over `bayesian/` for new work.
- Detailed tutorials live under `docs/source/tutorials/` (notebooks via
  jupytext). Build them with `pixi run -e docs build-docs`.
- General workflow / lint / test commands live in the repo-root `AGENTS.md`.
