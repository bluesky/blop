---
name: blop-overview
description: Overview of the Blop project structure, workflow, and use-cases
compatibility: opencode
---

## What is Blop?

Blop (Beamline Optimization Package) is a Python library for Bayesian optimization of experimental systems, particularly beamline experiments at synchrotron facilities. It bridges optimization algorithms (using Ax platform, BoTorch, GPyTorch) with the Bluesky ecosystem for data acquisition and device control, enabling efficient exploration of expensive-to-evaluate parameter spaces.

## Project Structure

- **`src/blop/ax/`** - Core Agent implementation and Ax platform integration (DOFs, objectives, constraints)
- **`src/blop/bayesian/`** - Bayesian optimization components and models
- **`src/blop/plans.py`** - Bluesky plans for running experiments and optimization iterations
- **`docs/`** - Comprehensive Sphinx documentation with tutorials, how-tos, and explanations

## Basic Workflow

1. **Define DOFs** - Specify degrees of freedom (parameters to optimize) using `RangeDOF` or `ChoiceDOF`
1. **Define Objectives** - Specify what to maximize or minimize with `Objective` or `ScalarizedObjective`
1. **Add Constraints** (optional) - Set boundaries with `DOFConstraint` or `OutcomeConstraint`
1. **Create Agent** - Instantiate the optimization agent with DOFs, objectives, and an evaluation function
1. **Run Optimization** - Execute iterations using Bluesky's RunEngine to collect data and update models
1. **Analyze Results** - Review health metrics, convergence, and Pareto frontiers for multi-objective problems

## Common Use-Cases

- **Beamline alignment** - Optimizing mirror positions, angles, and curvatures for X-ray focusing
- **Multi-objective optimization** - Balancing competing goals like maximizing intensity while minimizing beam spot size
- **Parameter tuning** - Finding optimal experimental settings when measurements are expensive or time-consuming
- **Automated calibration** - Systematic exploration of device parameters for optimal performance

## Documentation

The project has excellent documentation at `/docs/` with detailed tutorials and explanations. **Refer to the docs for specific implementation details, API references, and worked examples.**
