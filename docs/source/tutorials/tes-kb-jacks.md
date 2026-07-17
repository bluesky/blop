---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: dev
  language: python
  name: python3
---

# Aligning a Measured KB Mirror System

In this tutorial, you will align a Kirkpatrick-Baez (KB) mirror system whose response comes from **measured beamline data** rather than an analytic or ray-traced model. The workflow is the same as in the [XRT KB mirror tutorial](./xrt-kb-mirrors.md) — DOFs, objectives, an evaluation function, and an `Agent` — with two differences:

- The beam is produced by `TESBackend`, an emulator trained on 28,561 detector frames recorded on a complete 13⁴ grid of the four KB jack motors of the NSLS-II TES beamline (3 keV). Astigmatism, non-Gaussian streaks, and flux variation are those of the real machine.
- The degrees of freedom are the **four jack positions** (upstream/downstream per mirror) instead of two mirror radii, so the optimizer works in the same parameter space operators use.

```{note}
The trained weights (16 MB, derived from measured data) are not part of this repository. Download `emulator_weights.npz` from the [tes-emulator](https://github.com/FLlorente/tes-emulator) package and point the `BLOP_SIM_TES_WEIGHTS` environment variable at it before running this tutorial.
```

## Setting Up the Environment

As in the other tutorials, Blop uses [Bluesky](https://blueskyproject.io/) to run experiments and [Tiled](https://blueskyproject.io/tiled/) to store and retrieve data.

```{code-cell} ipython3
import logging
import warnings
from pathlib import PurePath

import bluesky.plan_stubs as bps
import bluesky.plans as bp
import matplotlib.pyplot as plt
import numpy as np
from ax.api.protocols import IMetric
from bluesky.run_engine import RunEngine
from bluesky_tiled_plugins import TiledWriter
from ophyd_async.core import StaticPathProvider, UUIDFilenameProvider
from tiled.client import from_uri  # type: ignore[import-untyped]
from tiled.client.container import Container
from tiled.server import SimpleTiledServer

from blop.ax import Agent, Objective, RangeDOF
from blop.ax.objective import OutcomeConstraint
from blop.protocols import EvaluationFunction

# Import simulation devices (requires: pip install -e sim/)
from blop_sim.backends.tes import TESBackend
from blop_sim.devices import DetectorDevice
from blop_sim.devices.simple import KBMirror

# Suppress noisy logs from httpx and dependency deprecation warnings
logging.getLogger("httpx").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)

# Enable interactive plotting
plt.ion()

DETECTOR_STORAGE = "/tmp/blop/sim"
```

```{code-cell} ipython3
tiled_server = SimpleTiledServer(readable_storage=[DETECTOR_STORAGE])
tiled_client = from_uri(tiled_server.uri)
tiled_writer = TiledWriter(tiled_client)

RE = RunEngine({})
RE.subscribe(tiled_writer)
```

## Defining Degrees of Freedom

The emulator is valid inside the motor box it was trained on, and `TESBackend.motor_box()` exposes those bounds directly — using them as DOF bounds means the optimizer can never ask for a configuration the model has not seen:

```{code-cell} ipython3
# Create the TES emulator backend (reads BLOP_SIM_TES_WEIGHTS)
backend = TESBackend()
box = backend.motor_box()
box
```

The backend reads the same jack-style KB mirror devices as `SimpleBackend`: one horizontal and one vertical mirror, each with an upstream and a downstream jack. We start every jack at the center of its range and let the optimizer explore from there:

```{code-cell} ipython3
kbh = KBMirror(backend, orientation="horizontal", name="kbh")
kbv = KBMirror(backend, orientation="vertical", name="kbv")

# Create detector device
det = DetectorDevice(backend, StaticPathProvider(UUIDFilenameProvider(), PurePath(DETECTOR_STORAGE)), name="det")

# Move the jacks to the center of the recorded motor box
RE(
    bps.mv(
        kbh.downstream, np.mean(box["kbh_downstream"]),
        kbh.upstream, np.mean(box["kbh_upstream"]),
        kbv.downstream, np.mean(box["kbv_downstream"]),
        kbv.upstream, np.mean(box["kbv_upstream"]),
    )
)

# Four DOFs: one per jack, bounded by the recorded motor box
dofs = [
    RangeDOF(actuator=kbh.downstream, bounds=box["kbh_downstream"], parameter_type="float"),
    RangeDOF(actuator=kbh.upstream, bounds=box["kbh_upstream"], parameter_type="float"),
    RangeDOF(actuator=kbv.downstream, bounds=box["kbv_downstream"], parameter_type="float"),
    RangeDOF(actuator=kbv.upstream, bounds=box["kbv_upstream"], parameter_type="float"),
]
```

## Defining the Objective and Constraints

Exactly as in the XRT tutorial: minimize the beam FWHM, and use an intensity `OutcomeConstraint` as a safety net against configurations where the beam is lost:

```{code-cell} ipython3
objectives = [
    Objective(name="fwhm", minimize=True),
]

# Track intensity without optimizing it
intensity_metric = IMetric(name="intensity")

# Soft constraint: reject configurations with too little flux on the detector
outcome_constraints = [
    OutcomeConstraint(constraint="i >= 10000", i=intensity_metric),
]
```

## Writing an Evaluation Function

The evaluation function is identical to the XRT tutorial's — it reads beam images back from Tiled and computes the FWHM of the marginal profiles, so it does not care which backend produced the image:

```{code-cell} ipython3
class DetectorEvaluation(EvaluationFunction):
    def __init__(self, tiled_client: Container):
        self.tiled_client = tiled_client

    def _fwhm_from_profile(self, profile: np.ndarray) -> float:
        """Compute FWHM from a 1D marginal profile.

        Finds the half-maximum crossing points with sub-pixel interpolation.
        Returns a large value if the beam is too dim or fills the entire detector.
        """
        peak = profile.max()
        if peak == 0:
            return float(len(profile))  # No signal — return detector width as penalty

        half_max = peak / 2.0
        above = profile >= half_max
        if not above.any():
            return float(len(profile))

        indices = np.where(above)[0]
        left_idx = indices[0]
        right_idx = indices[-1]

        # Sub-pixel interpolation at left crossing
        if left_idx > 0:
            left = left_idx - 1 + (half_max - profile[left_idx - 1]) / (profile[left_idx] - profile[left_idx - 1])
        else:
            left = 0.0

        # Sub-pixel interpolation at right crossing
        if right_idx < len(profile) - 1:
            right = right_idx + (half_max - profile[right_idx]) / (profile[right_idx + 1] - profile[right_idx])
        else:
            right = float(len(profile) - 1)

        return right - left

    def _compute_stats(self, image: np.ndarray) -> tuple[float, float]:
        """Compute FWHM and integrated intensity from a beam image.

        Returns
        -------
        fwhm : float
            Geometric mean of the horizontal and vertical FWHM (in pixels).
        intensity : float
            Total integrated intensity (sum of all pixel values).
        """
        gray = image.squeeze().astype(np.float64)
        if gray.ndim == 3:
            gray = gray.mean(axis=-1)

        # Integrated intensity (total flux on detector)
        intensity = gray.sum()

        if intensity == 0:
            return float(max(gray.shape)), 0.0  # No beam — return max FWHM penalty

        # Marginal profiles: project onto each axis
        x_profile = gray.sum(axis=0)  # sum along Y rows -> X profile
        y_profile = gray.sum(axis=1)  # sum along X cols -> Y profile

        fwhm_x = self._fwhm_from_profile(x_profile)
        fwhm_y = self._fwhm_from_profile(y_profile)

        # Geometric mean FWHM — targets a small, round spot
        fwhm = np.sqrt(fwhm_x * fwhm_y)

        return float(fwhm), float(intensity)

    def __call__(self, uid: str, suggestions: list[dict]) -> list[dict]:
        outcomes = []
        run = self.tiled_client[uid]

        # Read beam images from detector
        images = run["primary/det_image"].read()

        # Suggestion IDs stored in start document metadata
        suggestion_ids = [suggestion["_id"] for suggestion in run.metadata["start"]["blop_suggestions"]]

        # Compute statistics from each image
        for idx, sid in enumerate(suggestion_ids):
            image = images[idx]
            fwhm, intensity = self._compute_stats(image)

            outcome = {
                "_id": sid,
                "fwhm": fwhm,
                "intensity": intensity,
            }
            outcomes.append(outcome)
        return outcomes
```

## Creating and Running the Agent

```{code-cell} ipython3
agent = Agent(
    sensors=[det],
    dofs=dofs,
    objectives=objectives,
    evaluation_function=DetectorEvaluation(tiled_client),
    outcome_constraints=outcome_constraints,
    name="tes-blop-demo",
    description="Blop agent aligning the measured TES KB system",
    experiment_type="demo",
)

# Register intensity as a tracking metric (monitored but not optimized)
agent.ax_client.configure_metrics([intensity_metric])
```

With four DOFs the parameter space is larger than in the two-radius tutorial, so we use a somewhat larger initial exploration batch before switching to model-driven suggestions:

```{code-cell} ipython3
# Initial exploration: one batch of 16 quasi-random points
RE(agent.optimize(1, n_points=16))
```

```{code-cell} ipython3
# Model-driven optimization
RE(agent.optimize(16))
```

## Understanding the Results

```{code-cell} ipython3
_ = agent.ax_client.compute_analyses()
```

```{code-cell} ipython3
agent.ax_client.summarize()
```

The physics of a KB mirror ties focus to the **pitch** of each mirror — the difference between its two jacks — so a useful way to look at the surrogate is one jack against its partner:

```{code-cell} ipython3
_ = agent.plot_objective(x_dof_name="kbh-downstream", y_dof_name="kbh-upstream", objective_name="fwhm")
```

## Applying the Optimal Configuration

```{code-cell} ipython3
optimal_parameters, metrics, _, _ = agent.ax_client.get_best_parameterization(use_model_predictions=False)
optimal_parameters
```

```{code-cell} ipython3
# Move the jacks to the optimum, record one frame, and read it back from Tiled
moves = []
for dof in dofs:
    moves += [dof.actuator, optimal_parameters[dof.actuator.name.replace("_", "-")]]
RE(bps.mv(*moves))
(uid,) = RE(bp.count([det]))

image = tiled_client[uid]["primary/det_image"].read().squeeze()

plt.figure(figsize=(6, 4))
plt.imshow(np.log1p(image), origin="lower")
plt.title("Beam at the optimized jack positions (log scale)")
plt.colorbar(label="log(1 + counts)")
plt.show()
```

Because the emulator was trained on real frames, the optimum found here corresponds to the actual focal pitches of the TES KB system — the same configuration a human operator would align to.
