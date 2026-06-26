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

# Demonstrating a "bring your own beamline" methodology with XRT

In this tutorial we'll show a simulation optimization workflow by loading arbitrary XRT setups using xml/json.
By the end, you should be able to go to XRT qook/glow and build your beamline from there to export and drop in
to blop. Or if you are lucky enough for an XRT model to be already built for you, export to xml and load in
blop.

## Some Environment Setup

note, like all other demos you need the blop_sim subpackage to run

```{code-cell} ipython3
import logging
import warnings
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
from bluesky.callbacks.best_effort import BestEffortCallback

# Import simulation devices (requires: pip install -e sim/)
from bluesky.run_engine import RunEngine
from bluesky.utils import ProgressBarManager
from bluesky_tiled_plugins import TiledWriter
from tiled.client import from_uri  # type: ignore[import-untyped]
from tiled.client.container import Container
from tiled.server import SimpleTiledServer

from blop import RangeDOF
from blop.ax import Agent, Objective
from blop.protocols import EvaluationFunction
from blop_sim.backends import XRTBackend
from blop_sim.devices.xrt import infer_detectors, infer_variables

# Suppress noisy logs from httpx and dependency deprecation warnings
logging.getLogger("httpx").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)

# Enable interactive plotting
plt.ion()

DETECTOR_STORAGE = "/tmp/blop/sim"
```

```{code-cell} ipython3
from pathlib import Path
print([p.name for p in Path.cwd().iterdir()])
fileName = r"toroid_focus.xml"
beam = XRTBackend(file=fileName, n_iters=4, n_workers=4)
dets = infer_detectors(beam)
motors = infer_variables(beam, filter_for=None)
```

### A small view of the inferred motors

```{code-cell} ipython3
pprint(motors)
```

### Another glimpse into the inferred detectors

```{code-cell} ipython3
pprint(dets)
```

# Setting up an optimization

```{code-cell} ipython3
toro_R = motors["toroidMirror01"]["R"]
toro_R.alias = "big_r"
toro_r = motors["toroidMirror01"]["r"]

screen = dets["screen01"]
screen.set_primary()


VERTICAL_BOUNDS = (toro_R.val - 10000, toro_R.val + 10000)
HORIZONTAL_BOUNDS = (toro_r.val - 500, toro_r.val + 500)
# Define DOFs using mirror radius signals
dofs = [
    RangeDOF(actuator=toro_R, bounds=VERTICAL_BOUNDS, parameter_type="float"),
    RangeDOF(actuator=toro_r, bounds=HORIZONTAL_BOUNDS, parameter_type="float"),
]
```

```{code-cell} ipython3
tiled_server = SimpleTiledServer()
tiled_client = from_uri(tiled_server.uri)
tiled_writer = TiledWriter(tiled_client)

RE = RunEngine({})
bec = BestEffortCallback()

# Send all metadata/data captured to the BestEffortCallback.
# RE.subscribe(bec)
RE.waiting_hook = ProgressBarManager()

tiled_client = from_uri(tiled_server.uri)
tiled_writer = TiledWriter(tiled_client)
RE.subscribe(tiled_writer)
```

```{code-cell} ipython3
# Single objective: minimize the geometric-mean FWHM
objectives = [
    Objective(name="fwhm", minimize=True),
]
```

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
            return 400.0, 0.0  # No beam — return max FWHM penalty

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
        images = run[f"primary/{screen.name}"].read()

        # Suggestion IDs stored in start document metadata
        suggestion_ids = [suggestion["_id"] for suggestion in run.metadata["start"]["blop_suggestions"]]

        # Compute statistics from each image
        for idx, sid in enumerate(suggestion_ids):
            image = images[idx]
            fwhm, intensity = self._compute_stats(image)

            outcome = {
                "_id": sid,
                "fwhm": fwhm,
                # "intensity": intensity,
            }
            outcomes.append(outcome)
        return outcomes
```

```{code-cell} ipython3
agent = Agent(
    sensors=[screen],
    dofs=dofs,
    objectives=objectives,
    evaluation_function=DetectorEvaluation(tiled_client),
    name="xrt-blop-demo",
    description="A demo of the Blop agent with XRT simulated beamline",
    experiment_type="demo",
)
```

```{code-cell} ipython3
# Run 1 iteration with a batch of 10 points for initial exploration
RE(agent.optimize(1, n_points=5))
```

```{code-cell} ipython3
# Run remaining 10 iterations
RE(agent.optimize(10))
```

```{code-cell} ipython3
_ = agent.ax_client.compute_analyses()
```

```{code-cell} ipython3
agent.ax_client.summarize()
```

```{code-cell} ipython3
optimal_parameters, metrics, _, _ = agent.ax_client.get_best_parameterization(use_model_predictions=False)
optimal_parameters
```

```{code-cell} ipython3
from bluesky.plans import list_scan

uid = RE(
    list_scan(
        [screen],
        toro_r,
        [optimal_parameters[toro_r.name]],
        toro_R,
        [optimal_parameters[toro_R.name]],
    )
)
```

```{code-cell} ipython3
image = tiled_client[uid[0]][f"primary/{screen.name}"].read().squeeze()
plt.imshow(image)
plt.colorbar()
plt.title("Optimized toroid Mirror Beam")
plt.show()
```
