"""TES beamline simulation backend, driven by a data-trained emulator."""

import numpy as np

from .core import SimBackend
from .models.tes_model import TESModel


class TESBackend(SimBackend):
    """Simulation backend backed by the NSLS-II TES beamline emulator.

    Unlike :class:`SimpleBackend` (analytic Gaussian) and :class:`XRTBackend`
    (ray-tracing), this backend replays a model trained on 28,561 *measured*
    detector frames, so the beam response to the four KB jack motors — including
    astigmatism, non-Gaussian streaks, and flux variation — is that of the real
    beamline at 3 keV.

    It reads the same KB mirror devices as ``SimpleBackend``
    (:class:`blop_sim.devices.simple.KBMirror`): the *horizontal* mirror's
    (downstream, upstream) jacks map to motor columns (0, 1) and the *vertical*
    mirror's to columns (2, 3). Valid jack positions live inside the recorded
    motor box (see :meth:`motor_box`); outside it the model extrapolates and
    should not be trusted.

    Args:
        weights_path: Path to the trained weights (.npz). Defaults to the
            ``BLOP_SIM_TES_WEIGHTS`` environment variable (see
            ``models.tes_model.resolve_weights_path``).
        noise: If True, add real background-speck patterns to each image.
        seed: Seed for the noise sampling.
    """

    def __init__(self, weights_path: str | None = None, noise: bool = False, seed: int | None = None):
        super().__init__()
        self._model = TESModel(weights_path)
        self._image_shape = self._model.frame_shape
        self._noise = noise
        self._rng = np.random.default_rng(seed)

    def motor_box(self) -> dict[str, tuple[float, float]]:
        """Per-motor (lo, hi) validity bounds, keyed by column role.

        Use these as ``RangeDOF`` bounds so the optimizer never leaves the box
        the model was trained on.
        """
        lo, hi = self._model.box_lo, self._model.box_hi
        names = ("kbh_downstream", "kbh_upstream", "kbv_downstream", "kbv_upstream")
        return {name: (float(lo[i]), float(hi[i])) for i, name in enumerate(names)}

    async def generate_beam(self) -> np.ndarray:
        """Generate a detector-count beam image at the current jack positions.

        Returns:
            2D numpy array with shape ``self._image_shape`` (counts: the model's
            L1-normalized beam scaled by its predicted flux, so intensity-based
            outcome constraints behave like on a real detector).
        """
        m = await self._get_jack_positions()
        out = self._model.predict(m[None], sample=self._noise, rng=self._rng)
        return (out["flux"][0] * out["image"][0]).astype(np.float64)

    async def _get_jack_positions(self) -> np.ndarray:
        """Assemble the 4-motor vector from registered KB mirror devices.

        Motors default to the center of the recorded box, so a beamline with no
        mirrors registered (or only one) still produces a valid beam.
        """
        m = (self._model.box_lo + self._model.box_hi) / 2
        for name, device in self._device_states.items():
            if device["type"] == "kb_mirror_simple":
                state = await self._get_device_state(name)
                if state["orientation"] == "horizontal":
                    m[0], m[1] = state["downstream"], state["upstream"]
                elif state["orientation"] == "vertical":
                    m[2], m[3] = state["downstream"], state["upstream"]
        return m


__all__ = ["TESBackend"]
