"""

__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "2024-09-27"

Created with xrtQook




"""

import numpy as np
import xrt.backends.raycing as raycing
import xrt.backends.raycing.oes as roes
import xrt.backends.raycing.run as rrun
import xrt.backends.raycing.screens as rscreens
import xrt.backends.raycing.sources as rsources
import xrt.plotter as xrtplot
import xrt.runner as xrtrun
from ..core import SimBackend


def build_beamline():
    beamLine = raycing.BeamLine()

    beamLine.geometricSource01 = rsources.GeometricSource(
        bl=beamLine, center=[0, 0, 0], nrays=50000, energies=(9000, 100), distE="normal", dx=0.2, dz=0.1, dxprime=0.00015
    )

    beamLine.toroidMirror01 = roes.ToroidMirror(
        bl=beamLine,
        center=[0, 10000, 0],
        pitch=r"5deg",
        limPhysX=[-20.0, 20.0],
        limPhysY=[-150.0, 150.0],
        R=38245,
        r=100000000.0,
    )

    beamLine.toroidMirror02 = roes.ToroidMirror(
        bl=beamLine,
        center=[0, 11000, 176.33353257489432],
        pitch=r"5deg",
        yaw=r"10deg",
        positionRoll=r"90deg",
        rotationSequence=r"RyRxRz",
        limPhysX=[-20, 20],
        limPhysY=[-150, 150],
        R=21035,
        r=100000000.0,
    )

    beamLine.screen01 = rscreens.Screen(bl=beamLine, center=[164.87347936545572, 11935, 343.73164815693235])

    return beamLine


def run_process(beamLine):
    geometricSource01beamGlobal01 = beamLine.geometricSource01.shine()

    toroidMirror01beamGlobal01, toroidMirror01beamLocal01 = beamLine.toroidMirror01.reflect(
        beam=geometricSource01beamGlobal01
    )

    toroidMirror02beamGlobal01, toroidMirror02beamLocal01 = beamLine.toroidMirror02.reflect(beam=toroidMirror01beamGlobal01)

    screen01beamLocal01 = beamLine.screen01.expose(beam=toroidMirror02beamGlobal01)

    outDict = {
        "geometricSource01beamGlobal01": geometricSource01beamGlobal01,
        "toroidMirror01beamGlobal01": toroidMirror01beamGlobal01,
        "toroidMirror01beamLocal01": toroidMirror01beamLocal01,
        "toroidMirror02beamGlobal01": toroidMirror02beamGlobal01,
        "toroidMirror02beamLocal01": toroidMirror02beamLocal01,
        "screen01beamLocal01": screen01beamLocal01,
    }
    beamLine.prepare_flow()
    return outDict


rrun.run_process = run_process


def define_plots():
    plots = []

    plot01 = xrtplot.XYCPlot(
        beam=r"screen01beamLocal01",
        xaxis=xrtplot.XYCAxis(label=r"x", limits=limits[0], bins=400, ppb=1, fwhmFormatStr="%.3f"),
        yaxis=xrtplot.XYCAxis(label=r"z", limits=limits[1], bins=300, ppb=1, fwhmFormatStr="%.3f"),
        caxis=xrtplot.XYCAxis(label=r"energy", unit=r"eV", bins=300, ppb=1),
        title=r"plot01",
        aspect="auto",
    )
    plots.append(plot01)
    return plots


def main():
    beamLine = build_beamline()
    beamLine.glow()
    E0 = list(beamLine.geometricSource01.energies)[0]
    beamLine.alignE = E0
    plots = define_plots()
    xrtrun.run_ray_tracing(plots=plots, backend=r"raycing", beamLine=beamLine)

class KBBackend(SimBackend):
    """XRT ray-tracing simulation backend.

    Uses the XRT package to perform realistic ray-tracing through a KB mirror pair.
    Much slower than SimpleBackend but more physically accurate.
    """

    def __init__(self, noise: bool = False):
        """Initialize XRT backend."""
        super().__init__()
        self._beamline = None
        self._limits = [[-0.6, 0.6], [-0.45, 0.45]]
        self._noise = noise

    def _ensure_beamline(self):
        """Build XRT beamline if not already built."""
        if self._beamline is None:
            self._beamline = build_beamline()

    async def generate_beam(self) -> np.ndarray:
        """Generate beam using XRT ray-tracing.

        Returns:
            2D numpy array with shape (300, 400)
        """
        self._ensure_beamline()

        # Get KB mirror radii from devices
        mirror_radii = await self._get_mirror_radii()

        # Update XRT beamline mirror parameters
        self._beamline.toroidMirror01.R = mirror_radii[0]  # Vertical mirror
        self._beamline.toroidMirror02.R = mirror_radii[1]  # Horizontal mirror

        # Run ray tracing
        outDict = run_process(self._beamline)
        lb = outDict["screen01beamLocal01"]

        # Build histogram from ray data
        hist2d, _, _ = build_histRGB(lb, lb, limits=self._limits, isScreen=True, shape=[400, 300])
        image = hist2d

        # Add noise if requested
        if self._noise:
            image += 1e-3 * np.abs(np.random.standard_normal(size=image.shape))

        return image

    async def _get_mirror_radii(self) -> list[float]:
        """Get KB mirror radii from registered devices.

        Returns:
            [R1, R2] where R1 is first mirror (vertical), R2 is second mirror (horizontal)
        """
        # Default radii from xrt_kb_model.py
        radii = [38245.0, 21035.0]

        for name, device in self._device_states.items():
            if device["type"] == "kb_mirror_xrt":
                state = await self._get_device_state(name)
                mirror_index = state["mirror_index"]
                radius = state["radius"]
                if mirror_index < len(radii):
                    radii[mirror_index] = radius

        return radii


if __name__ == "__main__":
    main()
