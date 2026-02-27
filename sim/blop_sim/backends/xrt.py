"""XRT ray-tracing beam simulation backend."""

import numpy as np

from ..xrt_kb_pair.xrt_kb_model import build_beamline, build_histRGB, run_process
from . import SimBackend


class XRTBackend(SimBackend):
    """XRT ray-tracing beam simulation."""
    
    def __init__(self):
        """Initialize XRT backend with beamline."""
        super().__init__()
        
        if not self._initialized:
            return
        
        # Build XRT beamline (lazy initialization)
        self._beamline = None
        self._limits = [[-0.6, 0.6], [-0.45, 0.45]]
        # Override image shape for XRT (matches histogram dimensions)
        self._image_shape = (300, 400)
    
    def _ensure_beamline(self):
        """Lazy initialization of XRT beamline."""
        if self._beamline is None:
            self._beamline = build_beamline()
    
    def generate_beam(self, noise: bool = True) -> np.ndarray:
        """Generate beam using XRT ray tracing.
        
        The beam is affected by:
        - KB mirror curvature radii (R parameters)
        - XRT optical system (toroidal mirrors, screen)
        - Optional noise
        
        Args:
            noise: Whether to add noise to the image
            
        Returns:
            2D numpy array with shape (300, 400)
        """
        self._ensure_beamline()
        
        # Get KB mirror radii from devices
        mirror_radii = self._get_mirror_radii()
        
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
        if noise:
            image += 1e-3 * np.abs(np.random.standard_normal(size=image.shape))
        
        return image
    
    def _get_mirror_radii(self) -> list[float]:
        """Get KB mirror radii from registered devices.
        
        Returns:
            [R1, R2] where R1 is first mirror (vertical), R2 is second mirror (horizontal)
        """
        # Default radii from xrt_kb_model.py
        radii = [38245.0, 21035.0]
        
        for name, device in self._device_states.items():
            if device["type"] == "kb_mirror_xrt":
                state = device["get_state"]()
                mirror_index = state["mirror_index"]
                radius = state["radius"]
                if mirror_index < len(radii):
                    radii[mirror_index] = radius
        
        return radii


__all__ = ["XRTBackend"]
