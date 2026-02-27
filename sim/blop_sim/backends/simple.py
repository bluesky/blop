"""Simple mathematical beam simulation backend."""

import numpy as np
import scipy as sp  # type: ignore[import-untyped]

from . import SimBackend


class SimpleBackend(SimBackend):
    """Mathematical Gaussian beam simulation with 4th power falloff."""
    
    async def generate_beam(self, noise: bool = True) -> np.ndarray:
        """Generate beam using mathematical Gaussian model.
        
        The beam is affected by:
        - KB mirror jack positions (controls focus position and width)
        - Slit aperture (clips the beam)
        - Optional noise (white + pink noise)
        
        Args:
            noise: Whether to add noise to the image
            
        Returns:
            2D numpy array with shape (nx, ny)
        """
        nx, ny = self._image_shape
        
        # Get device states
        kb_states = await self._get_kb_states()
        slit_state = await self._get_slit_state()
        
        # Create meshgrid
        x = np.linspace(-10, 10, ny)
        y = np.linspace(-10, 10, nx)
        X, Y = np.meshgrid(x, y)
        
        # Calculate beam center from KB mirror positions
        x0 = kb_states["kbh"]["ush"] - kb_states["kbh"]["dsh"]
        y0 = kb_states["kbv"]["usv"] - kb_states["kbv"]["dsv"]
        
        # Calculate beam widths from KB mirror positions
        x_width = np.sqrt(0.2 + 5e-1 * (kb_states["kbh"]["ush"] + kb_states["kbh"]["dsh"] - 1) ** 2)
        y_width = np.sqrt(0.1 + 5e-1 * (kb_states["kbv"]["usv"] + kb_states["kbv"]["dsv"] - 2) ** 2)
        
        # Generate Gaussian beam with 4th power falloff
        beam = np.exp(-0.5 * (((X - x0) / x_width) ** 4 + ((Y - y0) / y_width) ** 4)) / (
            np.sqrt(2 * np.pi) * x_width * y_width
        )
        
        # Apply slit mask
        mask = X > slit_state["inboard"]
        mask &= X < slit_state["outboard"]
        mask &= Y > slit_state["lower"]
        mask &= Y < slit_state["upper"]
        mask = sp.ndimage.gaussian_filter(mask.astype(float), sigma=1)
        
        image = beam * mask
        
        # Add noise if requested
        if noise:
            kx = np.fft.fftfreq(n=len(x), d=0.1)
            ky = np.fft.fftfreq(n=len(y), d=0.1)
            KX, KY = np.meshgrid(kx, ky)
            
            power_spectrum = 1 / (1e-2 + KX**2 + KY**2)
            
            white_noise = 1e-3 * np.random.standard_normal(size=X.shape)
            pink_noise = 1e-3 * np.real(
                np.fft.ifft2(power_spectrum * np.fft.fft2(np.random.standard_normal(size=X.shape)))
            )
            
            image += white_noise + pink_noise
        
        return image
    
    async def _get_kb_states(self) -> dict:
        """Get KB mirror states from registered devices."""
        kbh_state = {"ush": 0.0, "dsh": 0.0}
        kbv_state = {"usv": 0.0, "dsv": 0.0}
        
        for name, device in self._device_states.items():
            if device["type"] == "kb_mirror_simple":
                state = await self._get_device_state(name)
                if state["orientation"] == "horizontal":
                    kbh_state["ush"] = state["upstream"]
                    kbh_state["dsh"] = state["downstream"]
                elif state["orientation"] == "vertical":
                    kbv_state["usv"] = state["upstream"]
                    kbv_state["dsv"] = state["downstream"]
        
        return {"kbh": kbh_state, "kbv": kbv_state}
    
    async def _get_slit_state(self) -> dict:
        """Get slit state from registered devices."""
        slit_state = {"inboard": -5.0, "outboard": 5.0, "lower": -5.0, "upper": 5.0}
        
        for name, device in self._device_states.items():
            if device["type"] == "slit":
                slit_state = await self._get_device_state(name)
                break
        
        return slit_state


__all__ = ["SimpleBackend"]
