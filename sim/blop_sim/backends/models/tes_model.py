"""Data-driven TES beamline model: motors -> beam image + flux.

A trimmed inference-only port of the ``tes-emulator`` package: a differentiable
motor->beam emulator of the NSLS-II TES beamline KB system, trained on 28,561 real
detector frames recorded on a complete 13^4 grid of the four KB jack motors (3 keV).

Motor vector columns (KB jack positions, in motor units):
    0 = KBH downstream jack, 1 = KBH upstream, 2 = KBV downstream, 3 = KBV upstream

The trained weights are NOT part of this repository (16 MB, derived from measured
beamline data). ``resolve_weights_path`` finds them from, in order: an explicit path,
the ``BLOP_SIM_TES_WEIGHTS`` environment variable, or a cached download from
``TES_WEIGHTS_URL`` when a release URL is configured.
"""

import os
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

# Release URL for the trained weights (filled in once the weights are published;
# can be overridden with the TES_WEIGHTS_URL environment variable).
WEIGHTS_URL: str | None = None

_CACHE_DIR = Path("~/.cache/blop-sim").expanduser()


def resolve_weights_path(weights_path: str | os.PathLike[str] | None = None) -> Path:
    """Locate the trained TES emulator weights (.npz).

    Resolution order: explicit ``weights_path`` argument, the ``BLOP_SIM_TES_WEIGHTS``
    environment variable, then a cached download from ``TES_WEIGHTS_URL`` /
    ``WEIGHTS_URL``. Raises with instructions if none is available.
    """
    if weights_path is not None:
        path = Path(weights_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"TES emulator weights not found at {path}")
        return path

    env_path = os.environ.get("BLOP_SIM_TES_WEIGHTS")
    if env_path:
        path = Path(env_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"BLOP_SIM_TES_WEIGHTS points to a missing file: {path}")
        return path

    url = os.environ.get("TES_WEIGHTS_URL", WEIGHTS_URL)
    if url:
        cached = _CACHE_DIR / "tes_emulator_weights.npz"
        if not cached.exists():
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            tmp = cached.with_suffix(".part")
            urllib.request.urlretrieve(url, tmp)  # noqa: S310
            tmp.rename(cached)
        return cached

    raise RuntimeError(
        "TES emulator weights not found. Either pass weights_path=..., set the "
        "BLOP_SIM_TES_WEIGHTS environment variable to the .npz file, or set "
        "TES_WEIGHTS_URL to a download location. The weights ship with the "
        "tes-emulator package (https://github.com/FLlorente/tes-emulator)."
    )


class TESModel:
    """Inference wrapper around the trained TES emulator weights.

    ``predict(m)`` maps (n, 4) KB jack positions to L1-normalized beam images,
    detector-count flux, and an in-domain flag. Heads: ``mlp`` (smooth, default),
    ``interp`` (exact at the recorded grid points), ``poly`` (physics ridge).
    """

    def __init__(self, weights_path: str | os.PathLike[str] | None = None, head: str | None = None):
        import torch  # deferred: keep blop_sim importable without torch

        self._torch = torch
        w: dict[str, Any] = dict(np.load(resolve_weights_path(weights_path), allow_pickle=False))
        self.K = int(w["pca_K"])
        self.patch = int(w["patch"])
        self.win = int(w["win"])
        self.frame_shape = (int(w["frame_h"]), int(w["frame_w"]))
        self.head = head or str(w.get("default_head", "mlp"))
        t = lambda k: torch.tensor(w[k], dtype=torch.float64)  # noqa: E731
        self._w = {k: t(k) for k in w if k.startswith(("mlp_", "poly_", "cenres_", "interp_", "cen_coef", "pca_"))}
        self.box_lo = np.asarray(w["box_lo"], dtype=float)
        self.box_hi = np.asarray(w["box_hi"], dtype=float)
        self._noise = None
        if "noise_ptr" in w:
            self._noise = {k: w[f"noise_{k}"] for k in ("rows", "cols", "vals", "ptr")}

    # -- heads ---------------------------------------------------------------

    @staticmethod
    def _kb_basis(m):
        """Degree-2 KB physics basis with intercept: (n, 4) -> (n, 15)."""
        import torch

        p_h, h_h = m[:, 0] - m[:, 1], m[:, 0] + m[:, 1]
        p_v, h_v = m[:, 2] - m[:, 3], m[:, 2] + m[:, 3]
        f = [p_h, h_h, p_v, h_v]
        feats = list(f)
        for i in range(4):
            for j in range(i, 4):
                feats.append(f[i] * f[j])
        return torch.stack([torch.ones_like(p_h), *feats], dim=1)

    def _interp(self, m):
        """4-D multilinear interpolation on the full-grid table (clamped to the box)."""
        torch = self._torch
        table, axes = self._w["interp_table"], self._w["interp_axes"]
        n = m.shape[0]
        idxs, fracs = [], []
        for c in range(4):
            ax = axes[c]
            mc = torch.clamp(m[:, c], ax[0], ax[-1])
            i = torch.clamp(torch.searchsorted(ax.contiguous(), mc.contiguous(), right=True) - 1, 0, len(ax) - 2)
            fracs.append((mc - ax[i]) / (ax[i + 1] - ax[i]))
            idxs.append(i)
        out = torch.zeros(n, table.shape[-1], dtype=m.dtype)
        for corner in range(16):
            wgt = torch.ones(n, dtype=m.dtype)
            ix = []
            for c in range(4):
                d = (corner >> c) & 1
                wgt = wgt * (fracs[c] if d else 1.0 - fracs[c])
                ix.append(idxs[c] + d)
            out = out + wgt[:, None] * table[ix[0], ix[1], ix[2], ix[3]]
        return out

    def _anchor(self, m):
        """Centroid anchor (cx, cy): interp table, or physics ridge + MLP residual."""
        torch, w = self._torch, self._w
        if self.head == "interp":
            return self._interp(m)[:, :2]
        a = self._kb_basis(m) @ w["cen_coef"]
        if "cenres_W0" in w:
            z = (m - w["cenres_xmean"]) / w["cenres_xstd"]
            z = torch.nn.functional.silu(z @ w["cenres_W0"].T + w["cenres_b0"])
            z = torch.nn.functional.silu(z @ w["cenres_W1"].T + w["cenres_b1"])
            z = z @ w["cenres_W2"].T + w["cenres_b2"]
            a = a + z * w["cenres_ystd"] + w["cenres_ymean"]
        return a

    def _codes_flux(self, m):
        """Head evaluation -> (PCA codes (n, K), flux (n,))."""
        torch, w = self._torch, self._w
        if self.head == "interp":
            y = self._interp(m)
            return y[:, 2 : 2 + self.K], y[:, 2 + self.K]
        if self.head == "mlp":
            z = (m - w["mlp_xmean"]) / w["mlp_xstd"]
            z = torch.nn.functional.silu(z @ w["mlp_W0"].T + w["mlp_b0"])
            z = torch.nn.functional.silu(z @ w["mlp_W1"].T + w["mlp_b1"])
            y = (z @ w["mlp_W2"].T + w["mlp_b2"]) * w["mlp_ystd"] + w["mlp_ymean"]
        elif self.head == "poly":
            x = self._kb_basis(m)
            xs = torch.cat([torch.ones_like(m[:, :1]), (x[:, 1:] - w["poly_xmean"]) / w["poly_xstd"]], dim=1)
            y = xs @ w["poly_coef"] * w["poly_ystd"] + w["poly_ymean"]
        else:
            raise ValueError(f"unknown head {self.head!r}")
        return y[:, : self.K], y[:, self.K]

    def _decode(self, codes, anchor):
        """PCA codes + anchor -> L1-normalized full frames via grid_sample placement."""
        torch = self._torch
        n = codes.shape[0]
        p, win = self.patch, self.win
        h, w_ = self.frame_shape
        small = (codes @ self._w["pca_components"] + self._w["pca_mean"]).reshape(n, 1, p, p)
        small = torch.clamp(small, min=0.0)
        ys = torch.arange(h, dtype=codes.dtype)
        xs = torch.arange(w_, dtype=codes.dtype)
        wx = xs[None, None, :] - (anchor[:, 0, None, None] - win // 2)
        wy = ys[None, :, None] - (anchor[:, 1, None, None] - win // 2)
        scale = (p - 1) / (win - 1)
        gx = 2.0 * (wx * scale) / (p - 1) - 1.0
        gy = 2.0 * (wy * scale) / (p - 1) - 1.0
        grid = torch.stack([gx.expand(n, h, w_), gy.expand(n, h, w_)], dim=-1)
        canvas = torch.nn.functional.grid_sample(small, grid, mode="bilinear", padding_mode="zeros", align_corners=True)[
            :, 0
        ]
        return canvas / torch.clamp(canvas.sum(dim=(1, 2), keepdim=True), min=1e-12)

    # -- public API ----------------------------------------------------------

    def in_domain(self, m: np.ndarray) -> np.ndarray:
        """True where the motor vector lies inside the recorded motor box."""
        m = np.atleast_2d(np.asarray(m, dtype=float))
        return ((m >= self.box_lo) & (m <= self.box_hi)).all(axis=1)

    def predict(self, m: np.ndarray, sample: bool = False, rng: np.random.Generator | None = None) -> dict[str, np.ndarray]:
        """(n, 4) motors -> dict(image (n, H, W) L1-normalized, flux (n,), in_domain (n,)).

        ``sample=True`` adds real background-speck patterns from the training data.
        """
        torch = self._torch
        mt = torch.as_tensor(np.atleast_2d(np.asarray(m, dtype=float)), dtype=torch.float64)
        with torch.no_grad():
            anchor = self._anchor(mt)
            codes, flux = self._codes_flux(mt)
            image = self._decode(codes, anchor).numpy()
        if sample:
            if self._noise is None:
                raise RuntimeError("weights file has no noise bank; use sample=False")
            rng = rng or np.random.default_rng()
            nb = self._noise
            n_bank = len(nb["ptr"]) - 1
            for i in range(image.shape[0]):
                k = int(rng.integers(0, n_bank))
                s, e = nb["ptr"][k], nb["ptr"][k + 1]
                image[i, nb["rows"][s:e], nb["cols"][s:e]] += nb["vals"][s:e]
        return {"image": image, "flux": flux.numpy(), "in_domain": self.in_domain(m)}


__all__ = ["TESModel", "resolve_weights_path", "WEIGHTS_URL"]
