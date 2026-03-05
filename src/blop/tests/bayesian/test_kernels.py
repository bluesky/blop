"""Tests for blop.bayesian.kernels module.

Tests the public API of LatentKernel - a Matérn kernel with learned affine
transformation using SO(N) parameterization for orthogonal rotations.
"""

import pytest
import torch

from blop.bayesian.kernels import LatentKernel


def test_latent_kernel_init():
    """Can construct LatentKernel with various skew_dims configurations."""
    # Default (all dims rotate together)
    kernel1 = LatentKernel(num_inputs=3)
    assert kernel1.num_inputs == 3
    assert kernel1.nu == 2.5  # default

    # Custom skew groups
    kernel2 = LatentKernel(num_inputs=4, skew_dims=[(0, 1), (2, 3)])
    assert kernel2.num_inputs == 4

    # No rotation
    kernel3 = LatentKernel(num_inputs=2, skew_dims=False)
    assert kernel3.num_inputs == 2

    # With different Matérn smoothness
    kernel4 = LatentKernel(num_inputs=2, nu=1.5)
    assert kernel4.nu == 1.5


def test_latent_kernel_forward_shapes():
    """forward() returns correct covariance shapes."""
    kernel = LatentKernel(num_inputs=3)

    x1 = torch.randn(10, 3, dtype=torch.float64)
    x2 = torch.randn(5, 3, dtype=torch.float64)

    # Square covariance
    cov_square = kernel(x1, x1)
    assert cov_square.shape == (10, 10)

    # Rectangular covariance
    cov_rect = kernel(x1, x2)
    assert cov_rect.shape == (10, 5)

    # Diagonal only
    cov_diag = kernel(x1, x1, diag=True)
    assert cov_diag.shape == (10,)


@pytest.mark.parametrize("nu", [0.5, 1.5, 2.5])
def test_latent_kernel_matern_variants(nu):
    """Works with supported Matérn smoothness values (nu=0.5, 1.5, 2.5)."""
    kernel = LatentKernel(num_inputs=2, nu=nu)
    x = torch.randn(5, 2, dtype=torch.float64)

    # Should compute without error
    cov = kernel(x, x).to_dense()
    assert cov.shape == (5, 5)


def test_latent_kernel_unsupported_nu_raises():
    """Unsupported nu value raises ValueError."""
    kernel = LatentKernel(num_inputs=2, nu=3.5)
    x = torch.randn(5, 2, dtype=torch.float64)

    with pytest.raises(ValueError, match="not supported"):
        kernel(x, x).to_dense()


def test_latent_kernel_invalid_skew_dims():
    """Invalid skew_dims configurations raise appropriate errors."""
    # Duplicate dimensions across groups
    with pytest.raises(ValueError, match="unique"):
        LatentKernel(num_inputs=3, skew_dims=[(0, 1), (1, 2)])

    # Dimension index out of range
    with pytest.raises(ValueError, match="invalid dimension"):
        LatentKernel(num_inputs=3, skew_dims=[(0, 3)])

    # Invalid type
    with pytest.raises((ValueError, TypeError)):
        LatentKernel(num_inputs=3, skew_dims="invalid")


def test_latent_kernel_forward_deterministic():
    """Forward call is deterministic for same input."""
    kernel = LatentKernel(num_inputs=2, nu=2.5)
    x = torch.tensor([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], dtype=torch.float64)

    cov1 = kernel(x, x).to_dense()
    cov2 = kernel(x, x).to_dense()

    assert torch.allclose(cov1, cov2, atol=1e-10)


def test_latent_kernel_golden_values():
    """Validate kernel output against pre-computed expected values.

    FIXME: This kernel is missing the sqrt(2*nu) scaling factor required by the
    standard Matérn formula. The correct Matérn 2.5 formula is:
        k(r) = σ² * (1 + √5*r + 5r²/3) * exp(-√5*r)
    where r = ||x1 - x2|| / lengthscale. The current implementation omits the
    √5 factor (and √3 for nu=1.5), causing correlation to decay more slowly
    than a true Matérn kernel. This should be fixed when refactoring to use
    native BoTorch/GPyTorch kernels.
    """
    kernel = LatentKernel(num_inputs=2, skew_dims=False, nu=2.5, scale_output=True)
    kernel.lengthscales = torch.tensor([[1.0, 2.0]])
    kernel.outputscale = torch.tensor([1.5])

    x = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 2.0],
        ],
        dtype=torch.float64,
    )

    # Pre-computed expected output for this configuration
    expected = torch.tensor(
        [
            [1.5000, 1.2876, 1.2876],
            [1.2876, 1.5000, 1.1235],
            [1.2876, 1.1235, 1.5000],
        ],
        dtype=torch.float64,
    )

    actual = kernel(x, x).to_dense()

    assert torch.allclose(actual, expected, atol=1e-4), (
        f"Kernel output doesn't match expected values.\nExpected:\n{expected}\nActual:\n{actual}"
    )
