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
    """Validate kernel output against mathematically computed expected values.

    With skew_dims=False (no rotation) and known lengthscales, we can compute
    the expected Matérn 2.5 covariance manually. The kernel uses the formula:
        k(x1, x2) = outputscale * (1 + d + d^2/3) * exp(-d)
    where d = ||D @ (x1 - x2)||_2 and D = diag(1/lengthscales).

    FIXME: This kernel is missing the sqrt(2*nu) scaling factor required by the
    standard Matérn formula. The correct Matérn 2.5 formula is:
        k(r) = σ² * (1 + √5*r + 5r²/3) * exp(-√5*r)
    where r = ||x1 - x2|| / lengthscale. The current implementation omits the
    √5 factor (and √3 for nu=1.5), causing correlation to decay more slowly
    than a true Matérn kernel. This should be fixed when refactoring to use
    native BoTorch/GPyTorch kernels.
    """
    import math

    # Setup: 2D kernel with no rotation, explicit lengthscales
    kernel = LatentKernel(num_inputs=2, skew_dims=False, nu=2.5, scale_output=True)
    kernel.lengthscales = torch.tensor([[1.0, 2.0]])  # different scales per dim
    kernel.outputscale = torch.tensor([1.5])

    # Test points
    x = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 2.0],
        ],
        dtype=torch.float64,
    )

    # Compute expected values manually
    # The kernel: centers x by subtracting mean, then applies D @ S @ (x - mean)
    # With S=I (skew_dims=False), this is just D @ (x - mean) where D = diag(1/lengthscales)
    mean = x.mean(dim=0)  # [1/3, 2/3]
    lengthscales = torch.tensor([1.0, 2.0])
    outputscale = 1.5

    def matern25_prescaled(d):
        """Matérn 2.5 kernel function (pre-scaled form): (1 + d + d^2/3) * exp(-d)"""
        return (1 + d + d**2 / 3) * math.exp(-d)

    def compute_distance(p1, p2):
        """Compute scaled distance: ||D @ (p1 - p2)|| where D = diag(1/lengthscales)"""
        diff = p1 - p2
        scaled_diff = diff / lengthscales
        return torch.norm(scaled_diff).item()

    # Expected covariance matrix (3x3)
    n = len(x)
    x_centered = x - mean
    expected = torch.zeros((n, n), dtype=torch.float64)
    for i in range(n):
        for j in range(n):
            d = compute_distance(x_centered[i], x_centered[j])
            expected[i, j] = outputscale * matern25_prescaled(d)

    # Get actual kernel output
    actual = kernel(x, x).to_dense()

    # Verify diagonal is outputscale (self-covariance with d=0)
    assert torch.allclose(torch.diag(actual), torch.full((n,), outputscale, dtype=torch.float64), atol=1e-10)

    # Verify full matrix matches expected values
    assert torch.allclose(actual, expected, atol=1e-10), (
        f"Kernel output doesn't match expected Matérn 2.5 values.\nExpected:\n{expected}\nActual:\n{actual}"
    )
