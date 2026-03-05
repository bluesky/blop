"""Tests for blop.bayesian.kernels module.

This module tests the LatentKernel class, which implements a Matérn kernel with
learned affine transformation using SO(N) parameterization for orthogonal rotations.

Mathematical background:
- The kernel applies a transformation T = D @ S where:
  - D is a diagonal matrix with 1/lengthscale (ARD)
  - S is an orthogonal rotation matrix from exp(skew_matrix) - generates SO(N)
- The skew matrix entries are constrained to [-2π, 2π]
- Supports Matérn variants with ν=0.5, 1.5, 2.5

Note: This module can be used within Blop by configuring Ax's generation strategy
to use the LatentKernel. These tests document the expected behavior.
"""

import numpy as np
import pytest
import torch

from blop.bayesian.kernels import LatentKernel


class TestLatentKernelInitialization:
    """Tests for LatentKernel initialization and configuration."""

    def test_default_initialization(self):
        """Test default initialization and basic properties."""
        kernel = LatentKernel(num_inputs=3)

        # Basic properties
        assert kernel.num_inputs == 3
        assert kernel.num_outputs == 1
        assert kernel.nu == 2.5  # default Matérn smoothness
        assert kernel.scale_output is True
        assert kernel.is_stationary is True

        # Lengthscales shape and positivity
        assert kernel.lengthscales.shape == (1, 3)
        assert torch.all(kernel.lengthscales > 0)

        # With skew_dims=True (default), n*(n-1)/2 skew entries for n=3: 3 entries
        assert kernel.n_skew_entries == 3

    @pytest.mark.parametrize(
        "skew_dims,num_inputs,expected_entries,expected_groups",
        [
            (True, 4, 6, 1),  # 4*(4-1)/2 = 6
            (False, 4, 6, 1),  # Same structure, no learned rotation
            ([(0, 1), (2, 3)], 4, 2, 2),  # Two 2-dim groups: 1+1 = 2
            ([(0, 1, 2)], 3, 3, 1),  # One 3-dim group: 3*(3-1)/2 = 3
            ([(0, 1)], 3, 1, 1),  # Partial: only 2 dims rotate
        ],
    )
    def test_skew_dims_configurations(self, skew_dims, num_inputs, expected_entries, expected_groups):
        """Test various skew_dims configurations."""
        kernel = LatentKernel(num_inputs=num_inputs, skew_dims=skew_dims)

        assert kernel.n_skew_entries == expected_entries
        assert len(kernel.skew_dims) == expected_groups

    def test_optional_features(self):
        """Test initialization with optional features disabled."""
        # Without priors
        kernel_no_priors = LatentKernel(num_inputs=2, priors=False)
        assert not hasattr(kernel_no_priors, "lengthscales_prior")

        # Without output scaling
        kernel_no_scale = LatentKernel(num_inputs=2, scale_output=False)
        assert kernel_no_scale.scale_output is False
        assert not hasattr(kernel_no_scale, "raw_outputscale")


class TestLatentKernelValidation:
    """Tests for input validation."""

    def test_invalid_skew_dims(self):
        """Test that invalid skew_dims configurations raise errors."""
        # Duplicate dimensions
        with pytest.raises(ValueError, match="unique"):
            LatentKernel(num_inputs=3, skew_dims=[(0, 1), (1, 2)])

        # Out of range
        with pytest.raises(ValueError, match="invalid dimension"):
            LatentKernel(num_inputs=3, skew_dims=[(0, 3)])

        # Invalid type (string is iterable but chars can't convert to tensor)
        with pytest.raises((ValueError, TypeError)):
            LatentKernel(num_inputs=3, skew_dims="invalid")


class TestLatentKernelParameters:
    """Tests for parameter getters and setters."""

    def test_parameter_getters_and_setters(self):
        """Test all parameter properties and their setters."""
        kernel = LatentKernel(num_inputs=3)

        # Lengthscales
        new_lengthscales = torch.tensor([[0.5, 1.0, 1.5]], dtype=torch.float64)
        kernel.lengthscales = new_lengthscales
        assert torch.allclose(kernel.lengthscales, new_lengthscales, atol=1e-5)

        # Lengthscales from numpy
        np_lengthscales = np.array([[0.3, 0.5, 0.7]])
        kernel.lengthscales = np_lengthscales
        assert torch.allclose(kernel.lengthscales, torch.tensor(np_lengthscales), atol=1e-5)

        # Skew entries (shape is (1, n_skew_entries))
        new_skew = torch.tensor([[0.5, -0.5, 1.0]], dtype=torch.float64)
        kernel.skew_entries = new_skew
        assert torch.allclose(kernel.skew_entries, new_skew, atol=1e-5)

        # Outputscale
        new_scale = torch.tensor([2.5], dtype=torch.float64)
        kernel.outputscale = new_scale
        assert torch.allclose(kernel.outputscale, new_scale, atol=1e-5)

    def test_skew_entries_constraints(self):
        """Test skew entries are within [-2π, 2π] constraint."""
        kernel = LatentKernel(num_inputs=3)

        skew_entries = kernel.skew_entries
        assert torch.all(skew_entries >= -2 * np.pi)
        assert torch.all(skew_entries <= 2 * np.pi)


class TestLatentKernelMatrices:
    """Tests for matrix computations - the core mathematical properties."""

    def test_diagonal_matrix_structure(self):
        """Test diag_matrix is diagonal with 1/lengthscale entries."""
        kernel = LatentKernel(num_inputs=3)

        D = kernel.diag_matrix
        assert D.shape == (1, 3, 3)

        D0 = D[0]  # First (only) output
        # Off-diagonal should be zero
        mask = ~torch.eye(3, dtype=bool)
        assert torch.allclose(D0[mask], torch.zeros_like(D0[mask]))

        # Diagonal should be 1/lengthscale
        expected_diag = 1.0 / kernel.lengthscales[0]
        assert torch.allclose(torch.diag(D0), expected_diag, atol=1e-5)

    def test_skew_matrix_orthogonality_and_determinant(self):
        """Test that exp(skew) produces orthogonal rotation in SO(N).

        Key mathematical property: exp(skew) generates SO(N).
        - Orthogonal: S @ S.T = I
        - Proper rotation: det(S) = 1
        """
        kernel = LatentKernel(num_inputs=4)

        # Set non-zero skew entries
        kernel.skew_entries = torch.tensor([[0.5, -0.3, 0.8, 0.2, -0.1, 0.6]])
        S = kernel.skew_matrix[0]

        # Orthogonality
        identity = torch.eye(4, dtype=torch.float64)
        assert torch.allclose(S @ S.T, identity, atol=1e-10)

        # Determinant = 1 (proper rotation, not reflection)
        assert torch.allclose(torch.linalg.det(S), torch.tensor(1.0, dtype=torch.float64), atol=1e-10)

        # Zero skew entries should give identity
        kernel2 = LatentKernel(num_inputs=3)
        S2 = kernel2.skew_matrix[0]
        assert torch.allclose(S2, torch.eye(3, dtype=torch.float64), atol=1e-10)

    def test_latent_transform_composition(self):
        """Test latent_transform = diag_matrix @ skew_matrix."""
        kernel = LatentKernel(num_inputs=3)
        kernel.skew_entries = torch.tensor([[0.3, -0.2, 0.5]])

        expected = torch.matmul(kernel.diag_matrix, kernel.skew_matrix)
        assert torch.allclose(kernel.latent_transform, expected, atol=1e-10)


class TestLatentKernelForward:
    """Tests for forward pass (covariance computation)."""

    def test_covariance_shapes(self):
        """Test forward pass returns correct shapes."""
        kernel = LatentKernel(num_inputs=3)

        # Square covariance
        x = torch.randn(10, 3, dtype=torch.float64)
        assert kernel(x, x).shape == (10, 10)

        # Rectangular covariance
        x2 = torch.randn(5, 3, dtype=torch.float64)
        assert kernel(x, x2).shape == (10, 5)

        # Diagonal only
        assert kernel(x, x, diag=True).shape == (10,)

    def test_covariance_properties(self):
        """Test mathematical properties of covariance matrix."""
        kernel = LatentKernel(num_inputs=2)
        x = torch.randn(8, 2, dtype=torch.float64)
        cov = kernel(x, x).to_dense()

        # Positive diagonal
        assert torch.all(torch.diag(cov) > 0)

        # Symmetric
        assert torch.allclose(cov, cov.T, atol=1e-10)

        # Positive semi-definite (eigenvalues >= 0)
        eigenvalues = torch.linalg.eigvalsh(cov)
        assert torch.all(eigenvalues >= -1e-10)

    def test_distance_affects_covariance(self):
        """Test that closer points have higher covariance."""
        kernel = LatentKernel(num_inputs=2)

        x1 = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
        x_close = torch.tensor([[0.01, 0.01]], dtype=torch.float64)
        x_far = torch.tensor([[10.0, 10.0]], dtype=torch.float64)

        cov_close = kernel(x1, x_close).to_dense().item()
        cov_far = kernel(x1, x_far).to_dense().item()

        assert cov_close > cov_far


class TestLatentKernelMaternVariants:
    """Tests for Matérn smoothness variants."""

    @pytest.mark.parametrize("nu", [0.5, 1.5, 2.5])
    def test_supported_nu_values(self, nu):
        """Test that supported nu values produce valid PSD covariances."""
        kernel = LatentKernel(num_inputs=2, nu=nu)
        assert kernel.nu == nu

        x = torch.randn(5, 2, dtype=torch.float64)
        cov = kernel(x, x).to_dense()

        # Should be positive semi-definite
        eigenvalues = torch.linalg.eigvalsh(cov)
        assert torch.all(eigenvalues >= -1e-10)

    def test_different_nu_different_covariance(self):
        """Test that different nu values produce different covariances."""
        x = torch.randn(5, 2, dtype=torch.float64)

        cov_05 = LatentKernel(num_inputs=2, nu=0.5)(x, x).to_dense()
        cov_25 = LatentKernel(num_inputs=2, nu=2.5)(x, x).to_dense()

        assert not torch.allclose(cov_05, cov_25, atol=1e-5)

    def test_unsupported_nu_raises(self):
        """Test that unsupported nu raises ValueError."""
        kernel = LatentKernel(num_inputs=2, nu=3.5)
        x = torch.randn(5, 2, dtype=torch.float64)

        with pytest.raises(ValueError, match="not supported"):
            kernel(x, x).to_dense()  # Lazy eval requires to_dense


class TestLatentKernelEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_point_and_high_dim(self):
        """Test kernel with single point and high-dimensional input."""
        # Single point
        kernel1 = LatentKernel(num_inputs=2)
        x1 = torch.randn(1, 2, dtype=torch.float64)
        cov1 = kernel1(x1, x1).to_dense()
        assert cov1.shape == (1, 1)
        assert cov1.item() > 0

        # High dimensional
        kernel2 = LatentKernel(num_inputs=10)
        x2 = torch.randn(5, 10, dtype=torch.float64)
        assert kernel2(x2, x2).shape == (5, 5)

    def test_extreme_lengthscales(self):
        """Test that lengthscale affects correlation as expected."""
        kernel = LatentKernel(num_inputs=2)
        x = torch.randn(3, 2, dtype=torch.float64)

        # Small lengthscale = less correlation
        kernel.lengthscales = torch.tensor([[0.001, 0.001]])
        cov_small = kernel(x, x).to_dense()

        # Large lengthscale = more correlation
        kernel.lengthscales = torch.tensor([[100.0, 100.0]])
        cov_large = kernel(x, x).to_dense()

        assert cov_large[0, 1].item() > cov_small[0, 1].item()

    def test_large_skew_entries_still_orthogonal(self):
        """Test orthogonality with skew entries near constraint bounds."""
        kernel = LatentKernel(num_inputs=3)

        max_skew = 2 * np.pi - 0.1
        kernel.skew_entries = torch.tensor([[max_skew, -max_skew, max_skew / 2]])

        S = kernel.skew_matrix[0]
        identity = torch.eye(3, dtype=torch.float64)
        assert torch.allclose(S @ S.T, identity, atol=1e-10)

    def test_batched_input(self):
        """Test kernel with batched input dimensions."""
        kernel = LatentKernel(num_inputs=2)
        x = torch.randn(3, 4, 2, dtype=torch.float64)

        cov = kernel(x, x)
        assert cov.shape[:2] == (3, 4)
