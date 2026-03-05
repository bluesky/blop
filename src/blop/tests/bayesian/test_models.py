"""Tests for blop.bayesian.models module.

This module tests the GP model classes:
- LatentGP: SingleTaskGP with LatentKernel
- MultiTaskLatentGP: MultiTaskGP with LatentKernel
- LatentConstraintModel: GP with fitness() method for constraint satisfaction
- LatentDirichletClassifier: GP with probabilities() method for classification

Note: These models can be used within Blop by configuring Ax's generation strategy
to use the LatentKernel. These tests document the expected behavior.
"""

import pytest
import torch

from blop.bayesian.kernels import LatentKernel
from blop.bayesian.models import (
    LatentConstraintModel,
    LatentDirichletClassifier,
    LatentGP,
    MultiTaskLatentGP,
)


def make_training_data(n_samples: int = 20, n_inputs: int = 2, n_outputs: int = 1):
    """Create synthetic training data for GP models."""
    torch.manual_seed(42)
    train_X = torch.rand(n_samples, n_inputs, dtype=torch.float64)
    train_Y = train_X.sum(dim=-1, keepdim=True) + 0.1 * torch.randn(n_samples, 1, dtype=torch.float64)
    if n_outputs > 1:
        train_Y = train_Y.expand(-1, n_outputs).clone()
        train_Y[:, 1:] += 0.2 * torch.randn(n_samples, n_outputs - 1, dtype=torch.float64)
    return train_X, train_Y


def make_multitask_data(n_samples: int = 20, n_inputs: int = 2, n_tasks: int = 2):
    """Create multi-task training data (last column is task index)."""
    torch.manual_seed(42)
    train_X = torch.cat(
        [torch.rand(n_samples, n_inputs, dtype=torch.float64), torch.randint(0, n_tasks, (n_samples, 1)).double()],
        dim=-1,
    )
    train_Y = torch.rand(n_samples, 1, dtype=torch.float64)
    return train_X, train_Y


class TestLatentGP:
    """Tests for LatentGP model."""

    def test_initialization_and_structure(self):
        """Test LatentGP initialization, kernel, and mean module."""
        train_X, train_Y = make_training_data()
        model = LatentGP(train_X, train_Y)

        # Basic properties
        assert model.trained is False
        assert isinstance(model.covar_module, LatentKernel)
        assert model.covar_module.num_inputs == train_X.shape[-1]

        # Mean module
        from gpytorch.means import ConstantMean

        assert isinstance(model.mean_module, ConstantMean)

        # Inheritance
        from botorch.models.gp_regression import SingleTaskGP

        assert isinstance(model, SingleTaskGP)

    @pytest.mark.parametrize(
        "skew_dims,n_inputs,expected_entries",
        [
            (True, 3, 3),  # Full rotation: 3*(3-1)/2 = 3
            ([(0, 1), (2, 3)], 4, 2),  # Two groups: 1+1 = 2
        ],
    )
    def test_skew_dims_configurations(self, skew_dims, n_inputs, expected_entries):
        """Test LatentGP with various skew_dims settings."""
        train_X, train_Y = make_training_data(n_inputs=n_inputs)
        model = LatentGP(train_X, train_Y, skew_dims=skew_dims)

        assert model.covar_module.n_skew_entries == expected_entries

    def test_posterior_computation(self):
        """Test posterior shape and sampling."""
        train_X, train_Y = make_training_data(n_samples=20)
        model = LatentGP(train_X, train_Y)
        model.eval()

        test_X = torch.rand(5, 2, dtype=torch.float64)
        posterior = model.posterior(test_X)

        # Shape
        assert posterior.mean.shape == (5, 1)
        assert posterior.variance.shape == (5, 1)

        # Sampling
        samples = posterior.sample(torch.Size((10,)))
        assert samples.shape == (10, 5, 1)


class TestMultiTaskLatentGP:
    """Tests for MultiTaskLatentGP model."""

    def test_initialization(self):
        """Test MultiTaskLatentGP initialization and inheritance."""
        train_X, train_Y = make_multitask_data()
        model = MultiTaskLatentGP(train_X, train_Y, task_feature=-1)

        assert model.trained is False
        assert isinstance(model.covar_module, LatentKernel)

        from botorch.models.multitask import MultiTaskGP

        assert isinstance(model, MultiTaskGP)

    def test_with_skew_dims(self):
        """Test MultiTaskLatentGP with custom skew_dims."""
        train_X, train_Y = make_multitask_data(n_inputs=3)
        model = MultiTaskLatentGP(train_X, train_Y, task_feature=-1, skew_dims=[(0, 1)])

        # Only 2 dims rotate: 1 skew entry
        assert model.covar_module.n_skew_entries == 1


class TestLatentConstraintModel:
    """Tests for LatentConstraintModel and fitness() method."""

    def test_fitness_returns_valid_probabilities(self):
        """Test fitness() returns valid probability distribution."""
        train_X, train_Y = make_training_data(n_outputs=3)
        model = LatentConstraintModel(train_X, train_Y)
        model.eval()

        test_X = torch.rand(5, 2, dtype=torch.float64)
        fitness = model.fitness(test_X, n_samples=128)

        # Shape
        assert fitness.shape == (5, 3)

        # Valid probabilities: non-negative, sum to 1
        assert torch.all(fitness >= 0)
        row_sums = fitness.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_fitness_batched_input(self):
        """Test fitness() with batched input."""
        train_X, train_Y = make_training_data(n_outputs=2)
        model = LatentConstraintModel(train_X, train_Y)
        model.eval()

        test_X = torch.rand(3, 4, 2, dtype=torch.float64)
        fitness = model.fitness(test_X, n_samples=32)

        assert fitness.shape == (3, 4, 2)

    def test_inheritance(self):
        """Test LatentConstraintModel inherits from LatentGP."""
        train_X, train_Y = make_training_data()
        model = LatentConstraintModel(train_X, train_Y)

        assert isinstance(model, LatentGP)
        assert hasattr(model, "fitness")


class TestLatentDirichletClassifier:
    """Tests for LatentDirichletClassifier and probabilities() method."""

    def test_probabilities_returns_valid_distribution(self):
        """Test probabilities() returns valid probability distribution."""
        # Use log-transformed data like DirichletClassificationLikelihood
        torch.manual_seed(42)
        train_X = torch.rand(30, 2, dtype=torch.float64)
        train_Y = torch.log(torch.rand(30, 4, dtype=torch.float64) + 0.1)

        model = LatentDirichletClassifier(train_X, train_Y)
        model.eval()

        test_X = torch.rand(5, 2, dtype=torch.float64)
        probs = model.probabilities(test_X, n_samples=128)

        # Shape
        assert probs.shape == (5, 4)

        # Valid probabilities
        assert torch.all(probs >= 0)
        row_sums = probs.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_probabilities_batched_input(self):
        """Test probabilities() with batched input."""
        torch.manual_seed(42)
        train_X = torch.rand(30, 2, dtype=torch.float64)
        train_Y = torch.log(torch.rand(30, 3, dtype=torch.float64) + 0.1)

        model = LatentDirichletClassifier(train_X, train_Y)
        model.eval()

        test_X = torch.rand(2, 3, 2, dtype=torch.float64)
        probs = model.probabilities(test_X, n_samples=32)

        assert probs.shape == (2, 3, 3)

    def test_probabilities_vs_fitness_equivalence(self):
        """Test that probabilities() and fitness() are mathematically equivalent."""
        train_X, train_Y = make_training_data(n_outputs=3)

        classifier = LatentDirichletClassifier(train_X, train_Y)
        constraint = LatentConstraintModel(train_X, train_Y)
        classifier.eval()
        constraint.eval()

        test_X = torch.rand(3, 2, dtype=torch.float64)

        # With same seed and n_samples, should be identical
        torch.manual_seed(42)
        probs = classifier.probabilities(test_X, n_samples=100)
        torch.manual_seed(42)
        fitness = constraint.fitness(test_X, n_samples=100)

        assert torch.allclose(probs, fitness, atol=1e-10)

    def test_inheritance(self):
        """Test LatentDirichletClassifier inherits from LatentGP."""
        torch.manual_seed(42)
        train_X = torch.rand(20, 2, dtype=torch.float64)
        train_Y = torch.log(torch.rand(20, 3, dtype=torch.float64) + 0.1)

        model = LatentDirichletClassifier(train_X, train_Y)

        assert isinstance(model, LatentGP)
        assert hasattr(model, "probabilities")


class TestModelsEdgeCases:
    """Tests for edge cases across all models."""

    def test_minimal_training_data(self):
        """Test models with single training point."""
        train_X = torch.rand(1, 2, dtype=torch.float64)
        train_Y = torch.rand(1, 1, dtype=torch.float64)

        model = LatentGP(train_X, train_Y)
        assert model is not None

    def test_high_dimensional_input(self):
        """Test models with high-dimensional input."""
        train_X = torch.rand(20, 10, dtype=torch.float64)
        train_Y = torch.rand(20, 1, dtype=torch.float64)

        model = LatentGP(train_X, train_Y)
        model.eval()

        test_X = torch.rand(5, 10, dtype=torch.float64)
        posterior = model.posterior(test_X)

        assert posterior.mean.shape == (5, 1)

    def test_many_outputs(self):
        """Test models with many output dimensions."""
        train_X = torch.rand(20, 2, dtype=torch.float64)
        train_Y = torch.rand(20, 5, dtype=torch.float64)

        model = LatentGP(train_X, train_Y)
        model.eval()

        test_X = torch.rand(3, 2, dtype=torch.float64)
        posterior = model.posterior(test_X)

        assert posterior.mean.shape == (3, 5)
