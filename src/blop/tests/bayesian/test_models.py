"""Tests for blop.bayesian.models module.

Tests the public API of GP model classes:
- LatentGP, MultiTaskLatentGP: initialization
- LatentConstraintModel: fitness() method
- LatentDirichletClassifier: probabilities() method
"""

import torch

from blop.bayesian.models import (
    LatentConstraintModel,
    LatentDirichletClassifier,
    LatentGP,
    MultiTaskLatentGP,
)


def test_latent_gp_init():
    """Can construct LatentGP with training data."""
    torch.manual_seed(42)
    train_X = torch.rand(20, 3, dtype=torch.float64)
    train_Y = torch.rand(20, 1, dtype=torch.float64)

    model = LatentGP(train_X, train_Y)
    assert model is not None

    # With custom skew_dims
    model2 = LatentGP(train_X, train_Y, skew_dims=[(0, 1)])
    assert model2 is not None


def test_multi_task_latent_gp_init():
    """Can construct MultiTaskLatentGP with training data."""
    torch.manual_seed(42)
    n_samples = 20
    n_inputs = 2

    # Last column is task index
    train_X = torch.cat(
        [torch.rand(n_samples, n_inputs, dtype=torch.float64), torch.randint(0, 2, (n_samples, 1)).double()],
        dim=-1,
    )
    train_Y = torch.rand(n_samples, 1, dtype=torch.float64)

    model = MultiTaskLatentGP(train_X, train_Y, task_feature=-1)
    assert model is not None


def test_latent_constraint_model_fitness():
    """fitness() returns valid probability distribution."""
    torch.manual_seed(42)
    train_X = torch.rand(20, 2, dtype=torch.float64)
    train_Y = torch.rand(20, 3, dtype=torch.float64)  # 3 classes

    model = LatentConstraintModel(train_X, train_Y)
    model.eval()

    test_X = torch.rand(5, 2, dtype=torch.float64)
    fitness = model.fitness(test_X, n_samples=64)

    # Correct shape
    assert fitness.shape == (5, 3)

    # Valid probabilities: non-negative, sum to 1
    assert torch.all(fitness >= 0)
    row_sums = fitness.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_latent_dirichlet_classifier_probabilities():
    """probabilities() returns valid probability distribution."""
    torch.manual_seed(42)
    train_X = torch.rand(20, 2, dtype=torch.float64)
    # Log-transformed data (like DirichletClassificationLikelihood produces)
    train_Y = torch.log(torch.rand(20, 4, dtype=torch.float64) + 0.1)

    model = LatentDirichletClassifier(train_X, train_Y)
    model.eval()

    test_X = torch.rand(5, 2, dtype=torch.float64)
    probs = model.probabilities(test_X, n_samples=64)

    # Correct shape
    assert probs.shape == (5, 4)

    # Valid probabilities: non-negative, sum to 1
    assert torch.all(probs >= 0)
    row_sums = probs.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
