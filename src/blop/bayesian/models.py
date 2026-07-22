"""Custom models for doing Bayesian optimization."""

from typing import Any

import gpytorch
import torch
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.multitask import MultiTaskGP

from . import kernels


class LatentGP(SingleTaskGP):
    """Latent model GP."""

    def __init__(
        self,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        skew_dims: bool | list[tuple[int, ...]] = True,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        # Disable outcome transform to avoid shape mismatches with multi-output kernel
        super().__init__(train_X, train_Y, *args, outcome_transform=None, **kwargs)

        self.mean_module = gpytorch.means.ConstantMean(constant_prior=gpytorch.priors.NormalPrior(loc=0, scale=1))

        self.covar_module = kernels.LatentKernel(
            num_inputs=train_X.shape[-1],
            num_outputs=train_Y.shape[-1],
            skew_dims=skew_dims,
            priors=True,
            scale=True,
            **kwargs,
        )

        self.trained: bool = False


class MultiTaskLatentGP(MultiTaskGP):
    """Multi-task latent model GP."""

    def __init__(
        self,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        task_feature: int,
        skew_dims: bool | list[tuple[int, ...]] = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(train_X, train_Y, task_feature, *args, **kwargs)

        self.mean_module = gpytorch.means.ConstantMean(constant_prior=gpytorch.priors.NormalPrior(loc=0, scale=1))

        self.covar_module = kernels.LatentKernel(
            num_inputs=self.num_non_task_features,
            skew_dims=skew_dims,
            priors=True,
            scale=True,
            **kwargs,
        )

        self.trained: bool = False


class LatentConstraintModel(LatentGP):
    """Constraint latent model GP."""

    def __init__(
        self,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        skew_dims: bool | list[tuple[int, ...]] = True,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(train_X, train_Y, skew_dims, *args, **kwargs)

        self.trained: bool = False

    def fitness(self, x: torch.Tensor, n_samples: int = 1024) -> torch.Tensor:
        """Given a (..., m) dimension tensor and returns a (..., n_classes) tensor."""
        *input_shape, n_dim = x.shape
        samples = self.posterior(x.reshape(-1, n_dim)).sample(torch.Size((n_samples,))).exp()
        return (samples / samples.sum(-1, keepdim=True)).mean(0).reshape(*input_shape, -1)


class LatentDirichletClassifier(LatentGP):
    """Classifier latent model GP."""

    def __init__(
        self,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        skew_dims: bool | list[tuple[int, ...]] = True,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(train_X, train_Y, skew_dims, *args, **kwargs)

        self.trained: bool = False

    def probabilities(self, x: torch.Tensor, n_samples: int = 256) -> torch.Tensor:
        """Given a (..., m) dimension tensor and returns a (..., n_classes) tensor."""
        *input_shape, n_dim = x.shape
        samples = self.posterior(x.reshape(-1, n_dim)).sample(torch.Size((n_samples,))).exp()
        return (samples / samples.sum(-1, keepdim=True)).mean(0).reshape(*input_shape, -1)
