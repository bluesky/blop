from typing import Any

import gpytorch  # type: ignore[import-untyped]
import torch  # type: ignore[import-untyped]
from botorch.models.gp_regression import SingleTaskGP  # type: ignore[import-untyped]
from botorch.models.multitask import MultiTaskGP  # type: ignore[import-untyped]
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.utils.types import DEFAULT, _DefaultType
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.means import ConstantMean
from gpytorch.priors import NormalPrior
from torch import Tensor

from . import kernels


class LatentGP(SingleTaskGP):
    def __init__(
        self,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        train_Tvar: torch.Tensor = None,
        train_Yvar: Tensor | None = None,
        likelihood: Likelihood | None = None,
        input_transform: InputTransform | None = None,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
        skew_dims: bool | list[tuple[int, ...]] = True,
        *args: Any,
        **kwargs: Any,
    ) -> None:

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
            *args,
            **kwargs,
        )

        m = train_Y.shape[-1]
        aug_batch_shape = train_X.shape[:-2] + (torch.Size([m]) if m > 1 else torch.Size())

        self.mean_module = ConstantMean(batch_shape=aug_batch_shape, constant_prior=NormalPrior(0.0, 1.0))
        self.covar_module = kernels.RotatedInputsKernel(
            d=train_X.shape[-1], batch_shape=aug_batch_shape, skew_dims=skew_dims
        )

        # return SingleTaskGP(
        #     train_X=train_X,
        #     train_Y=train_Y,
        #     mean_module=mean,
        #     covar_module=covar,
        #     outcome_transform=None,  # keep if you truly need it
        #     **kwargs,
        # )


# class LatentGP(SingleTaskGP):
#     def __init__(
#         self,
#         train_X: torch.Tensor,
#         train_Y: torch.Tensor,
#         train_Tvar: torch.Tensor = None,
#         train_Yvar: Tensor | None = None,
#         likelihood: Likelihood | None = None,
#         input_transform: InputTransform | None = None,
#         outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
#         skew_dims: bool | list[tuple[int, ...]] = True,
#         *args: Any,
#         **kwargs: Any,
#     ) -> None:

#         *batch_shape, n, d = train_X.shape
#         input_transform = input_transform or Normalize(d=d)
#         # outcome_transform = outcome_transform or Standardize(batch_shape=batch_shape)

#         super().__init__(train_X=train_X,
#                          train_Y=train_Y,
#                          input_transform=input_transform,
#                          outcome_transform=outcome_transform,
#                          *args, **kwargs)

#         self.mean_module = gpytorch.means.ConstantMean(constant_prior=gpytorch.priors.NormalPrior(loc=0, scale=1))

#         self.covar_module = kernels.LatentKernel(
#             num_inputs=train_X.shape[-1],
#             num_outputs=train_Y.shape[-1],
#             skew_dims=skew_dims,
#             priors=True,
#             scale=True,
#             **kwargs,
#         )

#         self.trained: bool = False


class MultiTaskLatentGP(MultiTaskGP):
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
        """
        Takes in a (..., m) dimension tensor and returns a (..., n_classes) tensor
        """
        *input_shape, n_dim = x.shape
        samples = self.posterior(x.reshape(-1, n_dim)).sample(torch.Size((n_samples,))).exp()
        return (samples / samples.sum(-1, keepdim=True)).mean(0).reshape(*input_shape, -1)


class LatentDirichletClassifier(LatentGP):
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
        """
        Takes in a (..., m) dimension tensor and returns a (..., n_classes) tensor
        """
        *input_shape, n_dim = x.shape
        samples = self.posterior(x.reshape(-1, n_dim)).sample(torch.Size((n_samples,))).exp()
        return (samples / samples.sum(-1, keepdim=True)).mean(0).reshape(*input_shape, -1)
