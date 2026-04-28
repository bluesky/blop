
import torch
from botorch.models.utils.gpytorch_modules import get_matern_kernel_with_gamma_prior
from gpytorch.kernels import Kernel


class RotatedInputsKernel(Kernel):
    """Applies a learned orthogonal rotation (blockwise if desired), then delegates to a base kernel."""

    is_stationary = True
    has_lengthscale = False  # lengthscale lives in the base kernel

    def __init__(
        self,
        d: int,
        batch_shape: torch.Size = torch.Size(),
        skew_dims: bool | list[tuple[int, ...]] = True,
    ):
        super().__init__(batch_shape=batch_shape)

        self.d = d
        # Choose groups of dims that can rotate among themselves
        if isinstance(skew_dims, bool):
            self.groups = [tuple(range(d))] if skew_dims else []
        else:
            self.groups = [tuple(g) for g in skew_dims]

        # Base kernel = ScaleKernel(MaternKernel(...)) with your Gamma priors
        self.base = get_matern_kernel_with_gamma_prior(ard_num_dims=d, batch_shape=batch_shape)

        # One unconstrained matrix per group; we’ll skew-symmetrize then expm -> orthogonal
        self.raw_group_mats = torch.nn.ParameterList()
        for i, g in enumerate(self.groups):
            k = len(g)
            raw = torch.zeros(int(k * (k - 1) / 2), dtype=torch.double)
            setattr(self, f"raw_group_entries_{i}", torch.nn.Parameter(raw))

            # self.register_constraint(f"raw_group_mats.{i}", Interval(-2 * math.pi, 2 * math.pi))

    def _rotation(self) -> torch.Tensor:
        # Start with identity, then fill block rotations
        A = torch.zeros(
            (self.d, self.d),
            dtype=torch.double,
            device=self.raw_group_entries_0.device if self.groups else None,
        )
        A = A.expand(*self.batch_shape, self.d, self.d).clone() if self.batch_shape else A

        for i, g in enumerate(self.groups):
            row, col = torch.triu_indices(len(g), len(g), 1)
            A[..., tuple(torch.tensor(g)[row]), tuple(torch.tensor(g)[col])] = getattr(self, f"raw_group_entries_{i}")

        return torch.linalg.matrix_exp(A - A.transpose(-1, -2))

    def _transform(self, X: torch.Tensor) -> torch.Tensor:
        # X is (..., n, d). Right-multiply by R^T to rotate features.
        R = self._rotation()
        return torch.matmul(X, R.transpose(-1, -2))

    def forward(self, x1, x2, diag=False, **params):
        return self.base(self._transform(x1), self._transform(x2), diag=diag, **params)


# class LatentKernel(gpytorch.kernels.Kernel):
#     @property
#     def is_stationary(self) -> bool:
#         return True

#     num_outputs: int = 1
#     batch_inverse_lengthscale: float = 1e6

#     def __init__(
#         self,
#         num_inputs: int = 1,
#         skew_dims: bool | Iterable[tuple[int, ...]] = True,
#         priors: bool = True,
#         scale_output: bool = True,
#         **kwargs,
#     ) -> None:
#         super().__init__()

#         self.num_inputs: int = num_inputs
#         self.scale_output: bool = scale_output

#         self.nu: float = kwargs.get("nu", 2.5)

#         if type(skew_dims) is bool:
#             if skew_dims:
#                 self.skew_dims: list[torch.Tensor] = [torch.arange(self.num_inputs)]
#             else:
#                 self.skew_dims = [torch.arange(num_inputs)]
#         elif hasattr(skew_dims, "__iter__"):
#             self.skew_dims = [torch.tensor(np.atleast_1d(skew_group)) for skew_group in skew_dims]
#         else:
#             raise ValueError('arg "skew_dims" must be True, False, or an iterable of tuples of ints.')

#         # if not all([len(skew_group) >= 2 for skew_group in self.skew_dims]):
#         #     raise ValueError("must have at least two dims per skew group")
#         skewed_dims = [dim.item() for skew_group in self.skew_dims for dim in skew_group]
#         if not len(set(skewed_dims)) == len(skewed_dims):
#             raise ValueError("values in skew_dims must be unique")
#         if not max(skewed_dims) < self.num_inputs:
#             raise ValueError("invalid dimension index in skew_dims")

#         skew_group_submatrix_indices: list[torch.Tensor] = []
#         for dim in range(self.num_outputs):
#             for skew_group in self.skew_dims:
#                 j, k = skew_group[torch.triu_indices(len(skew_group), len(skew_group), 1)].unsqueeze(1)
#                 i = dim * torch.ones(j.shape).long()
#                 skew_group_submatrix_indices.append(torch.cat((i, j, k), dim=0))

#         self.diag_matrix_indices: list[torch.Tensor] = [
#             torch.kron(torch.arange(self.num_outputs), torch.ones(self.num_inputs)).long(),
#             *2 * [torch.arange(self.num_inputs).repeat(self.num_outputs)],
#         ]

#         self.skew_matrix_indices: tuple[torch.Tensor, ...] = (
#             tuple(torch.cat(skew_group_submatrix_indices, dim=1))
#             if len(skew_group_submatrix_indices) > 0
#             else (torch.tensor([]), torch.tensor([]))
#         )

#         self.n_skew_entries: int = len(self.skew_matrix_indices[0])

#         lengthscale_constraint = gpytorch.constraints.Positive()
#         raw_lengthscales_initial = lengthscale_constraint.inverse_transform(torch.tensor(1e-1)) * torch.ones(
#             self.num_outputs, self.num_inputs, dtype=torch.double
#         )

#         self.register_parameter(name="raw_lengthscales", parameter=torch.nn.Parameter(raw_lengthscales_initial))
#         self.register_constraint(param_name="raw_lengthscales", constraint=lengthscale_constraint)

#         if priors:
#             self.register_prior(
#                 name="lengthscales_prior",
#                 prior=gpytorch.priors.GammaPrior(concentration=3, rate=6),
#                 param_or_closure=lambda m: m.lengthscales,
#                 setting_closure=lambda m, v: m._set_lengthscales(v),
#             )

#         if self.n_skew_entries > 0:
#             skew_entries_constraint = gpytorch.constraints.Interval(-2 * np.pi, 2 * np.pi)
#             skew_entries_initial = torch.zeros((self.num_outputs, self.n_skew_entries), dtype=torch.float64)
#             self.register_parameter(name="raw_skew_entries", parameter=torch.nn.Parameter(skew_entries_initial))
#             self.register_constraint(param_name="raw_skew_entries", constraint=skew_entries_constraint)

#         if self.scale_output:
#             outputscale_constraint = gpytorch.constraints.Positive()
#             outputscale_prior = gpytorch.priors.GammaPrior(concentration=2, rate=0.15)

#             self.register_parameter(
#                 name="raw_outputscale",
#                 parameter=torch.nn.Parameter(torch.ones(1, dtype=torch.double)),
#             )

#             self.register_constraint("raw_outputscale", constraint=outputscale_constraint)

#             self.register_prior(
#                 name="outputscale_prior",
#                 prior=outputscale_prior,
#                 param_or_closure=lambda m: m.outputscale,
#                 setting_closure=lambda m, v: m._set_outputscale(v),
#             )

#     @property
#     def lengthscales(self) -> torch.Tensor:
#         return self.raw_lengthscales_constraint.transform(self.raw_lengthscales)

#     @lengthscales.setter
#     def lengthscales(self, value: torch.Tensor | float | np.ndarray) -> None:
#         self._set_lengthscales(value)

#     @property
#     def skew_entries(self) -> torch.Tensor:
#         return self.raw_skew_entries_constraint.transform(self.raw_skew_entries)

#     @skew_entries.setter
#     def skew_entries(self, value: torch.Tensor | float | np.ndarray) -> None:
#         self._set_skew_entries(value)

#     @property
#     def outputscale(self) -> torch.Tensor:
#         return self.raw_outputscale_constraint.transform(self.raw_outputscale)

#     @outputscale.setter
#     def outputscale(self, value: torch.Tensor | float | np.ndarray) -> None:
#         self._set_outputscale(value)

#     def _set_lengthscales(self, value: torch.Tensor | float | np.ndarray) -> None:
#         if not torch.is_tensor(value):
#             value = torch.as_tensor(value).to(self.raw_lengthscales)
#         self.initialize(raw_lengthscales=self.raw_lengthscales_constraint.inverse_transform(value))

#     def _set_skew_entries(self, value: torch.Tensor | float | np.ndarray) -> None:
#         if not torch.is_tensor(value):
#             value = torch.as_tensor(value).to(self.raw_skew_entries)
#         self.initialize(raw_skew_entries=self.raw_skew_entries_constraint.inverse_transform(value))

#     def _set_outputscale(self, value: torch.Tensor | float | np.ndarray) -> None:
#         if not torch.is_tensor(value):
#             value = torch.as_tensor(value).to(self.raw_outputscale)
#         self.initialize(raw_outputscale=self.raw_outputscale_constraint.inverse_transform(value))

#     @property
#     def skew_matrix(self) -> torch.Tensor:
#         S = torch.zeros((self.num_outputs, self.num_inputs, self.num_inputs), dtype=torch.float64)
#         if self.n_skew_entries > 0:
#             # to construct an orthogonal matrix. fun fact: exp(skew(N)) is the generator of SO(N)
#             S[self.skew_matrix_indices] = self.skew_entries
#             S += -S.transpose(-1, -2)
#         return torch.linalg.matrix_exp(S)

#     @property
#     def diag_matrix(self) -> torch.Tensor:
#         D = torch.zeros((self.num_outputs, self.num_inputs, self.num_inputs), dtype=torch.float64)
#         D[self.diag_matrix_indices] = self.lengthscales.ravel() ** -1
#         return D

#     @property
#     def latent_transform(self) -> torch.Tensor:
#         return torch.matmul(self.diag_matrix, self.skew_matrix)

#     def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, **params: dict) -> torch.Tensor:
#         # adapted from the Matern kernel
#         mean = x1.reshape(-1, x1.size(-1)).mean(0)[(None,) * (x1.dim() - 1)]

#         transform = self.latent_transform.unsqueeze(1)

#         trans_x1 = torch.matmul(transform, (x1 - mean).unsqueeze(-1)).squeeze(-1)
#         trans_x2 = torch.matmul(transform, (x2 - mean).unsqueeze(-1)).squeeze(-1)

#         distance = self.covar_dist(trans_x1, trans_x2, diag=diag, **params)

#         if self.num_outputs == 1:
#             distance = distance.squeeze(0)

#         outputscale = self.outputscale if self.scale_output else 1.0

#         # special cases of the Matern function
#         if self.nu == 0.5:
#             return outputscale * torch.exp(-distance)
#         if self.nu == 1.5:
#             return outputscale * (1 + distance) * torch.exp(-distance)
#         if self.nu == 2.5:
#             return outputscale * (1 + distance + distance**2 / 3) * torch.exp(-distance)
#         raise ValueError(f"nu = {self.nu} is not supported")
