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
        batch_shape: torch.Size,
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
