import numpy as np
import scipy.sparse as sp
import torch
from torch import nn


class SimpleGraphConvolution(nn.Module):
    """
    SGC-style block that assumes S_spatial is a normalized adjacency matrix.

    x:  [B, N, F]
    returns: [B, N] if out_channels==1 else [B, N, out_channels]
    """

    def __init__(
            self,
            S_spatial,  # scipy.sparse matrix OR torch.sparse_coo_tensor
            in_channels: int,
            out_channels: int = 1,
            K: int = 1,  # number of propagation steps
            dropout: float = 0.0,
            bias: bool = True,
    ):
        super().__init__()
        self.K = K
        self.lin = nn.Linear(in_channels, out_channels, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Convert to a torch sparse COO tensor and store as a buffer (moves with .to(device))
        if sp.issparse(S_spatial):
            coo = S_spatial.tocoo()
            indices = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long)
            values = torch.tensor(coo.data, dtype=torch.float32)
            A = torch.sparse_coo_tensor(indices, values, torch.Size(coo.shape)).coalesce()
        elif isinstance(S_spatial, torch.Tensor) and S_spatial.is_sparse:
            A = S_spatial.coalesce().to(dtype=torch.float32)
        else:
            raise TypeError("S_spatial_norm must be a scipy.sparse matrix or a torch sparse COO tensor.")
        self.register_buffer("A_hat", A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, F]
        B, N, F = x.shape
        A = self.A_hat

        h = x.permute(1, 0, 2).reshape(N, B * F)
        for _ in range(self.K):
            h = torch.sparse.mm(A, h)
        h = h.reshape(N, B, F).permute(1, 0, 2)

        h = self.dropout(h)
        out = self.lin(h)
        if out.shape[-1] == 1:
            out = out.squeeze(-1)
        return out
