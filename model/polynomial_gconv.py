import torch
import torch.nn as nn

class PowerGConvDyn(nn.Module):
    """
    Dynamic power-series graph convolution (uses current adjacency matrix (A) in each forward run)
    """
    def __init__(self, in_channels: int, out_channels: int, K: int):
        super().__init__()
        self.K = int(K)
        self.lin = nn.Linear((self.K + 1) * in_channels, out_channels)

    def forward(self, X: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        Zs = [X]
        Z = X
        for _ in range(self.K):
            Z = torch.sparse.mm(A_hat, Z)
            Zs.append(Z)
        H = torch.cat(Zs, dim=1)
        return self.lin(H)