import scipy.sparse as sp
import torch
import torch.nn as nn

from utils.helper_methods import scipy_to_torch_sparse, torch_sparse_sym_norm, to_csr
from model.polynomial_gconv import PowerGConvDyn

class DisjointSTModel(nn.Module):
    """
    Disjoint spatio-temporal model.

    Two orders are supported:
      - order="ST": Spatial GNN per time-slice -> Temporal GRU per node
      - order="TS": Temporal GRU per node -> Spatial GNN once on node embeddings
    """
    def __init__(self,
                 S_spatial: sp.spmatrix,     # (N x N) spatial adjacency
                 T: int,                     # window length (time steps)
                 F_in: int = 1,
                 spatial_hidden=(64, 64),
                 temporal_hidden: int = 64,
                 K: int = 2,
                 order: str = "ST",          # "ST" or "TS" (see method description for explanation)
                 dropout: float = 0.1,
                 device: str | torch.device = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.order = order.upper()
        assert self.order in {"ST", "TS"}

        S = to_csr(S_spatial)
        self.N = S.shape[0]
        self.T = int(T)
        self.F_in = int(F_in)

        # Static normalized spatial adjacency
        A_sp = scipy_to_torch_sparse(S)                     
        A_hat = torch_sparse_sym_norm(A_sp).coalesce()
        self.register_buffer("A_hat", A_hat) 

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        if self.order == "ST":
            # Spatial stack on each snapshot
            sdims = [F_in, *spatial_hidden]
            self.spatial_layers = nn.ModuleList(
                [PowerGConvDyn(sdims[i], sdims[i+1], K=K) for i in range(len(sdims)-1)]
            )
            # Temporal module per node
            self.temporal = nn.GRU(input_size=sdims[-1],
                                   hidden_size=temporal_hidden,
                                   num_layers=1,
                                   batch_first=True)
            self.head = nn.Linear(temporal_hidden, 1)

        else:  # "TS"
            # Temporal module per node 
            self.temporal = nn.GRU(input_size=F_in,
                                   hidden_size=temporal_hidden,
                                   num_layers=1,
                                   batch_first=True)
            # Spatial stack
            sdims = [temporal_hidden, *spatial_hidden]
            self.spatial_layers = nn.ModuleList(
                [PowerGConvDyn(sdims[i], sdims[i+1], K=K) for i in range(len(sdims)-1)]
            )
            self.head = nn.Linear(sdims[-1], 1)

    def _apply_spatial(self, X_nt: torch.Tensor) -> torch.Tensor:
        H = X_nt
        for layer in self.spatial_layers:
            H = layer(H, self.A_hat)
            H = self.act(H)
            H = self.dropout(H)
        return H  

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.squeeze(1)       # [B, N, T]
        elif x.dim() == 3:
            B, F_or_N, L = x.shape
            # Case A: flattened [B, 1, N*T]  -> reshape back to [B, N, T]
            if F_or_N == 1 and L == self.N * self.T:
                x = x.view(B, 1, self.N, self.T).squeeze(1)
            # Case B: already [B, N, T] -> keep as-is
            elif F_or_N == self.N and L == self.T:
                pass
            else:
                raise ValueError(
                    f"Unexpected 3D input shape {x.shape}. "
                    f"Expected [B,1,N*T] (flattened) or [B,N,T] with (N,T)=({self.N},{self.T})."
                )
        else:
            raise ValueError(
                f"Expected 3D or 4D input; got {x.dim()}D with shape {tuple(x.shape)}."
            )

        B, N, T = x.shape
        assert (N, T) == (self.N, self.T), f"Got {(N,T)}, expected {(self.N,self.T)}"

        if self.order == "ST":
            # Spatial per time slice -> [B, T, N, C]
            C = None
            H_sp = []
            for b in range(B):
                H_bt = []
                for t in range(T):
                    Xt = x[b, :, t].unsqueeze(1)        # [N, F_in]
                    H = self._apply_spatial(Xt)         # [N, C]
                    if C is None:
                        C = H.size(1)
                    H_bt.append(H)
                H_bt = torch.stack(H_bt, dim=0)         # [T, N, C]
                H_sp.append(H_bt)
            H_sp = torch.stack(H_sp, dim=0)             # [B, T, N, C]

            # Temporal per node
            H_seq = H_sp.permute(0, 2, 1, 3).contiguous().view(B * N, T, C)  # [B*N, T, C]
            _, h_last = self.temporal(H_seq)                                  # h_last: [1, B*N, H]
            h_last = h_last[-1]                                               # [B*N, H]
            y = self.head(h_last).view(B, N)                                  # [B, N]
            return y

        else:  # "TS"
            # Temporal per node 
            X_seq = x.permute(0, 2, 1).contiguous().view(B * N, T, self.F_in)  # [B*N, T, F_in]
            _, h_last = self.temporal(X_seq)                                   # [1, B*N, H]
            E = h_last[-1].view(B, N, -1)                                      # [B, N, H]

            # Spatial on node embeddings (one snapshot per batch)
            outs = []
            for b in range(B):
                H = self._apply_spatial(E[b])                                  # [N, C]
                y_b = self.head(H).squeeze(-1)                                 # [N]
                outs.append(y_b)
            return torch.stack(outs, dim=0)                                    # [B, N]
