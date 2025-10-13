import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from utils.helper_methods import scipy_to_torch_sparse, torch_sparse_row_norm, to_csr
from model.polynomial_gconv import PowerGConvDyn

def temporal_kernel_adjacency(event_times, 
                              max_back_hops=1,
                              kernel="exp",
                              tau=3.0,               # in the same units as event_times (in our case days)
                              epsilon=1e-8):
    """
    event_times: 1D array-like of monotonically increasing timestamps (float or datetime converted to float)
    max_back_hops: connect each event t_j to up to this many previous events (j-1, j-2, ...)
    kernel: "exp" or "rational"
    returns: CSR (T x T), directed
    """
    t = np.asarray(event_times, dtype=float)
    T = t.shape[0]
    rows, cols, data = [], [], []

    def k(dt):
        if dt <= 0: return 0.0
        if kernel == "exp":
            return float(np.exp(-dt / max(tau, epsilon)))
        elif kernel == "rational":
            return float(1.0 / (1.0 + dt / max(tau, epsilon)))
        else:
            raise ValueError("Unknown kernel")

    for j in range(T):
        i_start = max(0, j - max_back_hops)
        for i in range(i_start, j):
            w = k(t[j] - t[i])
            if w > 0:
                rows.append(j)   # from i to j (past->future)
                cols.append(i)
                data.append(w)

    A = sp.coo_matrix((data, (rows, cols)), shape=(T, T)).tocsr()
    return A


class ParametricGTCNN_Event(nn.Module):
    def __init__(self,
                 S_spatial: sp.spmatrix,      # (N x N)
                 event_times,                 # list/np.array of event timestamps
                 F_in: int = 1,
                 hidden_dims=(64, 64),
                 K: int = 2,
                 pool: str = "mean",
                 init_s=(0.0, 1.0, 1.0, 0.0),
                 kernel="exp", tau=3.0, max_back_hops=3,
                 device: str | torch.device = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.pool = pool
        self.F_in = F_in

        # Base graphs
        S = to_csr(S_spatial)
        N = S.shape[0]; self.N = N

        # Build event-based temporal graph
        self.event_times = np.asarray(event_times, dtype=float)
        self.T = int(len(self.event_times)) # T: number of events

        S_T_evt = temporal_kernel_adjacency(self.event_times,
                                            max_back_hops=max_back_hops,
                                            kernel=kernel, tau=tau)

        I_T = sp.eye(self.T, format='csr'); I_N = sp.eye(N, format='csr')

        # Kronecker blocks (keep shapes: (T*N) x (T*N))
        K00 = scipy_to_torch_sparse(sp.kron(I_T, I_N, format='csr'))
        K01 = scipy_to_torch_sparse(sp.kron(I_T, S, format='csr'))
        K10 = scipy_to_torch_sparse(sp.kron(S_T_evt, I_N, format='csr'))
        K11 = scipy_to_torch_sparse(sp.kron(S_T_evt, S, format='csr'))

        self.register_buffer("K00", K00.coalesce())
        self.register_buffer("K01", K01.coalesce())
        self.register_buffer("K10", K10.coalesce())
        self.register_buffer("K11", K11.coalesce())

        # Learnable scalars (names start with s_ for L1 reg) for making the product graph parametric
        s00, s01, s10, s11 = init_s
        self.s_00 = nn.Parameter(torch.tensor(float(s00), dtype=torch.float32, device=self.device))
        self.s_01 = nn.Parameter(torch.tensor(float(s01), dtype=torch.float32, device=self.device))
        self.s_10 = nn.Parameter(torch.tensor(float(s10), dtype=torch.float32, device=self.device))
        self.s_11 = nn.Parameter(torch.tensor(float(s11), dtype=torch.float32, device=self.device))

        # GNN layers
        dims = [F_in, *hidden_dims]
        self.layers = nn.ModuleList([PowerGConvDyn(dims[i], dims[i+1], K=K) for i in range(len(dims)-1)])
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.head = nn.Linear(dims[-1], 1)

    def _current_A_hat(self) -> torch.Tensor:
        # ReLU to keep nonnegative edges
        s00 = torch.relu(self.s_00)
        s01 = torch.relu(self.s_01)
        s10 = torch.relu(self.s_10)
        s11 = torch.relu(self.s_11)

        A = (s00 * self.K00) + (s01 * self.K01) + (s10 * self.K10) + (s11 * self.K11)
        # Use row-normalization for directed temporal part
        return torch_sparse_row_norm(A)

    def _reshape_to_product_nodes(self, x: torch.Tensor) -> torch.Tensor:
        # Accept [B,F,N,T] or [B,F,N*T]; return [B, T*N, F]
        if x.dim() == 3:
            B, F, NT = x.shape
            assert NT == self.N * self.T
            x = x.view(B, F, self.N, self.T)
        elif x.dim() == 4:
            B, F, N, T = x.shape
            assert (N, T) == (self.N, self.T)
        else:
            raise ValueError("Expected x with 3 or 4 dims.")
        return x.permute(0, 3, 2, 1).contiguous().view(x.size(0), self.T * self.N, self.F_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        X_prod = self._reshape_to_product_nodes(x)   # [B, T*N, F]
        A_hat = self._current_A_hat()                # directed RW-norm

        outs = []
        for b in range(B):
            H = X_prod[b]
            for layer in self.layers:
                H = layer(H, A_hat)
                H = self.act(H); H = self.dropout(H)
            H = H.view(self.T, self.N, -1)
            H_pool = H.mean(dim=0) if self.pool == "mean" else H[-1]  
            y_hat = self.head(H_pool).squeeze(-1)
            outs.append(y_hat)
        return torch.stack(outs, dim=0)              # [B, N]

