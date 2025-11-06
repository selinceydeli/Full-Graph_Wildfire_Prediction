import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from functools import lru_cache

from utils.helper_methods import scipy_to_torch_sparse, torch_sparse_row_norm, to_csr
from model.polynomial_gconv import PowerGConvDyn


def temporal_kernel_adjacency(event_times,
                              max_back_hops=1,
                              kernel="exp",
                              tau=3.0,
                              epsilon=1e-8):
    """
    event_times: 1D array-like of *monotone* timestamps (floats in 'days' or datetime converted to float days).
    Returns a directed (T x T) CSR with edges from past(i) -> future(j) for i<j.
    """
    t = np.asarray(event_times, dtype=float)
    T = t.shape[0]
    rows, cols, data = [], [], []

    def k(dt):
        if dt <= 0:
            return 0.0
        if kernel == "exp":
            return float(np.exp(-dt / max(tau, epsilon)))
        elif kernel == "rational":
            return float(1.0 / (1.0 + dt / max(tau, epsilon)))
        raise ValueError("Unknown kernel")

    for j in range(T):
        i_start = max(0, j - max_back_hops)
        for i in range(i_start, j):
            w = k(t[j] - t[i])
            if w > 0:
                rows.append(j)  # past -> future
                cols.append(i)
                data.append(w)

    A = sp.coo_matrix((data, (rows, cols)), shape=(T, T)).tocsr()
    return A


class ParametricGTCNN_Event(nn.Module):
    """
    Event-based GTCNN over a product graph:
      S_♦ = s00*(I_T ⊗ I_N) + s01*(I_T ⊗ S) + s10*(S_T(evt) ⊗ I_N) + s11*(S_T(evt) ⊗ S)
    where S_T(evt) depends on the *current window's* event timestamps.

    Notes:
      - obs_window (T) is fixed for the dataset; event times vary per window.
      - We rebuild the temporal blocks (K10, K11) per sample in forward().
    """

    def __init__(self,
                 S_spatial: sp.spmatrix,  # (N x N) 
                 obs_window: int,  # fixed window length T
                 F_in: int = 1,
                 hidden_dims=(64, 64),
                 K: int = 2,
                 pool: str = "mean",  # "mean" or "last"
                 init_s=(0.0, 1.0, 1.0, 0.0),
                 kernel: str = "exp",
                 tau: float = 3.0,  # in days
                 max_back_hops: int = 3,
                 device: str | torch.device = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.pool = pool
        self.F_in = int(F_in)
        self.kernel = str(kernel)
        self.tau = float(tau)
        self.max_back_hops = int(max_back_hops)

        # Spatial graph 
        S = to_csr(S_spatial)
        N = int(S.shape[0])
        self.S_spatial_csr = S

        # Temporal window length (fixed for this dataset)
        self.T = int(obs_window)

        # Fixed Kronecker blocks (do not depend on per-window event times)
        I_T = sp.eye(self.T, format='csr')
        I_N = sp.eye(N, format='csr')

        K00 = scipy_to_torch_sparse(sp.kron(I_T, I_N, format='csr')).coalesce()
        K01 = scipy_to_torch_sparse(sp.kron(I_T, S, format='csr')).coalesce()

        self.register_buffer("K00", K00)
        self.register_buffer("K01", K01)

        # Placeholders for temporal blocks; they will be updated per sample
        # Initialize with a trivial evenly spaced window just so buffers exist
        init_evt = np.arange(self.T, dtype=float)
        K10, K11 = self._build_temporal_blocks(init_evt, N=N, S_spatial=S)
        self.register_buffer("K10", K10)
        self.register_buffer("K11", K11)

        # Learnable scalar weights
        s00, s01, s10, s11 = init_s
        self.s_00 = nn.Parameter(torch.tensor(float(s00), dtype=torch.float32, device=self.device))
        self.s_01 = nn.Parameter(torch.tensor(float(s01), dtype=torch.float32, device=self.device))
        self.s_10 = nn.Parameter(torch.tensor(float(s10), dtype=torch.float32, device=self.device))
        self.s_11 = nn.Parameter(torch.tensor(float(s11), dtype=torch.float32, device=self.device))

        # GNN stack
        dims = [self.F_in, *hidden_dims]
        self.layers = nn.ModuleList([PowerGConvDyn(dims[i], dims[i + 1], K=K) for i in range(len(dims) - 1)])
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.head = nn.Linear(dims[-1], 1)

    # ---- temporal blocks (re)builder + small cache ----
    def _build_temporal_blocks(self, event_times_np: np.ndarray, N: int = None, S_spatial: sp.spmatrix = None):
        """Build K10, K11 for given event_times (length = self.T)."""
        # enforce relative times; absolute offset irrelevant
        et = np.asarray(event_times_np, dtype=float)
        assert et.shape[0] == self.T, "event_times length must equal obs_window (T)."
        et = et - et[0]

        S_T_evt = temporal_kernel_adjacency(et,
                                            max_back_hops=self.max_back_hops,
                                            kernel=self.kernel,
                                            tau=self.tau)

        I_N = sp.eye(N, format='csr')
        S_spatial_csr = S_spatial

        K10 = scipy_to_torch_sparse(sp.kron(S_T_evt, I_N, format='csr')).coalesce().to(self.device)
        K11 = scipy_to_torch_sparse(sp.kron(S_T_evt, S_spatial_csr, format='csr')).coalesce().to(self.device)
        return K10, K11

    # TODO: if uncommented, causes bugs due to adj not being hashable
    # @lru_cache(maxsize=1024) 
    # def _cached_temporal_blocks(self, key: tuple, N: int = None, S_spatial: sp.spmatrix = None):
    #     et = np.array(key, dtype=float)
    #     return self._build_temporal_blocks(et, N=N, S_spatial=S_spatial)

    def _set_event_times_for_sample(self, event_times_np: np.ndarray, N: int = None, S_spatial: sp.spmatrix = None):
        # Cache key: use rounded relative times for stability
        et = np.asarray(event_times_np, dtype=float)
        et = et - et[0]
        key = tuple(np.round(et, 6))  # 1e-6 days ≈ 0.0864 seconds
        # K10, K11 = self._cached_temporal_blocks(key, N=N, S_spatial=S_spatial)
        et = np.array(key, dtype=float)
        K10, K11 = self._build_temporal_blocks(et, N=N, S_spatial=S_spatial)
        self.K10 = K10
        self.K11 = K11

    # ---- graph normalization + reshape helpers ----
    def _current_A_hat(self, adj_matrix):
        """ 
        Graph normalization with current temporal blocks.
        input: adj_matrix (spatial graph)
        output: A (product graph)
        """
        device = self.s_00.device
        # Spatial graph 
        S = to_csr(adj_matrix)
        N = int(S.shape[0])

        # Fixed Kronecker blocks (do not depend on per-window event times)
        I_T = sp.eye(self.T, format='csr')
        I_N = sp.eye(N, format='csr')

        K00 = scipy_to_torch_sparse(sp.kron(I_T, I_N, format='csr')).coalesce().to(device)
        K01 = scipy_to_torch_sparse(sp.kron(I_T, S, format='csr')).coalesce().to(device)

        # Placeholders for temporal blocks; they will be updated per sample
        # Initialize with a trivial evenly spaced window just so buffers exist
        init_evt = np.arange(self.T, dtype=float)
        K10, K11 = self._build_temporal_blocks(init_evt, N=N, S_spatial=S)

        # Nonnegative mix of the four Kronecker blocks
        s00 = torch.relu(self.s_00)
        s01 = torch.relu(self.s_01)
        s10 = torch.relu(self.s_10)
        s11 = torch.relu(self.s_11)

        # Row-normalization suits directed time edges
        A = (s00 * K00) + (s01 * K01) + (s10 * K10) + (s11 * K11)
        return torch_sparse_row_norm(A)

    def _reshape_to_product_nodes(self, x: torch.Tensor, N: int) -> torch.Tensor:
        """
        Accept [B,F,N,T] or [B,F,N*T]; return [B, T*N, F]
        """
        if x.dim() == 3:
            B, F, NT = x.shape
            assert F == self.F_in and NT == N * self.T
            x = x.view(B, F, N, self.T)
        elif x.dim() == 4:
            B, F, N, T = x.shape
            assert (F, N, T) == (self.F_in, N, self.T)
        else:
            raise ValueError("Expected x with 3 or 4 dims.")
        return x.permute(0, 3, 2, 1).contiguous().view(x.size(0), self.T * N, self.F_in)

    # ---- forward (now accepts per-batch event times) ----
    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor = None, event_times_batch=None) -> torch.Tensor:
        """
        x: [B, F, N, T] (preferred) or [B, F, N*T]
        event_times_batch: None or numpy array of shape [B, T] (float days, non-decreasing per row)
        """

        B = x.size(0)

        if adj_matrix is None:
            adj_matrix = self.S_spatial_csr

        # Get current A_hat with updated temporal blocks
        A_hat = self._current_A_hat(adj_matrix)
        N = A_hat.size(0) // self.T

        # Reshape input to match product graph structure
        X_prod = self._reshape_to_product_nodes(x, N=N)  # [B, T*N, F]
        outs = []

        for b in range(B):
            if event_times_batch is not None:
                self._set_event_times_for_sample(event_times_batch[b], N=N, S_spatial=adj_matrix)

            H = X_prod[b]
            for layer in self.layers:
                H = layer(H, A_hat)
                H = self.act(H)
                H = self.dropout(H)

            H = H.view(self.T, N, -1)  # [T, N, C]
            H_pool = H.mean(dim=0) if self.pool == "mean" else H[-1]
            y_hat = self.head(H_pool).squeeze(-1)  # [N]
            outs.append(y_hat)

        return torch.stack(outs, dim=0)  # [B, N]
