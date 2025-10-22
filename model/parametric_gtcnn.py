import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from utils.helper_methods import scipy_to_torch_sparse, torch_sparse_sym_norm, to_csr
from model.polynomial_gconv import PowerGConvDyn

def time_chain_adjacency(T: int, weight: float = 1.0) -> sp.csr_matrix:
    rows = np.arange(T-1); cols = rows + 1
    data = np.full(T-1, weight, dtype=float)
    A = sp.coo_matrix((data, (rows, cols)), shape=(T, T))
    A = A + A.T
    return A.tocsr()

class ParametricGTCNN(nn.Module):
    """
    Learns s_00, s_01, s_10, s_11 in:
       S_♦ = s00 * (I_T⊗I_N) + s01 * (I_T⊗S) + s10 * (S_T⊗I_N) + s11 * (S_T⊗S)
    Then applies K-hop polynomial graph convolutions on the product graph and pools over time.
    """
    def __init__(self,
                 S_spatial: sp.spmatrix,      # (N x N) spatial adjacency
                 T: int,                      # time steps (window length)
                 F_in: int = 1,
                 hidden_dims=(64, 64),
                 K: int = 2,
                 pool: str = "mean",          # pooling type: "mean" or "last"
                 init_s=(0.0, 1.0, 1.0, 0.0), # Cartesian product graph by default
                 device: str | torch.device = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.pool = pool
        self.F_in = F_in

        # Define base graphs
        S = to_csr(S_spatial)
        N = S.shape[0]
        self.T = int(T)
        S_T = time_chain_adjacency(self.T)

        I_T = sp.eye(self.T, format='csr'); I_N = sp.eye(N, format='csr')

        # Define Kronecker blocks
        K00 = scipy_to_torch_sparse(sp.kron(I_T, I_N, format='csr'))           # I ⊗ I
        K01 = scipy_to_torch_sparse(sp.kron(I_T, S, format='csr'))           # I ⊗ S
        K10 = scipy_to_torch_sparse(sp.kron(S_T, I_N, format='csr'))           # S_T ⊗ I
        K11 = scipy_to_torch_sparse(sp.kron(S_T, S, format='csr'))           # S_T ⊗ S

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

        # Define the GNN layers
        dims = [F_in, *hidden_dims]
        self.layers = nn.ModuleList([PowerGConvDyn(dims[i], dims[i+1], K=K) for i in range(len(dims)-1)])
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.head = nn.Linear(dims[-1], 1)   # to N outputs after temporal pooling

    def _current_A_hat(self) -> torch.Tensor:
        """
        Returns the current normalized product adjacency (recomputed in each forward run)
        """
        # Use relu so negative s do not create negative edges
        s00 = torch.relu(self.s_00)
        s01 = torch.relu(self.s_01)
        s10 = torch.relu(self.s_10)
        s11 = torch.relu(self.s_11)

        A = (s00 * self.K00) + (s01 * self.K01) + (s10 * self.K10) + (s11 * self.K11)
        return torch_sparse_sym_norm(A)   # D^{-1/2} A D^{-1/2}

    def _reshape_to_product_nodes(self, x: torch.Tensor, N: int) -> torch.Tensor:
        # Accept [B,F,N,T] or [B,F,N*T]; return [B, T*N, F]
        if x.dim() == 3:
            B, F, NT = x.shape
            print(x.shape)
            print(N)
            print(N * self.T)
            assert NT == N * self.T, "Input flattened dim != N*T"
            x = x.view(B, F, N, self.T)
        elif x.dim() == 4:
            B, F, N, T = x.shape
            assert (N, T) == (N, self.T)
        else:
            raise ValueError("Expected x with 3 or 4 dims: [B,F,N*T] or [B,F,N,T]")
        return x.permute(0, 3, 2, 1).contiguous().view(x.size(0), self.T * N, self.F_in)

    def _current_A_hat_subgraph(self, adj_matrix):
        # Define base graphs
        S = to_csr(adj_matrix)
        N = S.shape[0]

        S_T = time_chain_adjacency(self.T)
        I_T = sp.eye(self.T, format='csr')
        I_N = sp.eye(N, format='csr')

        # Define Kronecker blocks
        K00 = scipy_to_torch_sparse(sp.kron(I_T, I_N, format='csr'))           # I ⊗ I
        K01 = scipy_to_torch_sparse(sp.kron(I_T, S, format='csr'))           # I ⊗ S
        K10 = scipy_to_torch_sparse(sp.kron(S_T, I_N, format='csr'))           # S_T ⊗ I
        K11 = scipy_to_torch_sparse(sp.kron(S_T, S, format='csr'))           # S_T ⊗ S

        # Use relu so negative s do not create negative edges
        s00 = torch.relu(self.s_00)
        s01 = torch.relu(self.s_01)
        s10 = torch.relu(self.s_10)
        s11 = torch.relu(self.s_11)

        A_hat = (s00 * K00) + (s01 * K01) + (s10 * K10) + (s11 * K11)
        return torch_sparse_sym_norm(A_hat)  

    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor = None) -> torch.Tensor:
        B = x.size(0)

        if adj_matrix != None:
            A_hat = self._current_A_hat_subgraph(adj_matrix)
            N = adj_matrix.shape[0]
        else:
            A_hat = self._current_A_hat()
            N = A_hat.size(0) // self.T
            

        X_prod = self._reshape_to_product_nodes(x, N)         # [B, T*N, F]                     # sparse [T*N, T*N], same for all items

        outs = []
        for b in range(B):
            H = X_prod[b]                                  # [T*N, F]
            for layer in self.layers:
                H = layer(H, A_hat)
                H = self.act(H)
                H = self.dropout(H)
            # [T*N, C] -> [T,N,C] -> pool over time -> [N,C]
            H = H.view(self.T, N, -1)
            H_pool = H.mean(dim=0) if self.pool == "mean" else H[-1]
            y_hat = self.head(H_pool).squeeze(-1)          # [N]
            outs.append(y_hat)
        return torch.stack(outs, dim=0)                    # [B, N]
