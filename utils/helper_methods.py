import os
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from datetime import datetime
import torch

# --- Helper methods for sparse tensor conversion ---
def scipy_to_torch_sparse(A: sp.spmatrix, device=None) -> torch.Tensor:
    A = A.tocoo()
    idx = torch.tensor(np.vstack([A.row, A.col]), dtype=torch.long, device=device)
    val = torch.tensor(A.data, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(idx, val, torch.Size(A.shape), device=device).coalesce()

def torch_sparse_sym_norm(A: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Return D^{-1/2} A D^{-1/2} for a coalesced sparse COO tensor."""
    A = A.coalesce()
    n = A.size(0)
    deg = torch.sparse.sum(A, dim=1).to_dense().clamp_min(eps)              
    row, col = A.indices()
    scale = (deg[row] * deg[col]).sqrt()
    vals = A.values() / scale
    return torch.sparse_coo_tensor(A.indices(), vals, (n, n), device=A.device).coalesce()

def to_csr(M) -> sp.csr_matrix:
    if sp.issparse(M):
        A = M.tocsr().astype(float)
    else:
        A = sp.csr_matrix(np.asarray(M, dtype=float))
    A = (A + A.T) * 0.5
    A.setdiag(0); A.eliminate_zeros()
    return A

def torch_sparse_row_norm(A: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Row-normalize a (coalesced) torch sparse matrix: D^{-1} A (for directed time-chain representation for event-based model).
    """
    A = A.coalesce()
    row, _ = A.indices()
    deg = torch.zeros(A.size(0), device=A.device, dtype=A.dtype)
    deg.index_add_(0, row, A.values())
    inv_deg = 1.0 / torch.clamp(deg, min=eps)
    new_vals = inv_deg[row] * A.values()
    return torch.sparse_coo_tensor(A.indices(), new_vals, A.size(), device=A.device).coalesce()


# --- Helper method for dataset creation ---
def create_forecasting_dataset(graph_signals,
                               splits: list,
                               pred_horizon: int,
                               obs_window: int,
                               in_sample_mean: bool):
    
    T = graph_signals.shape[1]
    max_idx_trn = int(T * splits[0])
    max_idx_val = int(T * sum(splits[:-1]))
    split_idx = np.split(np.arange(T), [max_idx_trn , max_idx_val])

    data_dict = {}
    data_type = ['trn', 'val', 'tst']

    if in_sample_mean:
        in_sample_means = graph_signals[:,:max_idx_trn].mean(axis = 1, keepdims= True)
        data = graph_signals - in_sample_means
        data_dict["in_sample_means"] = in_sample_means
    else:
        data = graph_signals

    for i in range(3):

        split_data = data[:,split_idx[i]]
        data_points = []
        targets = []

        for j in range(len(split_idx[i])):
            try:        
                targets.append(split_data[:, list(range(j + obs_window, j + obs_window + pred_horizon))])
                data_points.append(split_data[:, list(range(j,j+obs_window))])
            except:
                break
        
        data_dict[data_type[i]] = {'data': np.stack(data_points, axis=0),
                                    'labels': np.stack(targets, axis=0)}

    print("dataset has been created.")
    print("-------------------------")
    print(f"{data_dict['trn']['data'].shape[0]} train data points")
    print(f"{data_dict['val']['data'].shape[0]} validation data points")
    print(f"{data_dict['tst']['data'].shape[0]} test data points")

    return data_dict


# --- Helper methods for product graph creation ---
def time_chain_adjacency(T: int, weight: float = 1.0):
    '''
    Create a sparse, undirected graph over time indices to be used as the temporal adjacency matrix S_T
    '''
    # Define the upper-diagonal edges in the following format:
    # (0,1),(1,2),…,(T−2,T−1)
    rows = np.arange(T-1)
    cols = rows + 1
    data = np.full(T-1, weight, dtype=float)
    A = sp.coo_matrix((data, (rows, cols)), shape=(T, T))
    A = A + A.T
    return A.tocsr()

def to_csr(M):
    '''
    Standard matrix conversion to prepare the input for the products graphs
    '''
    if sp.issparse(M):
        A = M.tocsr().astype(float)
    else:
        A = sp.csr_matrix(np.asarray(M, dtype=float))
    A = (A + A.T) * 0.5      # Symmetrize by making the matrix undirected
    A.setdiag(0)             # Drop self-loops
    A.eliminate_zeros()      
    return A

def sparsity_factor(A: sp.spmatrix, ignore_diag: bool = True) -> float:
    """Fraction of zeros in A (percentage). For undirected A, counts each edge once."""
    n = A.shape[0]
    A = A.tocsr()
    if ignore_diag:
        if A.diagonal().any():
            A = A - sp.diags(A.diagonal())
    m = A.nnz // (2 if (A != A.T).nnz == 0 else 1)   # Undirected (count only once)
    total_possible = n*(n-1)//2 if ignore_diag else n*n
    return 100.0 * (1.0 - (m/total_possible if total_possible > 0 else 0.0))

def create_product_graphs(s00, s01, s10, s11, S, num_timesteps):
    """Creates the Kronecker, Cartesian and Strong product graphs and returns them

    Parameters
    ----------
    s00: The value of hyperparameter s_00 in the equation above. 
    s01: The value of hyperparameter s_01 in the equation above. 
    s10: The value of hyperparameter s_10 in the equation above. 
    s11: The value of hyperparameter s_11 in the equation above. 
    S: Adjacency matrix of the spatial (original) graph
    num_timesteps: The number of time steps being connected by the product graph.
    """
    
    S = to_csr(S) # spatial adjacency matrix
    N = S.shape[0]
    T = int(num_timesteps)

    S_T = time_chain_adjacency(T)
    I_T = sp.eye(T, format='csr')
    I_N = sp.eye(N, format='csr')

    I_kron_I   = sp.kron(I_T, I_N, format='csr')
    I_kron_S   = sp.kron(I_T, S,   format='csr')
    ST_kron_I  = sp.kron(S_T, I_N, format='csr')
    ST_kron_S  = sp.kron(S_T, S,   format='csr')

    S_kron = s11 * ST_kron_S
    S_cart = s01 * I_kron_S + s10 * ST_kron_I
    S_strong = S_kron + S_cart
    
    return S_kron, S_cart, S_strong

def knn_graph(D, k):
    n = D.shape[0]
    
    # Symmetrize the distance matrix

    # D = 0.5 * (D + D.T)
    
    D_knn = D.copy()
    np.fill_diagonal(D_knn, np.inf) # Assign infinity to the diagonal to exclude self from nearest neighbors

    # Keep top k nearest neighbors
    k = max(1, min(k, n-1))
    cols = np.argpartition(D_knn, kth=k, axis=1)[:, :k] # Returns indices of the k nearest neighbors: (n, k) 
    rows = np.repeat(np.arange(n), k)
    data = np.ones(rows.shape[0], dtype=float)
    
    print("Number of zeros in KNN matrix:", np.sum(np.where(D == 0)))

    # Create sparse matrix
    A = sp.coo_matrix((data, (rows, cols.ravel())), shape=(n, n)).tocsr()
    A = A.maximum(A.T)                  
    A.setdiag(0)
    A.eliminate_zeros() # Remove zeroes from the sparse matrix
    print(A.shape)
    return A


# --- Helper method for plotting train and val losses ---
def plot_losses(trn_losses, val_losses, best_epoch=None, title="GTCNN training", model_name = "parametric_GTCNN", save_path=None):
    trn_losses = np.asarray(trn_losses, dtype=float)
    val_losses = np.asarray(val_losses, dtype=float)
    epochs = np.arange(1, len(trn_losses) + 1)

    plt.figure(figsize=(7,4.5))
    plt.plot(epochs, trn_losses, marker='o', label='train')
    plt.plot(epochs, val_losses, marker='s', label='val')
    if best_epoch is not None:
        # best_epoch is 0-based in the trainer; +1 to align with 1-based x-axis
        plt.axvline(best_epoch + 1, linestyle='--', linewidth=1, label=f'best (epoch {best_epoch})')
    plt.xlabel("epoch")
    plt.ylabel("loss (MSE)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    if save_path is None:
        os.makedirs("plots", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join("plots", f"{model_name}_loss_curve_{timestamp}.png")

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved loss plot to: {save_path}")
    return save_path

