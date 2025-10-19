import time
import torch
import numpy as np
import scipy.sparse as sp

from model.parametric_gtcnn import ParametricGTCNN
from model.disjoint_st_baseline import DisjointSTModel
from model.simple_gc import SimpleGraphConvolution
from model.vanilla_gcnn import VanillaGCN
from utils.train_utils import train_model
from utils.eval_utils import evaluate_model
from utils.helper_methods import plot_losses, create_forecasting_dataset, knn_graph

MODEL_NAMES = ["parametric_gtcnn", "disjoint_st_baseline", "vanilla_gcnn", "simple_gc"]
SELECTED_MODEL = MODEL_NAMES[3] # choose model here

def main():
    # Load the dataset 
    timeseries_data = np.load(file='data/timeseries_data.npy')

    # Define the parameters
    splits = [0.6, 0.2, 0.2]
    # splits = [0.1, 0.1, 0.8] # for quick testing
    pred_horizon = 1
    obs_window = 4
    n_stations = timeseries_data.shape[0]
    n_timestamps = timeseries_data.shape[1]

    print(f"split: {splits}, pred_horizon: {pred_horizon}, obs_window: {obs_window}")
    print(f"The dataset contains {n_timestamps} measurements over {n_stations} stations.")

    # shapes: timeseries_data -> (N_nodes, T_time)
    assert timeseries_data.ndim == 2, "timeseries_data must be 2D (nodes x time)"

    # sanity checks
    T = timeseries_data.shape[1]
    idx_trn = int(T * splits[0])
    idx_val = int(T * (splits[0] + splits[1]))
    L_trn = idx_trn
    L_val = idx_val - idx_trn
    L_tst = T - idx_val
    need = obs_window + pred_horizon
    assert all(L >= need for L in [L_trn, L_val, L_tst]), \
        f"Each split must have at least {need} time steps."

    dataset = create_forecasting_dataset(
        timeseries_data,
        splits=splits,
        pred_horizon=pred_horizon,
        obs_window=obs_window,
        in_sample_mean=False
    )

    # Load the distance matrix
    dist_matrix = np.load(file='data/dist_matrix.npy')
    normalized_dist = dist_matrix / np.max(dist_matrix)

    # Create the kNN graph to be used as the spatial adjacency matrix
    # k is staticly set to 4 (to capture south-north-east-west neighbors)
    k = 4 
    A = knn_graph(normalized_dist, k)
    n = A.shape[0]
    sparsity = 1 - (A.nnz / (n * n)) 
    print(f"Graph sparsity: {sparsity:.4f}") # k = 4 results in sparsity 0.9996

    N = timeseries_data.shape[0]
    density = A.nnz / (N*(N-1))
    print(f"[Graph] N={N}, edges={A.nnz//2} (undirected), density={density:.6f}")

    # Sanity checks:
    # 1. verify node order matches our time series rows
    assert A.shape[0] == timeseries_data.shape[0]

    # 2. verify node order matches our saved node order
    try:
        node_order = np.load("data/node_order.npy", allow_pickle=True)
        assert A.shape[0] == node_order.shape[0] == timeseries_data.shape[0], \
            "Mismatch among A, node_order, timeseries_data shapes."
    except FileNotFoundError:
        # ok if you didn’t save it; shapes still must match
        assert A.shape[0] == timeseries_data.shape[0], "A rows must equal timeseries nodes."

    # 3. verify sparsity
    assert sp.isspmatrix_csr(A), "Adjacency must be CSR (convert with .tocsr())."
    assert A.shape == (N, N), "Adjacency must be square NxN."

    # 4. verify zero diagonals
    A_sym_diff = (A - A.T)
    assert A_sym_diff.nnz == 0 or float(A_sym_diff.power(2).sum()) < 1e-12, "Adjacency must be symmetric."
    assert A.diagonal().sum() == 0, "Adjacency should have zero diagonal (no self-loops here)."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the model
    if SELECTED_MODEL == "parametric_gtcnn":
        model = ParametricGTCNN(
            S_spatial=A, 
            T=obs_window, 
            F_in=1,
            hidden_dims=(64,64), 
            K=2, 
            pool="mean",
            init_s=(0.0, 1.0, 1.0, 0.0),      # start as Cartesian; it can learn toward Strong/Kronecker
            device=device
        ).to(device)

    elif SELECTED_MODEL == "disjoint_st_baseline":
        model = DisjointSTModel(
            S_spatial=A,
            T=obs_window,
            F_in=1,
            spatial_hidden=(64, 64),
            temporal_hidden=64,
            K=2,
            order="ST",          # or "TS" to flip the processing order
            device=device
        ).to(device)
    
    elif SELECTED_MODEL == "vanilla_gcnn":
        model = VanillaGCN(
            S_spatial=A,
            in_channels=4,
            hidden_channels=24,
            out_channels=1,
            num_layers=10,
            dropout=0.1,
            device=device
        ).to(device)

    elif SELECTED_MODEL == "simple_gc":
        model = SimpleGraphConvolution(
            in_channels=4,
            out_channels=1,
        ).to(device)

    # Prepare the data for training (model-specific reshaping is done in train_model)
    trn_X = torch.tensor(dataset['trn']['data'], dtype=torch.float32)               # [B,N,T]
    val_X = torch.tensor(dataset['val']['data'], dtype=torch.float32)
    tst_X = torch.tensor(dataset['tst']['data'], dtype=torch.float32)
    trn_y = torch.tensor(dataset['trn']['labels'][:, :, 0], dtype=torch.float32)    # [B,N]
    val_y = torch.tensor(dataset['val']['labels'][:, :, 0], dtype=torch.float32)
    tst_y = torch.tensor(dataset['tst']['labels'][:, :, 0], dtype=torch.float32) 

    # Train with L1 on s_* (γ>0), using your revised loop
    gamma = 1e-4 if SELECTED_MODEL == "parametric_gtcnn" else 0.0
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Configure training parameters
    num_epochs = 5
    batch_size = 16 # we do not want too large batches
    # loss_criterion = torch.nn.MSELoss()
    loss_criterion = torch.nn.BCEWithLogitsLoss()

    training_start = time.time()
    # Training loop
    best_model, epoch_best, trn_loss_per_epoch, val_loss_per_epoch = train_model(
        model,
        model_name = SELECTED_MODEL,
        training_data=trn_X.to(device),
        validation_data=val_X.to(device),
        single_step_trn_labels=trn_y.to(device),
        single_step_val_labels=val_y.to(device),
        num_epochs=num_epochs, batch_size=batch_size,
        loss_criterion=loss_criterion,
        optimizer=optimizer, scheduler=scheduler,
        val_metric_criterion=None,
        log_dir = f"./runs/{SELECTED_MODEL}",
        not_learning_limit=15,
        gamma=gamma 
    )

    training_end = time.time()
    print(f"Training took {training_end - training_start} seconds.")

    # Plot train and val loss per epoch
    plot_losses(trn_loss_per_epoch, val_loss_per_epoch, best_epoch=epoch_best,
            title=f"{SELECTED_MODEL} — train/val loss", model_name=SELECTED_MODEL, save_path=None)
    
    # Model-specific reshaping of test data
    if SELECTED_MODEL in ["parametric_gtcnn", "disjoint_st_baseline"]:
        tst_X = tst_X.unsqueeze(1).flatten(2, 3)
    # else: for vanilla gcnn, no reshaping needed

    # Evaluate the best model on the test set
    metrics = evaluate_model(best_model, tst_X, tst_y, loss_criterion)

    print("Test set metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

