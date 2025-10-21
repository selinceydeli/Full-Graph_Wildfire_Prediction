import time
import torch
import numpy as np
import scipy.sparse as sp

from model.parametric_gtcnn_event import ParametricGTCNN_Event
from model.parametric_gtcnn import ParametricGTCNN
from model.disjoint_st_baseline import DisjointSTModel
from model.vanilla_gcnn import VanillaGCN
from utils.train_utils import train_model
from utils.eval_utils import evaluate_model
from utils.helper_methods import plot_losses, create_forecasting_dataset, knn_graph

MODEL_NAMES = ["parametric_gtcnn", "disjoint_st_baseline", "vanilla_gcnn", "parametric_gtcnn_event"]
SELECTED_MODEL = MODEL_NAMES[2] # choose model here

def main():
    # Load timeline and create per-window event times (length = obs_window)
    days = np.load("data/days.npy")
    
    # sanity check
    assert np.all((days[1:] - days[:-1]) >= np.timedelta64(0, 'ns')), "days must be non-decreasing"

    # Load the dataset 
    timeseries_features = np.load(file='data/timeseries_data.npy') # shape: (N_stations, T_timestamps, F_features)
    timeseries_labels = np.load(file='data/labels.npy')   # shape: (N_stations, T_timestamps)

    # Define the parameters
    # splits = [0.6, 0.2, 0.2]
    splits = [0.1, 0.1, 0.8] # for quick testing
    pred_horizon = 1
    obs_window = 4
    n_stations = timeseries_features.shape[0]
    n_timestamps = timeseries_features.shape[1]
    n_features = timeseries_features.shape[2]

    print(f"split: {splits}, pred_horizon: {pred_horizon}, obs_window: {obs_window}")
    print(f"There are {n_stations} nodes.")
    print(f"There are {n_timestamps} measurements.")
    print(f"There are {n_features} features.")

    # shapes: timeseries_data -> (N_nodes, T_time, F_features)
    assert timeseries_features.ndim == 3, "timeseries_features must be 3D (nodes x time x features)"

    # sanity checks
    T = timeseries_features.shape[1]
    idx_trn = int(T * splits[0])
    idx_val = int(T * (splits[0] + splits[1]))
    L_trn = idx_trn
    L_val = idx_val - idx_trn
    L_tst = T - idx_val
    need = obs_window + pred_horizon
    assert all(L >= need for L in [L_trn, L_val, L_tst]), \
        f"Each split must have at least {need} time steps."

    dataset = create_forecasting_dataset(
        features=timeseries_features,
        labels=timeseries_labels,
        splits=splits,
        pred_horizon=pred_horizon,
        obs_window=obs_window,
        in_sample_mean=False,
        days=days,
        return_event_times=True   # needed for the event-based GTCNN model
    )

    # Load the distance matrix
    dist_matrix = np.load(file='data/distance_matrix.npy')
    normalized_dist = dist_matrix / np.max(dist_matrix)

    # Create the kNN graph to be used as the spatial adjacency matrix
    # k is staticly set to 4 (to capture south-north-east-west neighbors)
    k = 4 
    A = knn_graph(normalized_dist, k)
    n = A.shape[0]
    sparsity = 1 - (A.nnz / (n * n)) 
    print(f"Graph sparsity: {sparsity:.4f}") # k = 4 results in sparsity 0.9996

    N = timeseries_features.shape[0]
    density = A.nnz / (N*(N-1))
    print(f"[Graph] N={N}, edges={A.nnz//2} (undirected), density={density:.6f}")

    # Sanity checks
    assert A.shape[0] == N
    try:
        node_order = np.load("data/node_order.npy", allow_pickle=True)
        assert A.shape[0] == node_order.shape[0] == N, \
            "Mismatch among A, node_order, timeseries_features shapes."
    except FileNotFoundError:
        pass
    assert sp.isspmatrix_csr(A)
    assert A.shape == (N, N)
    A_sym_diff = (A - A.T)
    assert A_sym_diff.nnz == 0 or float(A_sym_diff.power(2).sum()) < 1e-12
    assert A.diagonal().sum() == 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare the data for training
    trn_X = torch.tensor(dataset['trn']['data'], dtype=torch.float32)   # [B,N,T]
    val_X = torch.tensor(dataset['val']['data'], dtype=torch.float32)
    tst_X = torch.tensor(dataset['tst']['data'], dtype=torch.float32)

    trn_evt = dataset['trn'].get('event_times', None)   # numpy [B, T]
    val_evt = dataset['val'].get('event_times', None)
    tst_evt = dataset['tst'].get('event_times', None)

    # trn_y = torch.tensor(dataset['trn']['labels'], dtype=torch.float32)  # [B,N]
    # val_y = torch.tensor(dataset['val']['labels'], dtype=torch.float32)
    # tst_y = torch.tensor(dataset['tst']['labels'], dtype=torch.float32)
    # print("Before:")
    # print(f"Training samples: {trn_y.shape}, Validation samples: {val_y.shape}, Test samples: {tst_y.shape}")

    trn_y = torch.tensor(dataset['trn']['labels'][:, :, 0], dtype=torch.float32)  # [B,N]
    val_y = torch.tensor(dataset['val']['labels'][:, :, 0], dtype=torch.float32)
    tst_y = torch.tensor(dataset['tst']['labels'][:, :, 0], dtype=torch.float32)
    # print("After:")
    # print(f"Training samples: {trn_y.shape}, Validation samples: {val_y.shape}, Test samples: {tst_y.shape}")

   # Define the model
    if SELECTED_MODEL == "parametric_gtcnn":
        model = ParametricGTCNN(
            S_spatial=A,
            T=obs_window,
            F_in=n_features,
            hidden_dims=(64,64),
            K=2,
            pool="mean",
            init_s=(0.0, 1.0, 1.0, 0.0),
            device=device
        ).to(device)

    elif SELECTED_MODEL == "disjoint_st_baseline":
        model = DisjointSTModel(
            S_spatial=A,
            T=obs_window,
            F_in=n_features,
            spatial_hidden=(64, 64),
            temporal_hidden=64,
            K=2,
            order="ST",
            device=device
        ).to(device)

    elif SELECTED_MODEL == "vanilla_gcnn":
        in_channels = n_features * obs_window
        model = VanillaGCN(
            S_spatial=A,
            in_channels=in_channels,
            hidden_channels=24,
            out_channels=1,
            num_layers=10,
            dropout=0.1
        ).to(device)

    elif SELECTED_MODEL == "parametric_gtcnn_event":
        # If you updated the class to take obs_window (recommended):
        model = ParametricGTCNN_Event(
            S_spatial=A,
            obs_window=obs_window,
            F_in=n_features, hidden_dims=(64,64), K=2, pool="mean",
            init_s=(0.0,1.0,1.0,0.0), kernel="exp", tau=3.0, max_back_hops=3,
            device=device
        ).to(device)

    # Model-specific reshaping
    if SELECTED_MODEL in ["parametric_gtcnn", "disjoint_st_baseline"]:
        # [B,N,T,F] -> [B,F,N,T] -> [B,F,N*T]
        trn_X = trn_X.permute(0, 3, 1, 2).flatten(2, 3)
        val_X = val_X.permute(0, 3, 1, 2).flatten(2, 3)
        tst_X = tst_X.permute(0, 3, 1, 2).flatten(2, 3)
    elif SELECTED_MODEL in ["parametric_gtcnn_event"]:
        # [B,N,T,F] -> [B,F,N,T]
        trn_X = trn_X.permute(0, 3, 1, 2)
        val_X = val_X.permute(0, 3, 1, 2)
        tst_X = tst_X.permute(0, 3, 1, 2)
    elif SELECTED_MODEL in ["vanilla_gcnn"]:
        # [B,N,T,F] -> [B,N,F*T]
        trn_X = trn_X.permute(0, 1, 3, 2).flatten(2, 3) # TODO: check if this is correct
        val_X = val_X.permute(0, 1, 3, 2).flatten(2, 3)
        tst_X = tst_X.permute(0, 1, 3, 2).flatten(2, 3)

    # Train config
    gamma = 1e-4 if SELECTED_MODEL == "parametric_gtcnn" else 0.0
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    num_epochs = 1
    batch_size = 64

    loss_criterion = torch.nn.BCEWithLogitsLoss()

    # Training loop
    training_start = time.time()
    print("Training starts with model:", SELECTED_MODEL)

    best_model, epoch_best, trn_loss_per_epoch, val_loss_per_epoch = train_model(
        model,
        model_name=SELECTED_MODEL,
        training_data=trn_X.to(device),
        validation_data=val_X.to(device),
        single_step_trn_labels=trn_y.to(device),
        single_step_val_labels=val_y.to(device),
        num_epochs=num_epochs, batch_size=batch_size,
        loss_criterion=loss_criterion,
        optimizer=optimizer, scheduler=scheduler,
        val_metric_criterion=None,
        log_dir=f"./runs/{SELECTED_MODEL}",
        not_learning_limit=15,
        gamma=gamma,
        trn_event_times=trn_evt,       # pass event times (numpy) to train
        val_event_times=val_evt        # pass event times (numpy) to val
    )
    
    training_end = time.time()
    print(f"Training took {training_end - training_start} seconds.")

    # Plot train and val loss per epoch
    plot_losses(trn_loss_per_epoch, val_loss_per_epoch, best_epoch=epoch_best,
            title=f"{SELECTED_MODEL} — train/val loss", model_name=SELECTED_MODEL, save_path=None)

    # Evaluate the best model on the test set
    metrics = evaluate_model(
        best_model, tst_X, tst_y, loss_criterion,
        apply_sigmoid=True,            # True for BCE/logits; False for regression heads
        chunk_size=256,
        event_times=tst_evt            # pass event times (numpy) to test
    )

    print("Test set metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

