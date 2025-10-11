import torch
import numpy as np

from model.parametric_gtcnn import ParametricGTCNN
from utils.train_utils import train_parametric_gtcnn
from utils.helper_methods import plot_losses, create_forecasting_dataset, knn_graph

def main():
    # Load the dataset 
    timeseries_data = np.load(file='lab2_NOAA_dataset/NOA_109_data.npy')

    # Define the parameters
    splits = [0.3, 0.2, 0.5] # TODO: try [0.6, 0.2, 0.2] split as well (common for machine learning) - I am curious about its performance on graphs
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
    dist_matrix = np.load(file='lab2_NOAA_dataset/NOA_109_original_adj.npy')
    normalized_dist = dist_matrix / np.max(dist_matrix)

    # Create the kNN graph to be used as the spatial adjacency matrix
    n = normalized_dist.shape[0]
    sparsity = 0.9
    density = 1 - sparsity
    k = max(1, int(density * (n - 1)))        # ~90% sparsity -> ~10% density -> edges formed between ~10% of n-1 neighbors
    A = knn_graph(normalized_dist, k)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the model
    param_GTCNN_model = ParametricGTCNN(
        S_spatial=A, 
        T=obs_window, 
        F_in=1,
        hidden_dims=(64,64), 
        K=2, 
        pool="mean",
        init_s=(0.0, 1.0, 1.0, 0.0),      # start as Cartesian; it can learn toward Strong/Kronecker
        device=device
    ).to(device)

    # Prepare data to shapes the trainer expects
    trn_X = torch.tensor(dataset['trn']['data'], dtype=torch.float32).unsqueeze(1)  # [B,1,N,T]
    val_X = torch.tensor(dataset['val']['data'], dtype=torch.float32).unsqueeze(1)
    trn_y = torch.tensor(dataset['trn']['labels'][:, :, 0], dtype=torch.float32)   # [B,N]
    val_y = torch.tensor(dataset['val']['labels'][:, :, 0], dtype=torch.float32)

    # Train with L1 on s_* (γ>0), using your revised loop
    gamma = 1e-4
    optimizer = torch.optim.Adam(param_GTCNN_model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Configure training parameters
    num_epochs = 5
    batch_size = 64

    # Training loop
    best_model, epoch_best, trn_loss_per_epoch, val_loss_per_epoch = train_parametric_gtcnn(
        param_GTCNN_model,
        training_data=trn_X.to(device),
        validation_data=val_X.to(device),
        single_step_trn_labels=trn_y.to(device),
        single_step_val_labels=val_y.to(device),
        num_epochs=num_epochs, batch_size=batch_size,
        loss_criterion=torch.nn.MSELoss(),
        optimizer=optimizer, scheduler=scheduler,
        val_metric_criterion=None,
        log_dir="./gtcnn_param",
        not_learning_limit=15,
        gamma=gamma 
    )

    # Plot train and val loss per epoch
    plot_losses(trn_loss_per_epoch, val_loss_per_epoch, best_epoch=epoch_best,
            title="Parametric GTCNN — train/val loss", model_name="parametric_GTCNN", save_path=None)


if __name__ == "__main__":
    main()

