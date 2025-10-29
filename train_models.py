import time
import torch
import numpy as np
import scipy.sparse as sp
import argparse
from typing import List
from model.parametric_gtcnn_event import ParametricGTCNN_Event
from model.parametric_gtcnn import ParametricGTCNN
from model.simple_gc import SimpleGraphConvolution
# from model.disjoint_st_baseline import DisjointSTModel
from model.vanilla_gcnn import VanillaGCN
from utils.train_utils import train_model, train_model_clustering
from utils.eval_utils import evaluate_model
from losses.focal_loss import FocalLoss
from losses.dice_loss import DiceLoss
from utils.helper_methods import plot_losses, create_forecasting_dataset, knn_graph, impute_nan_with_feature_mean
from torcheval.metrics import BinaryF1Score


def parse_args():
    parser = argparse.ArgumentParser(prog='Training Loop for Wildfire Graph Machine Learning Project')
    parser.add_argument('--days_data_path', type=str,
                        help="The path to the timeline to create per-window event images.", default="data/days.npy")
    parser.add_argument('--timeseries_data_path', type=str, help="The filepath to our timeseries",
                        default='data/timeseries_data.npy')
    parser.add_argument('--distance_matrix_filepath', type=str, default='data/distance_matrix.npy')
    parser.add_argument('--labels_path', type=str, help="The filepath to te labels for our timeseries.",
                        default='data/labels.npy')
    parser.add_argument('--pred_horizon', type=int, help='How many timesteps we try to predict ahead.', default=1)
    parser.add_argument('--obs_window', type=int, help='How many timesteps are used in our convolutions.', default=4)
    parser.add_argument('--k', type=int, help="How many neighbors we keep in our adjacency matrix.", default=4)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--selected_loss_function', choices=["bce", "weighted_bce", "focal", "dice"], type=str,
                        default="weighted_bce")
    parser.add_argument('--selected_model', type=str,
                        choices=["parametric_gtcnn", "vanilla_gcnn", "parametric_gtcnn_event", "simple_gc"],
                        default="simple_gc")
    parser.add_argument('--train_val_test_split', nargs=3, type=float, default=[0.6, 0.2, 0.2])
    parser.add_argument('--threshold_tp', help="Threshold for the confidence needed to be a true positive.", type=float,
                        default=0.5)
    parser.add_argument('--clustering', help="Boolean for using clustering or not.", type=bool, default=False)
    return parser


def main(days_data_path: str, timeseries_data_path: str, labels_path: str, distance_matrix_filepath: str,
         pred_horizon: int, obs_window: int, k: int, num_epochs: int, batch_size: int, selected_loss_function: str,
         selected_model: str, train_val_test_split: List[int], threshold_tp: float, clustering: bool):
    # Load the days
    days = np.load(days_data_path)

    # Load the dataset 
    timeseries_features = np.load(file=timeseries_data_path)  # shape: (N_stations, T_timestamps, F_features)
    timeseries_labels = np.load(file=labels_path)  # shape: (N_stations, T_timestamps)

    # Load the distance matrix
    dist_matrix = np.load(file=distance_matrix_filepath)
    normalized_dist = dist_matrix / np.max(dist_matrix)

    # sanity check
    assert np.all((days[1:] - days[:-1]) >= np.timedelta64(0, 'ns')), "days must be non-decreasing"

    # Define the parameters
    splits = train_val_test_split
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
        in_sample_mean=True,
        days=days,
        return_event_times=True  # needed for the event-based GTCNN model
    )

    # Create the kNN graph to be used as the spatial adjacency matrix
    # k is staticly set to 4 (to capture south-north-east-west neighbors)
    A = knn_graph(normalized_dist, k)
    n = A.shape[0]
    sparsity = 1 - (A.nnz / (n * n))
    print(f"Graph sparsity: {sparsity:.4f}")  # k = 4 results in sparsity 0.9996

    N = timeseries_features.shape[0]
    density = A.nnz / (N * (N - 1))
    print(f"[Graph] N={N}, edges={A.nnz // 2} (undirected), density={density:.6f}")

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
    trn_X = torch.tensor(dataset['trn']['data'], dtype=torch.float32)  # [B,N,T,F]
    val_X = torch.tensor(dataset['val']['data'], dtype=torch.float32)
    tst_X = torch.tensor(dataset['tst']['data'], dtype=torch.float32)

    trn_evt = dataset['trn'].get('event_times', None)  # numpy [B,T]
    val_evt = dataset['val'].get('event_times', None)
    tst_evt = dataset['tst'].get('event_times', None)

    trn_y = torch.tensor(dataset['trn']['labels'][:, :, 0], dtype=torch.float32)  # [B,N,pred_horizon] -> [B,N]
    val_y = torch.tensor(dataset['val']['labels'][:, :, 0], dtype=torch.float32)
    tst_y = torch.tensor(dataset['tst']['labels'][:, :, 0], dtype=torch.float32)

    # Check for NaNs and handle them (mean imputation here)
    trn_X = impute_nan_with_feature_mean(trn_X, show_nan_info=True)
    val_X = impute_nan_with_feature_mean(val_X, show_nan_info=True)
    tst_X = impute_nan_with_feature_mean(tst_X, show_nan_info=True)

    # Define the model
    if selected_model == "vanilla_gcnn":
        in_channels = n_features * obs_window
        model = VanillaGCN(
            S_spatial=A,
            in_channels=in_channels,
            hidden_channels=24,
            out_channels=1,
            num_layers=10,
            dropout=0.1
        ).to(device)

    elif selected_model == "parametric_gtcnn":
        model = ParametricGTCNN(
            S_spatial=A,
            T=obs_window,
            F_in=n_features,
            hidden_dims=(64, 64),
            K=2,
            pool="mean",
            init_s=(0.0, 1.0, 1.0, 0.0),
            device=device
        ).to(device)

    elif selected_model == "parametric_gtcnn_event":
        # If you updated the class to take obs_window (recommended):
        model = ParametricGTCNN_Event(
            S_spatial=A,
            obs_window=obs_window,
            F_in=n_features, hidden_dims=(64, 64), K=2, pool="mean",
            init_s=(0.0, 1.0, 1.0, 0.0), kernel="exp", tau=3.0, max_back_hops=3,
            device=device
        ).to(device)

    elif selected_model == "simple_gc":
        in_channels = n_features * obs_window
        model = SimpleGraphConvolution(
            in_channels=in_channels,
            out_channels=1,
        ).to(device)

    # elif SELECTED_MODEL == "disjoint_st_baseline":
    #     model = DisjointSTModel(
    #         S_spatial=A,
    #         T=obs_window,
    #         F_in=n_features,
    #         spatial_hidden=(64, 64),
    #         temporal_hidden=64,
    #         K=2,
    #         order="ST",
    #         device=device
    #     ).to(device)

    # Model-specific reshaping
    if selected_model in ["vanilla_gcnn", "simple_gc"]:
        # [B,N,T,F] -> [B,N,F*T]
        trn_X = trn_X.permute(0, 1, 3, 2).flatten(2, 3)
        val_X = val_X.permute(0, 1, 3, 2).flatten(2, 3)
        tst_X = tst_X.permute(0, 1, 3, 2).flatten(2, 3)
    elif selected_model in ["parametric_gtcnn"]:
        # [B,N,T,F] -> [B,F,N,T]
        trn_X = trn_X.permute(0, 3, 1, 2)
        val_X = val_X.permute(0, 3, 1, 2).flatten(2, 3)
        tst_X = tst_X.permute(0, 3, 1, 2).flatten(2, 3)
    elif selected_model in ["parametric_gtcnn_event"]:
        # [B,N,T,F] -> [B,F,N,T]
        trn_X = trn_X.permute(0, 3, 1, 2)
        val_X = val_X.permute(0, 3, 1, 2)
        tst_X = tst_X.permute(0, 3, 1, 2)
    # elif SELECTED_MODEL in ["disjoint_st_baseline"]:
    #     # [B,N,T,F] -> [B,F,N,T] -> [B,F,N*T]
    #     trn_X = trn_X.permute(0, 3, 1, 2).flatten(2, 3)
    #     val_X = val_X.permute(0, 3, 1, 2).flatten(2, 3)
    #     tst_X = tst_X.permute(0, 3, 1, 2).flatten(2, 3)

    # Train config
    gamma = 1e-4 if selected_model == "parametric_gtcnn" else 0.0  # TODO: should it be non-zero for event-based too?
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    not_learning_limit = 5
    num_clusters = 50  # only used if CLUSTERING=True

    if selected_loss_function == "bce":
        loss_criterion = torch.nn.BCEWithLogitsLoss()
    elif selected_loss_function == "weighted_bce":
        with torch.no_grad():
            pos = trn_y.sum().item()
            total = trn_y.numel()
            neg = total - pos
        if pos <= 0:
            pos_weight_value = 1.0
        else:
            pos_weight_value = neg / pos
        pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)
        pos_percentage = 100 * (pos / total)
        print(f"[Imbalance] percentage of train positives={pos_percentage:.2f}%, pos_weight={pos_weight_value:.2f}")
        loss_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif selected_loss_function == "focal":
        loss_criterion = FocalLoss()
    elif selected_loss_function == "dice":
        loss_criterion = DiceLoss()
    else:
        raise NotImplementedError("No other loss functions")

    # Training loop
    training_start = time.time()
    print("Training starts with model:", selected_model)

    # Small batch training with clustering
    if clustering:
        if selected_model not in ["parametric_gtcnn", "parametric_gtcnn_event"]:
            print("Clustering should only be used with parametric_gtcnn model (event based or normal).")
            return None

        print("Small-batch training with clustering...")
        best_model, epoch_best, trn_loss_per_epoch, val_loss_per_epoch = train_model_clustering(
            model=model,
            model_name=selected_model,
            S_spatial=A,
            training_data=trn_X.to(device),
            validation_data=val_X.to(device),
            single_step_trn_labels=trn_y.to(device),
            single_step_val_labels=val_y.to(device),
            num_epochs=num_epochs,
            batch_size=batch_size,
            num_clusters=num_clusters,
            loss_criterion=loss_criterion,
            optimizer=optimizer, scheduler=scheduler,
            val_metric_criterion=BinaryF1Score(threshold=threshold_tp),
            log_dir=f"./runs/{selected_model}",
            not_learning_limit=not_learning_limit,
            gamma=gamma,
            trn_event_times=trn_evt,  # pass event times (numpy) to train
            val_event_times=val_evt  # pass event times (numpy) to val
        )

    # Full-batch training
    else:
        print("Full-batch training...")
        best_model, epoch_best, trn_loss_per_epoch, val_loss_per_epoch = train_model(
            model,
            model_name=selected_model,
            training_data=trn_X.to(device),
            validation_data=val_X.to(device),
            single_step_trn_labels=trn_y.to(device),
            single_step_val_labels=val_y.to(device),
            batch_size=batch_size,
            num_epochs=num_epochs,
            loss_criterion=loss_criterion,
            optimizer=optimizer, scheduler=scheduler,
            val_metric_criterion=BinaryF1Score(threshold=threshold_tp),
            log_dir=f"./runs/{selected_model}",
            not_learning_limit=15,
            gamma=gamma,
            trn_event_times=trn_evt,  # pass event times (numpy) to train
            val_event_times=val_evt  # pass event times (numpy) to val
        )

    # TODO: save the best model?

    training_end = time.time()
    print(f"Training took {training_end - training_start} seconds.")

    # Plot train and val loss per epoch
    plot_losses(trn_loss_per_epoch, val_loss_per_epoch, best_epoch=epoch_best,
                title=f"{selected_model} — {selected_loss_function} - train/val loss", model_name=selected_model, loss_name=selected_loss_function, save_path=None)

    # Evaluate the best model on the test set
    metrics = evaluate_model(
        best_model, tst_X, tst_y, loss_criterion,
        apply_sigmoid=True,  # True for BCE/logits; False for regression heads
        chunk_size=256,
        event_times=tst_evt  # pass event times (numpy) to test
    )

    print("Test set metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()
    days_data_path = args.days_data_path
    timeseries_data_path = args.timeseries_data_path
    labels_path = args.labels_path
    distance_matrix_filepath = args.distance_matrix_filepath
    pred_horizon = args.pred_horizon
    obs_window = args.obs_window
    k = args.k
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    selected_loss_function = args.selected_loss_function
    selected_model = args.selected_model
    train_val_test_split = args.train_val_test_split
    threshold_tp = args.threshold_tp
    clustering = args.clustering
    main(days_data_path, timeseries_data_path, labels_path, distance_matrix_filepath, pred_horizon, obs_window, k,
         num_epochs, batch_size, selected_loss_function, selected_model, train_val_test_split, threshold_tp, clustering)
