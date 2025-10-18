import time
import torch
import numpy as np
import random
from tensorboardX import SummaryWriter
from sklearn.cluster import SpectralClustering

from typing import Optional
from tqdm import tqdm 
from torcheval.metrics import BinaryF1Score

# Helper methods for training the parametric GTCNN 
def _l1_over_s_params(model: torch.nn.Module) -> torch.Tensor:
    reg = None
    for name, p in model.named_parameters():
        if p.requires_grad and name.startswith("s_"):
            reg = (p.abs().sum() if reg is None else reg + p.abs().sum())
    if reg is None:
        # If there are no learnable s_* params, return 0
        first_param = next(model.parameters(), None)
        dev = first_param.device if first_param is not None else "cpu"
        return torch.zeros((), device=dev)
    return reg


def perform_chunk_predictions(model: torch.nn.Module,
                              data: torch.Tensor,
                              chunk_size: int = 300,
                              event_times: Optional[np.ndarray] = None) -> torch.Tensor:
    """
    Runs forward passes in chunks to obtain predictions for the whole set.
    If `event_times` (numpy [B, T]) is provided, it will be sliced per chunk and
    passed to event-based models via `event_times_batch=...`.
    Returns a tensor of shape [B, N].
    """
    model.eval()
    device = next(model.parameters()).device
    n_samples = data.shape[0]
    preds = []

    with torch.no_grad():
        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            batch_x = data[start:end].to(device, non_blocking=True)

            evt_batch = None
            if event_times is not None:
                evt_batch = event_times[start:end]  # keep numpy

            if "event" in model.__class__.__name__.lower():
                batch_pred = model(batch_x, event_times_batch=evt_batch)  # [b, N]
            else:
                batch_pred = model(batch_x)                                # [b, N]

            preds.append(batch_pred.detach().cpu())

    return torch.cat(preds, dim=0) if preds else torch.empty(0)


def compute_loss_in_chunks(model: torch.nn.Module,
                           data: torch.Tensor,
                           labels: torch.Tensor,
                           criterion,
                           chunk_size: int = 300,
                           event_times: np.ndarray | None = None) -> float:
    """
    Computes validation/test loss in chunks to save memory.
    If `event_times` is provided (numpy array shaped [B, T]), it will be sliced
    per chunk and passed to event-based models via `event_times_batch=...`.
    Returns a float (rounded to 3 decimals) for logging/scheduling.
    """
    model.eval()
    device = next(model.parameters()).device
    n_samples = data.shape[0]
    losses = []

    with torch.no_grad():
        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)

            batch_x = data[start:end].to(device, non_blocking=True)
            batch_y = labels[start:end].to(device, non_blocking=True)

            # Slice per-chunk event times (keep as numpy for the model helper)
            evt_batch = None
            if event_times is not None:
                evt_batch = event_times[start:end]

            # Call the right forward signature
            if "event" in model.__class__.__name__.lower():
                pred = model(batch_x, event_times_batch=evt_batch)   # -> [batch, N]
            else:
                pred = model(batch_x)                                # -> [batch, N]
            
            if type(criterion) == type(BinaryF1Score()):
                criterion.update(pred.view(-1), batch_y.view(-1))
                
            else:
                loss = criterion(pred, batch_y)
                losses.append(float(loss.item()))   
            
            # Debug block
            # print("Loss computations:")
            # print(pred)
            # print(batch_y)
            # print(f"Loss: {loss.item()}")
            # print()

    val_loss = float(np.mean(losses)) if losses else 0.0

    if type(criterion) == type(BinaryF1Score()):
        val_loss = criterion.compute()
 
    return val_loss


# Main training loop 
def train_model(model, model_name, training_data, validation_data, single_step_trn_labels, single_step_val_labels,
                           num_epochs, batch_size,
                           loss_criterion, optimizer, scheduler,
                           val_metric_criterion,
                           log_dir, not_learning_limit,
                           gamma: float = 0.0,   # defines L1 weight on s_*
                           trn_event_times=None,
                           val_event_times=None
):
    """
    If gamma>0, trains with: J = MSE + gamma * ||s||_1, where s are model params named 's_*'.
    Validation uses plain MSE for fair model selection.
    """
    start = time.time()
    tensorboard = SummaryWriter(log_dir=log_dir)
    trn_loss_per_epoch, val_loss_per_epoch = [], []

    n_trn_samples = training_data.size()[0]
    n_batches_per_epoch = int(n_trn_samples / batch_size)

    best_val_loss = 10e10
    print(f"{n_batches_per_epoch} batches per epoch "
        f"({n_trn_samples} trn samples in total | batch_size: {batch_size})")

    not_learning_count = 0
    for epoch in tqdm(range(num_epochs)):

        # permutation = torch.randperm(n_trn_samples)  # shuffle the training data
        batch_losses = []

        model.train()
        for batch_idx in tqdm(range(0, n_trn_samples, batch_size // 2)):
            batch_indices = np.arange(batch_idx, min(n_trn_samples - 1, batch_idx + batch_size), 1)
            # batch_indices = permutation[batch_idx:batch_idx + batch_size]
            # print("Batch Index:" batch_idx, training_data.shape)
            batch_trn_data = training_data[batch_indices]
            batch_one_step_trn_labels = single_step_trn_labels[batch_indices]

            evt_batch = None
            if trn_event_times is not None:
                # trn_event_times is numpy [B, T]
                evt_batch = trn_event_times[batch_indices]

            if "event" in model.__class__.__name__.lower():
                one_step_pred_trn = model(batch_trn_data, event_times_batch=evt_batch)
            else:
                # For SimpleGTCNN, it uses the same data for all time steps
                one_step_pred_trn = model(batch_trn_data)

            # Loss: base MSE + gamma * ||s||_1 (if any s_* exist)
            base = loss_criterion(one_step_pred_trn, batch_one_step_trn_labels)
            reg  = gamma * _l1_over_s_params(model)
            batch_trn_loss = base + reg
            batch_losses.append(batch_trn_loss.item())

            optimizer.zero_grad()
            batch_trn_loss.backward()
            optimizer.step()

        epoch_trn_loss = float(np.mean(batch_losses))
        trn_loss_per_epoch.append(epoch_trn_loss)
        tensorboard.add_scalar('train-loss', epoch_trn_loss, epoch)

        # Validation: MSE only (no L1) for fair comparison/early stopping
        def _val_mse(crit):
            return compute_loss_in_chunks(model, validation_data, single_step_val_labels, crit,
                                          event_times=val_event_times,  
                                          chunk_size=batch_size)        

        val_loss = _val_mse(loss_criterion)
        val_loss_per_epoch.append(val_loss)
        tensorboard.add_scalar('val-loss', val_loss, epoch)

        # Metric for scheduling/early stopping (use MSE unless a separate metric is supplied)
        val_metric = _val_mse(val_metric_criterion) if val_metric_criterion else val_loss
        tensorboard.add_scalar('val-metric', val_metric, epoch)

        if scheduler:
            scheduler.step(val_loss)

        # Log diff, lr, and current L1 on s_*
        diff_loss = abs(epoch_trn_loss - val_loss)
        tensorboard.add_scalar('diff-loss', diff_loss, epoch)
        tensorboard.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        # Log s_ij values (already in your code)
        names = list(dict(model.named_parameters()).keys())
        s_parameters_names = [name for name in names if str(name).startswith("s_")]
        for name in s_parameters_names:
            tensorboard.add_scalar(
                name.replace(".", "/").replace("GFL/", ""),
                round(dict(model.named_parameters())[name].item(), 3),
                epoch
            )
        # Also log L1 magnitude
        if gamma > 0:
            tensorboard.add_scalar('l1_s_params', _l1_over_s_params(model).item(), epoch)

        print(f"Epoch {epoch}"
            f"\n\t train-loss: {epoch_trn_loss} | valid-loss: {val_loss} "
            f"\t| valid-metric: {val_metric} | lr: {optimizer.param_groups[0]['lr']}")
        
        tensorboard.flush()

        # Early stopping bookkeeping
        if val_loss < best_val_loss:
            not_learning_count = 0
            print(f"\n\t\t\t\tNew best val_metric: {val_loss}. Saving model...\n")
            end = time.time()
            print(f"Training took {end - start} seconds.")
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()},
                    log_dir + f"/best_one_step_{model_name}.pth")
            best_val_loss = val_loss
        else:
            not_learning_count += 1

        if not_learning_count > not_learning_limit:
            print("Training is INTERRUPTED.")
            tensorboard.close()
            checkpoint_best = torch.load(log_dir + f"/best_one_step_{model_name}.pth")
            model.load_state_dict(checkpoint_best['model_state_dict'])
            epoch_best = checkpoint_best['epoch']
            model.eval()
            print(f"Best model was at epoch: {epoch_best}")
            return model, epoch_best, trn_loss_per_epoch, val_loss_per_epoch

    print("Training is finished.")
    tensorboard.close()
    checkpoint_best = torch.load(log_dir + f"/best_one_step_{model_name}.pth")
    model.load_state_dict(checkpoint_best['model_state_dict'])
    epoch_best = checkpoint_best['epoch']
    model.eval()
    print(f"Best model was at epoch: {epoch_best}")
    return model, epoch_best, trn_loss_per_epoch, val_loss_per_epoch


### =========================Clustering==================================

def make_graph_clusters(S_spatial, num_clusters: int = 100):
    """
    Returns a list of node index arrays (each cluster).
    """
    N = S_spatial.shape[0]
    clustering = SpectralClustering(
        n_clusters=num_clusters,
        affinity='precomputed',
        assign_labels='kmeans',
        random_state=42
    )
    labels = clustering.fit_predict(S_spatial.toarray())
    clusters = [np.where(labels == c)[0] for c in np.unique(labels)]
    return clusters


def train_mode_clusterGCN(
        model, model_name, training_data, validation_data, 
        single_step_trn_labels, single_step_val_labels,
        S_spatial, clusters,
        num_epochs, clusters_per_batch,
        loss_criterion, optimizer, scheduler,
        val_metric_criterion,
        log_dir, not_learning_limit,
        gamma: float = 0.0):
    """
    ClusterGCN-like training loop: samples clusters (subgraphs) each batch.
    `clusters` is a list of np.arrays with node indices.
    """

    start = time.time()
    tensorboard = SummaryWriter(log_dir=log_dir)
    trn_loss_per_epoch, val_loss_per_epoch = [], []

    n_trn_clusters = len(clusters)
    print(f"{n_trn_clusters} total clusters | {clusters_per_batch} per batch")

    best_val_metric = 10e10
    not_learning_count = 0

    # Model-specific reshaping of training/validation data
    if model_name in ["parametric_gtcnn", "disjoint_st_baseline"]:
        # [B, N, T] -> [B,1,N,T] -> [B,1,N*T]
        training_data = training_data.unsqueeze(1).flatten(2, 3)  
        validation_data = validation_data.unsqueeze(1).flatten(2, 3)

    for epoch in range(num_epochs):
        print(f"\nEpoch: {epoch}")
        model.train()
        batch_losses = []

        random.shuffle(clusters)

        for batch_idx in range(0, n_trn_clusters, clusters_per_batch):

            batch_clusters = clusters[batch_idx:batch_idx + clusters_per_batch]
            batch_nodes = np.concatenate(batch_clusters)
            batch_nodes = np.unique(batch_nodes)

            # Extract subgraph adjacency
            S_sub = S_spatial[batch_nodes][:, batch_nodes]

            # Slice data tensors (shape [B, F, N, T] or flattened [B, F, N*T])
            batch_trn_data = training_data[:, :, batch_nodes]  # [B,F,N_sub*T] if flattened
            batch_trn_labels = single_step_trn_labels[:, batch_nodes]

            # Run model on subgraph
            one_step_pred_trn = model(batch_trn_data)
            print(f"Prediction done.")

            # Compute loss with optional regularization
            base = loss_criterion(one_step_pred_trn, batch_trn_labels)
            reg = gamma * _l1_over_s_params(model)
            batch_loss = base + reg

            optimizer.zero_grad()
            print(f"Ready to backprop...")
            batch_loss.backward()
            print(f"Backprop done.")
            optimizer.step()

            batch_losses.append(batch_loss.item())

        epoch_trn_loss = float(np.mean(batch_losses))
        trn_loss_per_epoch.append(epoch_trn_loss)
        tensorboard.add_scalar('train-loss', epoch_trn_loss, epoch)

        # Validation: MSE only (no L1) for fair comparison/early stopping
        val_loss = compute_loss_in_chunks(model, validation_data, single_step_val_labels, loss_criterion)
        val_loss_per_epoch.append(val_loss)
        tensorboard.add_scalar('val-loss', val_loss, epoch)
        
        val_metric = compute_loss_in_chunks(model, validation_data, single_step_val_labels, val_metric_criterion)
        tensorboard.add_scalar('val-metric', val_metric, epoch)

        if scheduler:
            scheduler.step(val_loss)

        # Log diff, lr, and current L1 on s_*
        diff_loss = abs(epoch_trn_loss - val_loss)
        tensorboard.add_scalar('diff-loss', diff_loss, epoch)
        tensorboard.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        # Log s_ij values (already in your code)
        names = list(dict(model.named_parameters()).keys())
        s_parameters_names = [name for name in names if str(name).startswith("s_")]
        for name in s_parameters_names:
            tensorboard.add_scalar(
                name.replace(".", "/").replace("GFL/", ""),
                round(dict(model.named_parameters())[name].item(), 3),
                epoch
            )
        # Also log L1 magnitude
        if gamma > 0:
            tensorboard.add_scalar('l1_s_params', _l1_over_s_params(model).item(), epoch)

        print(f"Epoch {epoch}"
            f"\n\t train-loss: {epoch_trn_loss} | valid-loss: {val_loss} "
            f"\t| valid-metric: {val_metric} | lr: {optimizer.param_groups[0]['lr']}")
        
        tensorboard.flush()

        # Early stopping bookkeeping
        if val_loss < best_val_loss:
            not_learning_count = 0
            print(f"\n\t\t\t\tNew best val_metric: {val_loss}. Saving model...\n")
            end = time.time()
            print(f"Training took {end - start} seconds.")
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()},
                    log_dir + f"/best_one_step_{model_name}.pth")
            best_val_loss = val_loss
        else:
            not_learning_count += 1

        if not_learning_count > not_learning_limit:
            print("Training is INTERRUPTED.")
            tensorboard.close()
            checkpoint_best = torch.load(log_dir + f"/best_one_step_{model_name}.pth")
            model.load_state_dict(checkpoint_best['model_state_dict'])
            epoch_best = checkpoint_best['epoch']
            model.eval()
            print(f"Best model was at epoch: {epoch_best}")
            return model, epoch_best, trn_loss_per_epoch, val_loss_per_epoch

    print("Training is finished.")
    tensorboard.close()
    checkpoint_best = torch.load(log_dir + f"/best_one_step_{model_name}.pth")
    model.load_state_dict(checkpoint_best['model_state_dict'])
    epoch_best = checkpoint_best['epoch']
    model.eval()
    print(f"Best model was at epoch: {epoch_best}")
    return model, epoch_best, trn_loss_per_epoch, val_loss_per_epoch



### =========================Clustering==================================

def make_graph_clusters(S_spatial, num_clusters: int = 100):
    """
    Returns a list of node index arrays (each cluster).
    """
    N = S_spatial.shape[0]
    clustering = SpectralClustering(
        n_clusters=num_clusters,
        affinity='precomputed',
        assign_labels='kmeans',
        random_state=42
    )
    labels = clustering.fit_predict(S_spatial.toarray())
    clusters = [np.where(labels == c)[0] for c in np.unique(labels)]
    return clusters


def train_mode_clusterGCN(
        model, model_name, training_data, validation_data, 
        single_step_trn_labels, single_step_val_labels,
        S_spatial, clusters,
        num_epochs, clusters_per_batch,
        loss_criterion, optimizer, scheduler,
        val_metric_criterion,
        log_dir, not_learning_limit,
        gamma: float = 0.0):
    """
    ClusterGCN-like training loop: samples clusters (subgraphs) each batch.
    `clusters` is a list of np.arrays with node indices.
    """

    start = time.time()
    tensorboard = SummaryWriter(log_dir=log_dir)
    trn_loss_per_epoch, val_loss_per_epoch = [], []

    n_trn_clusters = len(clusters)
    print(f"{n_trn_clusters} total clusters | {clusters_per_batch} per batch")

    best_val_metric = 10e10
    not_learning_count = 0

    # Model-specific reshaping of training/validation data
    if model_name in ["parametric_gtcnn", "disjoint_st_baseline"]:
        # [B, N, T] -> [B,1,N,T] -> [B,1,N*T]
        training_data = training_data.unsqueeze(1).flatten(2, 3)  
        validation_data = validation_data.unsqueeze(1).flatten(2, 3)

    for epoch in range(num_epochs):
        print(f"\nEpoch: {epoch}")
        model.train()
        batch_losses = []

        random.shuffle(clusters)

        for batch_idx in range(0, n_trn_clusters, clusters_per_batch):

            batch_clusters = clusters[batch_idx:batch_idx + clusters_per_batch]
            batch_nodes = np.concatenate(batch_clusters)
            batch_nodes = np.unique(batch_nodes)

            # Extract subgraph adjacency
            S_sub = S_spatial[batch_nodes][:, batch_nodes]

            # Slice data tensors (shape [B, F, N, T] or flattened [B, F, N*T])
            batch_trn_data = training_data[:, :, batch_nodes]  # [B,F,N_sub*T] if flattened
            batch_trn_labels = single_step_trn_labels[:, batch_nodes]

            # Run model on subgraph
            one_step_pred_trn = model(batch_trn_data)
            print(f"Prediction done.")

            # Compute loss with optional regularization
            base = loss_criterion(one_step_pred_trn, batch_trn_labels)
            reg = gamma * _l1_over_s_params(model)
            batch_loss = base + reg

            optimizer.zero_grad()
            print(f"Ready to backprop...")
            batch_loss.backward()
            print(f"Backprop done.")
            optimizer.step()

            batch_losses.append(batch_loss.item())

        epoch_trn_loss = float(np.mean(batch_losses))
        trn_loss_per_epoch.append(epoch_trn_loss)
        tensorboard.add_scalar('train-loss', epoch_trn_loss, epoch)

        # Validation: MSE only (no L1) for fair comparison/early stopping
        val_loss = compute_loss_in_chunks(model, validation_data, single_step_val_labels, loss_criterion)
        val_loss_per_epoch.append(val_loss)
        tensorboard.add_scalar('val-loss', val_loss, epoch)
        
        val_metric = compute_loss_in_chunks(model, validation_data, single_step_val_labels, val_metric_criterion)
        tensorboard.add_scalar('val-metric', val_metric, epoch)

        if scheduler:
            scheduler.step(val_metric)

        # Log diff, lr, and current L1 on s_*
        diff_loss = abs(epoch_trn_loss - val_loss)
        tensorboard.add_scalar('diff-loss', diff_loss, epoch)
        tensorboard.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        # Log s_ij values (already in your code)
        names = list(dict(model.named_parameters()).keys())
        s_parameters_names = [name for name in names if str(name).startswith("s_")]
        for name in s_parameters_names:
            tensorboard.add_scalar(
                name.replace(".", "/").replace("GFL/", ""),
                round(dict(model.named_parameters())[name].item(), 3),
                epoch
            )
        # Also log L1 magnitude
        if gamma > 0:
            tensorboard.add_scalar('l1_s_params', _l1_over_s_params(model).item(), epoch)

        print(f"Epoch {epoch}"
            f"\n\t train-loss: {round(epoch_trn_loss, 3)} | valid-loss: {round(val_loss, 3)} "
            f"\t| valid-metric: {val_metric} | lr: {optimizer.param_groups[0]['lr']}")

        # Early stopping bookkeeping
        if val_metric < best_val_metric:
            not_learning_count = 0
            print(f"\n\t\t\t\tNew best val_metric: {val_metric}. Saving model...\n")
            end = time.time()
            print(f"Training took {end - start} seconds.")
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()},
                    log_dir + f"/best_one_step_{model_name}.pth")
            best_val_metric = val_metric
        else:
            not_learning_count += 1

        if not_learning_count > not_learning_limit:
            print("Training is INTERRUPTED.")
            tensorboard.close()
            checkpoint_best = torch.load(log_dir + f"/best_one_step_{model_name}.pth")
            model.load_state_dict(checkpoint_best['model_state_dict'])
            epoch_best = checkpoint_best['epoch']
            model.eval()
            print(f"Best model was at epoch: {epoch_best}")
            return model, epoch_best

    print("Training is finished.")
    tensorboard.close()
    checkpoint_best = torch.load(log_dir + f"/best_one_step_{model_name}.pth")
    model.load_state_dict(checkpoint_best['model_state_dict'])
    epoch_best = checkpoint_best['epoch']
    model.eval()
    print(f"Best model was at epoch: {epoch_best}")
    return model, epoch_best, trn_loss_per_epoch, val_loss_per_epoch