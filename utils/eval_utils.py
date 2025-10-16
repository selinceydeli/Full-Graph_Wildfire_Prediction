import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, confusion_matrix
from utils.train_utils import perform_chunk_predictions, compute_loss_in_chunks

def evaluate_model(model: torch.nn.Module,
                   data: torch.Tensor,
                   labels: torch.Tensor,
                   loss_criterion: torch.nn.BCEWithLogitsLoss,
                   apply_sigmoid: bool = True,
                   chunk_size: int = 300) -> dict:
    """
    Evaluates the model on the test set and computes various metrics.
    Returns a dictionary with test loss, accuracy, F1 score, precision, recall, and confusion matrix
    """

    print("Computing test metrics...")
    test_loss = compute_loss_in_chunks(model, data, labels, loss_criterion)
    preds = perform_chunk_predictions(model, data, chunk_size)
    
    # Binarize predictions and labels for classification metrics
    threshold = 0.5
    if apply_sigmoid:
        preds = torch.sigmoid(preds)
    preds_bin = (preds >= threshold).int()
    labels_bin = (labels >= threshold).int()

    # Convert to NumPy arrays for sklearn
    preds_bin_np = preds_bin.cpu().numpy().flatten()
    labels_bin_np = labels_bin.cpu().numpy().flatten()

    accuracy = accuracy_score(labels_bin_np, preds_bin_np)
    f1 = f1_score(labels_bin_np, preds_bin_np, zero_division=0)
    precision, recall, f1_per_class, _ = precision_recall_fscore_support(labels_bin_np, preds_bin_np, zero_division=0)
    cm = confusion_matrix(labels_bin_np, preds_bin_np, labels=[0,1])

    return {
        'test_loss': test_loss,
        'accuracy': float(accuracy),
        'f1': float(f1),
        'precision_macro': float(np.mean(precision)),
        'recall_macro': float(np.mean(recall)),
        'precision_per_class': precision.tolist(),
        'recall_per_class': recall.tolist(),
        'f1_per_class': f1_per_class.tolist(),
        'confusion_matrix': cm.tolist()
    }