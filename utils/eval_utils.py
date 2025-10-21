import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, confusion_matrix
from utils.train_utils import perform_chunk_predictions, compute_loss_in_chunks
from typing import Optional

def evaluate_model(model: torch.nn.Module,
                   data: torch.Tensor,
                   labels: torch.Tensor,
                   loss_criterion,                    # allow any torch loss
                   apply_sigmoid: bool = True,        # set True only for BCE-with-logits style heads
                   chunk_size: int = 300,
                   event_times: Optional[np.ndarray] = None) -> dict:
    """
    Evaluates the model on the test set and computes metrics.

    If `event_times` (numpy [B, T]) is provided, it is used for event-based models
    during both loss computation and prediction.

    Returns a dictionary with:
      - test_loss
      - accuracy, f1 (global)
      - precision_macro, recall_macro
      - precision_per_class, recall_per_class, f1_per_class
      - confusion_matrix
    NOTE: Classification metrics assume binary labels (0/1). If you're doing regression,
    use an MSE-like loss and skip interpreting the classification metrics.
    """
    # Loss (chunked)
    test_loss = compute_loss_in_chunks(
        model, data, labels, loss_criterion,
        chunk_size=chunk_size,
        event_times=event_times
    )

    # Predictions (chunked)
    preds = perform_chunk_predictions(
        model, data,
        chunk_size=chunk_size,
        event_times=event_times
    )

    # Binarize predictions and labels for classification metrics
    if apply_sigmoid:
        preds = torch.sigmoid(preds)

    threshold = 0.5
    preds_bin = (preds >= threshold).int()
    labels_bin = (labels >= threshold).int()

    # To numpy
    preds_bin_np = preds_bin.cpu().numpy().ravel()
    labels_bin_np = labels_bin.cpu().numpy().ravel()

    accuracy = accuracy_score(labels_bin_np, preds_bin_np)
    f1 = f1_score(labels_bin_np, preds_bin_np, zero_division=0)
    precision, recall, f1_per_class, _ = precision_recall_fscore_support(
        labels_bin_np, preds_bin_np, zero_division=0
    )
    cm = confusion_matrix(labels_bin_np, preds_bin_np, labels=[0, 1])

    return {
        'test_loss': float(test_loss),
        'accuracy': float(accuracy),
        'f1': float(f1),
        'precision_macro': float(np.mean(precision)),
        'recall_macro': float(np.mean(recall)),
        'precision_per_class': precision.tolist(),
        'recall_per_class': recall.tolist(),
        'f1_per_class': f1_per_class.tolist(),
        'confusion_matrix': cm.tolist()
    }