"""Downstream task metrics for TabReD evaluation.

Classification: ROC-AUC (binary), accuracy (multiclass).
Regression: RMSE.
See EXPERIMENT_PLAN.md §7.1. Statistical comparison utilities live in stats.py.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error


def compute_metric(y_true: np.ndarray, y_pred: np.ndarray, task: str) -> float:
    """Compute the primary metric for a TabReD dataset.

    Args:
        y_true: ground-truth labels/values.
        y_pred: for binary -> probability of positive class; for multiclass ->
                (N, C) probabilities or argmax labels; for regression -> values.
        task: one of {"binclass", "multiclass", "regression"}.
    Returns:
        scalar metric (higher better for class, lower better for RMSE).
    """
    task = task.lower()
    if task == "binclass":
        return float(roc_auc_score(y_true, y_pred))
    if task == "multiclass":
        preds = np.asarray(y_pred)
        if preds.ndim == 2:
            preds = preds.argmax(axis=1)
        return float(accuracy_score(y_true, preds))
    if task == "regression":
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    raise ValueError(f"Unknown task: {task}")


def metric_name(task: str) -> str:
    return {"binclass": "roc_auc", "multiclass": "accuracy", "regression": "rmse"}[task.lower()]
