from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score


def classification_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    probability = np.asarray(probability)
    y_pred = (probability >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    result = {
        "accuracy": float((y_true == y_pred).mean()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auprc": float(average_precision_score(y_true, probability)),
    }
    result["auroc"] = float(roc_auc_score(y_true, probability)) if len(np.unique(y_true)) == 2 else float("nan")
    return result


def best_f1_threshold(y_true: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    """Select a threshold on validation data only."""
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in np.linspace(0.15, 0.85, 141):
        score = classification_metrics(y_true, probability, float(threshold))["f1"]
        if score > best_f1:
            best_threshold, best_f1 = float(threshold), score
    return best_threshold, best_f1


def set_recovery_metrics(reference: np.ndarray, predicted: np.ndarray, prefix: str) -> dict[str, float]:
    # Mesh entities are unordered; canonicalization avoids penalizing a triangle
    # merely because Delaunay returned a different vertex orientation.
    ref = {tuple(sorted(map(int, row))) for row in reference}
    pred = {tuple(sorted(map(int, row))) for row in predicted}
    tp = len(ref & pred)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(ref) if ref else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {f"{prefix}_precision": precision, f"{prefix}_recall": recall, f"{prefix}_f1": f1}
