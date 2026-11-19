"""Binary metrics with 'defective' (label 1) as the positive class."""

import numpy as np


def confusion_counts(labels, predictions):
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    return {
        "tp": int(np.sum((labels == 1) & (predictions == 1))),
        "fp": int(np.sum((labels == 0) & (predictions == 1))),
        "fn": int(np.sum((labels == 1) & (predictions == 0))),
        "tn": int(np.sum((labels == 0) & (predictions == 0))),
    }


def safe_divide(numerator, denominator):
    return numerator / denominator if denominator > 0 else 0.0


def binary_scores(labels, predictions):
    """precision: of parts flagged defective, how many were.
    recall: of defective parts, how many were caught (a miss ships a bad part)."""
    counts = confusion_counts(labels, predictions)
    precision = safe_divide(counts["tp"], counts["tp"] + counts["fp"])
    recall = safe_divide(counts["tp"], counts["tp"] + counts["fn"])
    f1 = safe_divide(2 * precision * recall, precision + recall)
    accuracy = safe_divide(counts["tp"] + counts["tn"], len(labels))
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy, **counts}
