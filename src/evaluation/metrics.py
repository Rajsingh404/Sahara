"""Per-class and macro evaluation metrics for multi-label sound classification."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

from src.config import SOUND_CLASSES


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> dict:
    """
    y_true, y_pred: shape (N, num_classes) binary or probability arrays.
    Returns per-class and macro-averaged precision/recall/F1 plus mAP.
    """
    y_bin = (y_pred >= threshold).astype(int)
    num_classes = y_true.shape[1]

    per_class: dict[str, dict[str, float]] = {}
    precisions, recalls, f1s, aps = [], [], [], []

    for i, name in enumerate(SOUND_CLASSES[:num_classes]):
        yt = y_true[:, i]
        yp = y_bin[:, i]
        yscore = y_pred[:, i]
        if yt.sum() == 0:
            continue
        p = precision_score(yt, yp, zero_division=0)
        r = recall_score(yt, yp, zero_division=0)
        f = f1_score(yt, yp, zero_division=0)
        ap = average_precision_score(yt, yscore) if yt.sum() > 0 else 0.0
        per_class[name] = {"precision": p, "recall": r, "f1": f, "ap": ap}
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
        aps.append(ap)

    macro = {
        "precision": float(np.mean(precisions)) if precisions else 0.0,
        "recall": float(np.mean(recalls)) if recalls else 0.0,
        "f1": float(np.mean(f1s)) if f1s else 0.0,
        "mAP": float(np.mean(aps)) if aps else 0.0,
    }
    return {"per_class": per_class, "macro": macro}


def print_metrics(metrics: dict) -> None:
    print("\nPer-class metrics:")
    print("class\tprecision\trecall\tF1\tAP")
    for name, values in metrics["per_class"].items():
        print(f"{name}\t{values['precision']:.3f}\t{values['recall']:.3f}\t{values['f1']:.3f}\t{values['ap']:.3f}")
    m = metrics["macro"]
    print(f"\nMacro avg: precision={m['precision']:.3f} recall={m['recall']:.3f} F1={m['f1']:.3f} mAP={m['mAP']:.3f}")
