"""Per-class and macro evaluation metrics for multi-label sound classification."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, precision_score, recall_score

from src.config import SAFETY_CRITICAL_CLASSES, SAFETY_PRECISION_FLOOR, SOUND_CLASSES

BACKGROUND = "background"


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


def safety_floor_recall(
    y_true: np.ndarray, y_score: np.ndarray, floor: float = SAFETY_PRECISION_FLOOR,
    classes: tuple[str, ...] = SAFETY_CRITICAL_CLASSES,
) -> dict[str, dict]:
    """Best recall each safety-critical class reaches with precision >= ``floor``.

    recall 0.0 means no threshold meets the floor. ``positives`` is the number of positive
    windows the figure rests on.
    """
    out = {}
    for name in classes:
        i = SOUND_CLASSES.index(name)
        yt, ys = y_true[:, i], y_score[:, i]
        if yt.sum() == 0:
            out[name] = {"recall": 0.0, "precision": 0.0, "threshold": None, "positives": 0}
            continue
        precision, recall, thresholds = precision_recall_curve(yt, ys)
        ok = np.flatnonzero(precision[:-1] >= floor)
        if ok.size == 0:
            out[name] = {"recall": 0.0, "precision": 0.0, "threshold": None, "positives": int(yt.sum())}
            continue
        best = ok[np.argmax(recall[ok])]
        out[name] = {"recall": float(recall[best]), "precision": float(precision[best]),
                     "threshold": round(float(thresholds[best]), 4), "positives": int(yt.sum())}
    return out


def window_confusion(y_true: np.ndarray, y_score: np.ndarray, out_stem, threshold: float = 0.5) -> np.ndarray:
    """Confusion matrix over single-label windows: rows true, columns predicted (9 x 9).

    Predicted = the top-scoring class if it reaches ``threshold``, else background.
    Windows with more than one positive label are left out. Writes <out_stem>.csv and .png.
    """
    from pathlib import Path

    labels = [*SOUND_CLASSES, BACKGROUND]
    n_pos = y_true.sum(axis=1)
    keep = n_pos <= 1
    true_idx = np.where(n_pos[keep] == 0, len(SOUND_CLASSES), y_true[keep].argmax(axis=1))
    scores = y_score[keep]
    pred_idx = np.where(scores.max(axis=1) >= threshold, scores.argmax(axis=1), len(SOUND_CLASSES))
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    np.add.at(matrix, (true_idx, pred_idx), 1)

    out_stem = Path(out_stem)
    with out_stem.with_suffix(".csv").open("w", encoding="utf-8") as handle:
        handle.write("true\\pred," + ",".join(labels) + "\n")
        for name, row in zip(labels, matrix):
            handle.write(name + "," + ",".join(str(v) for v in row) + "\n")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 7))
        norm = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        for r in range(len(labels)):
            for c in range(len(labels)):
                if matrix[r, c]:
                    ax.text(c, r, matrix[r, c], ha="center", va="center", fontsize=7,
                            color="white" if norm[r, c] > 0.5 else "black")
        ax.set_xlabel(f"predicted (top class >= {threshold}, else background)")
        ax.set_ylabel("true")
        ax.set_title(f"{out_stem.name}: single-label windows, row-normalised colour")
        fig.tight_layout()
        fig.savefig(out_stem.with_suffix(".png"), dpi=120)
        plt.close(fig)
    except Exception:  # plotting is optional; the CSV is the record
        pass
    return matrix


def print_metrics(metrics: dict) -> None:
    print("\nPer-class metrics:")
    print("class\tprecision\trecall\tF1\tAP")
    for name, values in metrics["per_class"].items():
        print(f"{name}\t{values['precision']:.3f}\t{values['recall']:.3f}\t{values['f1']:.3f}\t{values['ap']:.3f}")
    m = metrics["macro"]
    print(f"\nMacro avg: precision={m['precision']:.3f} recall={m['recall']:.3f} F1={m['f1']:.3f} mAP={m['mAP']:.3f}")


def write_evaluation_report(metrics: dict, output_path) -> None:
    """Persist the exact printed metrics for review and reproducibility."""
    lines = [
        "# SAHARA Baseline Evaluation", "",
        "Metrics use per-class one-vs-rest decisions at a 0.5 threshold. "
        "mAP uses the prediction scores and is less threshold-sensitive.", "",
        "| Class | Precision | Recall | F1 | AP |", "|---|---:|---:|---:|---:|",
    ]
    for name in SOUND_CLASSES:
        values = metrics["per_class"].get(name, {"precision": 0, "recall": 0, "f1": 0, "ap": 0})
        lines.append(f"| {name} | {values['precision']:.3f} | {values['recall']:.3f} | {values['f1']:.3f} | {values['ap']:.3f} |")
    macro = metrics["macro"]
    lines.extend([
        f"| **Macro average** | **{macro['precision']:.3f}** | **{macro['recall']:.3f}** | **{macro['f1']:.3f}** | **{macro['mAP']:.3f}** |",
        "", "## Error analysis", "",
        "The baseline remains sensitive to class imbalance. The small smoke-alarm and baby-cry subsets and acoustic overlap among alarm/beep-like events make thresholded recall unstable. Indian-context recordings and threshold calibration are deliberately deferred to the next phase.",
    ])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
