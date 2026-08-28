"""Confusion matrix generation for single-label argmax evaluation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import MODELS_DIR, PROCESSED_DATA_DIR, REPO_ROOT, SOUND_CLASSES
from src.evaluation.metrics import compute_metrics, print_metrics, write_evaluation_report
from src.training.losses import weighted_binary_crossentropy


def _to_multihot(y: np.ndarray, num_classes: int) -> np.ndarray:
    one_hot = np.zeros((len(y), num_classes), dtype=np.float32)
    for i, label in enumerate(y):
        if 0 <= label < num_classes:
            one_hot[i, label] = 1.0
    return one_hot


def evaluate_and_plot(
    checkpoint: Path | None = None,
    output_dir: Path | None = None,
) -> Path:
    checkpoint = checkpoint or MODELS_DIR / "checkpoints" / "best_model.keras"
    if not checkpoint.exists():
        checkpoint = MODELS_DIR / "checkpoints" / "best_classifier.keras"
    output_dir = output_dir or REPO_ROOT / "docs"
    output_dir.mkdir(parents=True, exist_ok=True)

    test_path = PROCESSED_DATA_DIR / "embeddings_test.npz"
    data = np.load(test_path)
    X_test, y_test = data["X"], data["y"]
    y_true = _to_multihot(y_test, len(SOUND_CLASSES))

    model = tf.keras.models.load_model(checkpoint, custom_objects={"weighted_binary_crossentropy": weighted_binary_crossentropy})
    y_pred = model.predict(X_test, verbose=0)

    metrics = compute_metrics(y_true, y_pred)
    print_metrics(metrics)
    write_evaluation_report(metrics, output_dir / "evaluation_report.md")

    y_true_argmax = np.argmax(y_true, axis=1)
    y_pred_argmax = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_true_argmax, y_pred_argmax, labels=list(range(len(SOUND_CLASSES))))

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=SOUND_CLASSES,
        yticklabels=SOUND_CLASSES,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("SAHARA Baseline Confusion Matrix (test set)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    out_path = output_dir / "confusion_matrix.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nConfusion matrix saved to {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    evaluate_and_plot(args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
