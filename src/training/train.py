"""Train the SAHARA classifier head on cached YAMNet embeddings."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    SOUND_CLASSES,
)
from src.models.classifier_head import build_classifier_head
from src.training.losses import weighted_binary_crossentropy


def _to_multihot(y: np.ndarray, num_classes: int) -> np.ndarray:
    one_hot = np.zeros((len(y), num_classes), dtype=np.float32)
    for i, label in enumerate(y):
        if 0 <= label < num_classes:
            one_hot[i, label] = 1.0
    return one_hot


def _load_split(name: str) -> tuple[np.ndarray, np.ndarray]:
    npy_x = PROCESSED_DATA_DIR / f"{name}_embeddings.npy"
    npy_y = PROCESSED_DATA_DIR / f"{name}_labels.npy"
    if npy_x.exists() and npy_y.exists():
        X, y = np.load(npy_x), np.load(npy_y)
        return X, y
    path = PROCESSED_DATA_DIR / f"embeddings_{name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path} — run build_features_cache.py first")
    data = np.load(path)
    return data["X"], data["y"]


def train(
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
) -> dict:
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    X_train, y_train = _load_split("train")
    X_val, y_val = _load_split("val")
    num_classes = len(SOUND_CLASSES)

    y_train_oh = y_train if y_train.ndim == 2 else _to_multihot(y_train, num_classes)
    y_val_oh = y_val if y_val.ndim == 2 else _to_multihot(y_val, num_classes)

    model = build_classifier_head()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=weighted_binary_crossentropy,
        metrics=["accuracy"],
    )

    ckpt_dir = MODELS_DIR / "checkpoints"
    log_dir = MODELS_DIR / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss"),
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_dir / "best_model.keras",
            save_best_only=True,
            monitor="val_loss",
        ),
        tf.keras.callbacks.TensorBoard(log_dir=str(log_dir)),
    ]

    history = model.fit(
        X_train,
        y_train_oh,
        validation_data=(X_val, y_val_oh),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    final_train_loss = float(history.history["loss"][-1])
    final_train_acc = float(history.history["accuracy"][-1])
    final_val_loss = float(history.history["val_loss"][-1])
    final_val_acc = float(history.history["val_accuracy"][-1])

    print(f"\nFinal train loss={final_train_loss:.4f} acc={final_train_acc:.4f}")
    print(f"Final val   loss={final_val_loss:.4f} acc={final_val_acc:.4f}")

    return {
        "train_loss": final_train_loss,
        "train_acc": final_train_acc,
        "val_loss": final_val_loss,
        "val_acc": final_val_acc,
        "model": model,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    args = parser.parse_args()
    train(args.batch_size, args.epochs, args.learning_rate)


if __name__ == "__main__":
    main()
