"""Train the SAHARA classifier head on cached window vectors and record the run.

Every run writes models/logs/<YYYYMMDD-HHMM>_<name>/:
  config.json        all src/config.py values, CLI args, git SHA, manifest + feature hashes
  label_order.txt    SOUND_CLASSES, one per line (the head's output order)
  metrics.json       validation metrics of the best epoch (per class + macro, safety floor)
  history.csv        per-epoch loss and validation mAP
  confusion_val.csv / confusion_val.png
  model.keras        best epoch by validation mAP

Examples (from the repo root):
  python src/training/train.py --name a_baseline --no-synthetic
  python src/training/train.py --name b_synthetic
  python src/training/train.py --name c_finetune_team --finetune-from models/logs/<run of b>
  python src/training/train.py --name z_clipmean --features-dir data/processed/clip_mean --no-synthetic
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import src.config as config
from src.config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    METADATA_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    REPO_ROOT,
    SAFETY_CRITICAL_CLASSES,
    SAFETY_PRECISION_FLOOR,
    SOUND_CLASSES,
)
from src.evaluation.metrics import compute_metrics, safety_floor_recall, window_confusion
from src.models.classifier_head import HEAD_DEVICE, build_classifier_head, head_predict
from src.training.losses import weighted_binary_crossentropy

FINETUNE_REPLAY_RATIO = 2  # base-train windows replayed per team window while fine-tuning


def load_split(name: str, features_dir: Path = PROCESSED_DATA_DIR) -> tuple[np.ndarray, np.ndarray, dict]:
    x_path = features_dir / f"{name}_embeddings.npy"
    if not x_path.exists():
        raise FileNotFoundError(f"Missing {x_path}: run src/preprocessing/build_features_cache.py first")
    X = np.load(x_path)
    Y = np.load(features_dir / f"{name}_labels.npy")
    meta_path = features_dir / f"{name}_windows_meta.npz"
    meta = dict(np.load(meta_path, allow_pickle=False)) if meta_path.exists() else {}
    return X, Y, meta


def balanced_sample_weights(Y: np.ndarray, power: float) -> np.ndarray:
    """Weight each window by (group frequency)^-power; groups = 8 classes + background.

    power=0 means no balancing, 1 = full inverse frequency, 0.5 (default) = inverse sqrt,
    which lifts smoke_alarm / baby_cry without letting a few clips dominate. A window with
    several positives takes its rarest class's weight. Weights are scaled to mean 1.
    """
    counts = np.append(Y.sum(axis=0), (Y.sum(axis=1) == 0).sum()).astype(np.float64)
    group_w = np.where(counts > 0, counts, 1.0) ** -power
    per_window = np.where(Y > 0, Y * group_w[:-1], 0).max(axis=1)
    per_window = np.where(Y.sum(axis=1) == 0, group_w[-1], per_window)
    return (per_window / per_window.mean()).astype(np.float32)


class ValMAP(tf.keras.callbacks.Callback):
    """Adds val_mAP (macro AP over classes present in validation) to the logs."""

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        super().__init__()
        self.X, self.Y = X, Y

    def on_epoch_end(self, epoch, logs=None):
        scores = head_predict(self.model, self.X)
        logs["val_mAP"] = compute_metrics(self.Y, scores)["macro"]["mAP"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.exists() else "missing"


def validation_fingerprint(features_dir: Path = PROCESSED_DATA_DIR) -> str:
    return _sha256(features_dir / "val_embeddings.npy") + ":" + _sha256(features_dir / "val_labels.npy")


def _git_sha() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT,
                               capture_output=True, text=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except OSError:
        return "unknown"


def _config_snapshot() -> dict:
    return {k: v for k, v in vars(config).items() if k.isupper()}


def train(
    name: str,
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
    features_dir: Path = PROCESSED_DATA_DIR,
    include_synthetic: bool = True,
    balance_power: float = 0.5,
    finetune_from: Path | None = None,
    patience: int = 6,
    seed: int = RANDOM_SEED,
) -> Path:
    tf.keras.utils.set_random_seed(seed)
    run_dir = MODELS_DIR / "logs" / f"{datetime.now():%Y%m%d-%H%M}_{name}"
    run_dir.mkdir(parents=True, exist_ok=False)

    X_train, Y_train, meta_train = load_split("train", features_dir)
    # Validation is always the phone-style window set, whatever features the model trains on.
    X_val, Y_val, _ = load_split("val", PROCESSED_DATA_DIR)

    source_type = meta_train.get("source_type", np.full(len(X_train), "real"))
    if not include_synthetic:
        keep = source_type != "synthetic_mixed"
        X_train, Y_train, source_type = X_train[keep], Y_train[keep], source_type[keep]

    # Build, train and reload the head on the CPU (see classifier_head.HEAD_DEVICE).
    with tf.device(HEAD_DEVICE):
        if finetune_from is not None:
            # Team train-side windows plus a replay sample of the base data, so classes the team did
            # not record (smoke alarm, siren, glass, doorbell, dog) are not forgotten.
            team = source_type == "self_recorded"
            rng = np.random.default_rng(seed)
            base_idx = np.flatnonzero(~team)
            replay = rng.choice(base_idx, size=min(len(base_idx), FINETUNE_REPLAY_RATIO * int(team.sum())), replace=False)
            idx = np.concatenate([np.flatnonzero(team), replay])
            X_train, Y_train = X_train[idx], Y_train[idx]
            model = tf.keras.models.load_model(Path(finetune_from) / "model.keras", compile=False)
        else:
            model = build_classifier_head()

        sample_weight = balanced_sample_weights(Y_train, balance_power)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss=weighted_binary_crossentropy)

        callbacks = [
            ValMAP(X_val, Y_val),
            tf.keras.callbacks.EarlyStopping(monitor="val_mAP", mode="max", patience=patience, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(run_dir / "model.keras", monitor="val_mAP", mode="max", save_best_only=True),
            tf.keras.callbacks.CSVLogger(str(run_dir / "history.csv")),
        ]
        history = model.fit(
            X_train, Y_train, sample_weight=sample_weight,
            validation_data=(X_val, Y_val),
            epochs=epochs, batch_size=batch_size, shuffle=True, callbacks=callbacks, verbose=2,
        )

        model = tf.keras.models.load_model(run_dir / "model.keras", compile=False)
        val_scores = head_predict(model, X_val)
    metrics = compute_metrics(Y_val, val_scores)
    floor = safety_floor_recall(Y_val, val_scores)
    best_epoch = int(np.argmax(history.history["val_mAP"])) + 1
    result = {
        "run": run_dir.name,
        "split": "val",
        "best_epoch": best_epoch,
        "epochs_run": len(history.history["loss"]),
        "val_mAP": metrics["macro"]["mAP"],
        "macro_at_0.5": metrics["macro"],
        "per_class_at_0.5": metrics["per_class"],
        "safety_floor": {"precision_floor": SAFETY_PRECISION_FLOOR, **floor},
        "safety_floor_met": all(floor[c]["recall"] > 0 for c in SAFETY_CRITICAL_CLASSES),
        "train_windows": int(len(X_train)),
        "train_positives": {c: int(Y_train[:, i].sum()) for i, c in enumerate(SOUND_CLASSES)},
        "val_windows": int(len(X_val)),
        # Fingerprint of the validation set: runs are only comparable when this matches.
        "val_sha256": validation_fingerprint(),
    }
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    (run_dir / "label_order.txt").write_text("\n".join(SOUND_CLASSES) + "\n")
    window_confusion(Y_val, val_scores, run_dir / "confusion_val")

    feature_summary = features_dir / "features_summary.json"
    (run_dir / "config.json").write_text(json.dumps({
        "name": name,
        "git_sha": _git_sha(),
        "created": datetime.now().isoformat(timespec="seconds"),
        "args": {"batch_size": batch_size, "epochs": epochs, "learning_rate": learning_rate,
                 "features_dir": str(features_dir), "include_synthetic": include_synthetic,
                 "balance_power": balance_power, "finetune_from": str(finetune_from) if finetune_from else None,
                 "finetune_replay_ratio": FINETUNE_REPLAY_RATIO if finetune_from else None,
                 "patience": patience, "seed": seed},
        "manifest_sha256": {s: _sha256(METADATA_DIR / f"{s}_manifest.csv") for s in ("full", "train", "val", "test")},
        "features_sha256": {s: _sha256(features_dir / f"{s}_embeddings.npy") for s in ("train", "val", "test")},
        "features_summary": json.loads(feature_summary.read_text()) if feature_summary.exists() else None,
        "config": _config_snapshot(),
    }, indent=2, default=str))

    print(f"\n{run_dir.name}: val mAP {result['val_mAP']:.3f} (best epoch {best_epoch}); safety floor met: "
          f"{result['safety_floor_met']}")
    for c in SAFETY_CRITICAL_CLASSES:
        print(f"  {c}: recall {floor[c]['recall']:.3f} at precision >= {SAFETY_PRECISION_FLOOR} (threshold {floor[c]['threshold']})")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="Run name, e.g. a_baseline")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--features-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--no-synthetic", action="store_true", help="Drop synthetic_mixed windows from train.")
    parser.add_argument("--balance-power", type=float, default=0.5)
    parser.add_argument("--finetune-from", type=Path, help="Run folder whose model.keras is fine-tuned on team windows.")
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    train(args.name, args.batch_size, args.epochs, args.learning_rate, args.features_dir, not args.no_synthetic,
          args.balance_power, args.finetune_from, args.patience, args.seed)


if __name__ == "__main__":
    main()
