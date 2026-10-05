"""Average several trained heads into one model (same 1024 -> 8 interface) and record it as a run.

    python scripts/make_ensemble.py --name e_bg8_hn_x5 models/logs/<run1> models/logs/<run2> ...

Writes models/logs/<YYYYMMDD-HHMMSS>_<name>/ with model.keras, config.json (members), metrics.json
(validation, with val_sha256), label_order.txt and confusion_val.*, like src/training/train.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import MODELS_DIR, SAFETY_CRITICAL_CLASSES, SAFETY_PRECISION_FLOOR, SOUND_CLASSES
from src.evaluation.metrics import compute_metrics, safety_floor_recall, window_confusion
from src.models.classifier_head import HEAD_DEVICE, head_predict
from src.training.train import _config_snapshot, _git_sha, load_split, validation_fingerprint


def build_ensemble(members: list[Path]) -> tf.keras.Model:
    with tf.device(HEAD_DEVICE):
        models = []
        for i, m in enumerate(members):
            model = tf.keras.models.load_model(m / "model.keras", compile=False)
            model.name = f"member_{i}"
            models.append(model)
        inputs = tf.keras.Input(shape=(1024,), name="embedding")
        outputs = tf.keras.layers.Average(name="predictions")([m(inputs) for m in models])
        return tf.keras.Model(inputs, outputs, name="sahara_head_ensemble")


def main() -> Path:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("members", nargs="+", type=Path)
    args = parser.parse_args()
    run_dir = MODELS_DIR / "logs" / f"{datetime.now():%Y%m%d-%H%M%S}_{args.name}"
    run_dir.mkdir(parents=True)
    model = build_ensemble(args.members)
    model.save(run_dir / "model.keras")
    X_val, Y_val, _ = load_split("val")
    scores = head_predict(model, X_val)
    metrics = compute_metrics(Y_val, scores)
    floor = safety_floor_recall(Y_val, scores)
    result = {
        "run": run_dir.name, "split": "val", "best_epoch": None, "epochs_run": None,
        "val_mAP": metrics["macro"]["mAP"], "macro_at_0.5": metrics["macro"], "per_class_at_0.5": metrics["per_class"],
        "safety_floor": {"precision_floor": SAFETY_PRECISION_FLOOR, **floor},
        "safety_floor_met": all(floor[c]["recall"] > 0 for c in SAFETY_CRITICAL_CLASSES),
        "train_windows": None, "val_windows": int(len(X_val)), "val_sha256": validation_fingerprint(),
    }
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    (run_dir / "label_order.txt").write_text("\n".join(SOUND_CLASSES) + "\n")
    window_confusion(Y_val, scores, run_dir / "confusion_val")
    member_cfg = json.loads((args.members[0] / "config.json").read_text())
    (run_dir / "config.json").write_text(json.dumps({
        "name": args.name, "git_sha": _git_sha(), "created": datetime.now().isoformat(timespec="seconds"),
        "ensemble_of": [m.name for m in args.members],
        "args": {**member_cfg["args"], "seed": [json.loads((m / "config.json").read_text())["args"]["seed"]
                                                for m in args.members]},
        "manifest_sha256": member_cfg["manifest_sha256"], "features_sha256": member_cfg["features_sha256"],
        "config": _config_snapshot(),
    }, indent=2, default=str))
    print(f"{run_dir.name}: ensemble of {len(args.members)}, val mAP {result['val_mAP']:.3f}")
    return run_dir


if __name__ == "__main__":
    np.random.seed(0)
    main()
