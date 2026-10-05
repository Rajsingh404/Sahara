"""Calibrate per-class alert thresholds on the validation windows -> data/metadata/class_thresholds.json.

- Safety-critical classes (smoke_alarm, siren, glass_break): the threshold with the highest recall
  whose precision is >= SAFETY_PRECISION_FLOOR. If none reaches the floor, the most precise
  threshold is kept and the class is flagged ``floor_met: false``.
- Other classes: the threshold with the best F1.
Test data is never used here.

    python src/evaluation/threshold_search.py [--run <run folder>]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import (
    METADATA_DIR,
    PROCESSED_DATA_DIR,
    SAFETY_CRITICAL_CLASSES,
    SAFETY_PRECISION_FLOOR,
    SOUND_CLASSES,
    THRESHOLD_GRID_END,
    THRESHOLD_GRID_START,
    THRESHOLD_GRID_STEP,
)
from src.evaluation.streaming import load_head, selected_run_dir
from src.models.classifier_head import head_predict


def _grid() -> np.ndarray:
    return np.round(np.arange(THRESHOLD_GRID_START, THRESHOLD_GRID_END + 1e-9, THRESHOLD_GRID_STEP), 2)


def _prf(y: np.ndarray, s: np.ndarray, t: float) -> tuple[float, float, float]:
    pred = s >= t
    tp = float((pred & (y > 0)).sum())
    p = tp / pred.sum() if pred.sum() else 0.0
    r = tp / (y > 0).sum() if (y > 0).sum() else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def choose_thresholds(Y: np.ndarray, scores: np.ndarray) -> dict[str, dict]:
    results = {}
    for i, name in enumerate(SOUND_CLASSES):
        cand = [(float(t), *_prf(Y[:, i], scores[:, i], float(t))) for t in _grid()]
        if name in SAFETY_CRITICAL_CLASSES:
            ok = [c for c in cand if c[1] >= SAFETY_PRECISION_FLOOR and c[2] > 0]
            chosen = max(ok, key=lambda c: (c[2], c[0])) if ok else max(cand, key=lambda c: (c[1], c[2]))
            rule = f"max recall with precision >= {SAFETY_PRECISION_FLOOR}" if ok else "floor not reachable: max precision"
        else:
            chosen = max(cand, key=lambda c: (c[3], c[0]))
            rule = "max F1"
        results[name] = {"threshold": chosen[0], "precision": round(chosen[1], 4), "recall": round(chosen[2], 4),
                         "f1": round(chosen[3], 4), "rule": rule, "val_positive_windows": int(Y[:, i].sum())}
        if name in SAFETY_CRITICAL_CLASSES:
            results[name]["floor_met"] = bool(ok)
    return results


def run(run: str | None = None) -> dict[str, dict]:
    run_dir = selected_run_dir(run)
    model = load_head(run_dir)
    X = np.load(PROCESSED_DATA_DIR / "val_embeddings.npy")
    Y = np.load(PROCESSED_DATA_DIR / "val_labels.npy")
    results = choose_thresholds(Y, head_predict(model, X))
    # Per-class objects first (the app's parser reads {"class": {"threshold": x, ...}}); provenance last.
    payload = {**results, "_meta": {"run": run_dir.name, "split": "val", "windows": int(len(X)),
                                    "grid": [THRESHOLD_GRID_START, THRESHOLD_GRID_END, THRESHOLD_GRID_STEP]}}
    text = json.dumps(payload, indent=2) + "\n"
    (METADATA_DIR / "class_thresholds.json").write_text(text, encoding="utf-8")
    (run_dir / "class_thresholds.json").write_text(text, encoding="utf-8")
    print(f"Thresholds from {run_dir.name} on {len(X)} validation windows")
    print("class            thr   prec  recall  f1    rule")
    for name, r in results.items():
        print(f"{name:16s} {r['threshold']:.2f}  {r['precision']:.3f} {r['recall']:.3f}  {r['f1']:.3f} {r['rule']}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", help="Run folder name or path (default: models/logs/selected_run.txt)")
    run(parser.parse_args().run)
