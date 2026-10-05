"""Evaluate a run on the held-out test data, reported as three separate test sets.

  public       FSD50K / AudioSet / ESC-50 targets + DESED / ESC-50 / FSD50K background (real)
  indian_team  the team's test-side recordings (per-label split, all recorders): self-recorded, real
  synthetic    test-split targets mixed onto test-split background (synthetic_mixing --split test);
               never used for training and never reported as real-world performance

Per set: window-level AP / precision / recall / F1 at the tuned thresholds (class_thresholds.json)
and macro mAP; a window confusion matrix; and clip-level detection with the app's policy (does
playing the clip through the phone pipeline raise the right alert, and any wrong ones?).
Writes models/logs/<run>/test_metrics.json and confusion_test_<set>.csv/.png.

    python src/evaluation/evaluate.py [--run <run folder>]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import METADATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, SOUND_CLASSES
from src.evaluation.metrics import window_confusion
from src.evaluation.policy import DetectionPolicy
from src.models.classifier_head import head_predict
from src.evaluation.streaming import clip_hop_scores, load_head, load_hits, load_thresholds, run_policy, selected_run_dir
from src.preprocessing.build_features_cache import build_hop_cache
from src.preprocessing.features import hop_cache_path, load_hop_cache
from src.preprocessing.windowing import clip_examples

PUBLIC = {"fsd50k", "audioset", "esc50", "desed"}
SYNTHETIC_TEST_DIR = RAW_DATA_DIR / "synthetic_mixed_test"


def window_metrics(Y: np.ndarray, scores: np.ndarray, thresholds: dict[str, float]) -> dict:
    from sklearn.metrics import average_precision_score

    per_class, aps = {}, []
    for i, c in enumerate(SOUND_CLASSES):
        y = Y[:, i] > 0
        if not y.any():
            continue
        pred = scores[:, i] >= thresholds.get(c, 0.5)
        tp = int((pred & y).sum())
        p = tp / pred.sum() if pred.sum() else 0.0
        r = tp / y.sum()
        ap = float(average_precision_score(y, scores[:, i]))
        aps.append(ap)
        per_class[c] = {"ap": round(ap, 4), "precision": round(p, 4), "recall": round(r, 4),
                        "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
                        "threshold": thresholds.get(c, 0.5), "positive_windows": int(y.sum())}
    neg = Y.sum(axis=1) == 0
    return {
        "mAP": round(float(np.mean(aps)), 4) if aps else None,
        "classes_in_mAP": list(per_class),
        "per_class": per_class,
        "windows": int(len(Y)),
        "background_windows": int(neg.sum()),
        "background_windows_with_any_alert_score": int((scores[neg] >= np.array([thresholds.get(c, 0.5) for c in SOUND_CLASSES])).any(axis=1).sum()),
    }


def clip_detection(model, clips: list[tuple[str, set[str]]], thresholds: dict[str, float]) -> dict:
    """Play each target clip alone through the policy; count right and wrong alerts."""
    hit, total, wrong = defaultdict(int), defaultdict(int), defaultdict(int)
    first_alert_s = defaultdict(list)
    for filepath, labels in clips:
        scores = clip_hop_scores(model, filepath)
        if scores is None:
            continue
        policy = DetectionPolicy(SOUND_CLASSES, thresholds, hits_per_class=load_hits())
        detections, _ = run_policy(policy, scores)
        fired = {d.label for d in detections}
        for c in labels:
            total[c] += 1
            if c in fired:
                hit[c] += 1
                # +1 s: the phone's first decision needs a full 1 s buffer.
                first_alert_s[c].append(1.0 + min(d.timestamp_ms for d in detections if d.label == c) / 1000)
        for c in fired - labels:
            wrong[c] += 1
    return {
        c: {"clips": total[c], "detected": hit[c], "recall": round(hit[c] / total[c], 4),
            "median_first_alert_s_from_clip_start": round(float(np.median(first_alert_s[c])), 2) if first_alert_s[c] else None}
        for c in SOUND_CLASSES if total[c]
    } | {"_wrong_class_alerts": dict(wrong), "_clips": sum(1 for _ in clips)}


def _real_test_sets() -> dict[str, dict]:
    X = np.load(PROCESSED_DATA_DIR / "test_embeddings.npy")
    Y = np.load(PROCESSED_DATA_DIR / "test_labels.npy")
    meta = dict(np.load(PROCESSED_DATA_DIR / "test_windows_meta.npz"))
    with (METADATA_DIR / "test_manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    clips: dict[str, dict] = {}
    for r in rows:
        c = clips.setdefault(r["filepath"], {"labels": set(), "source_dataset": r["source_dataset"]})
        if r["label"] in SOUND_CLASSES:
            c["labels"].add(r["label"])
    sets = {}
    for name, keep_src in (("public", lambda s: s in PUBLIC), ("indian_team", lambda s: s == "indian_ambient")):
        mask = np.array([keep_src(s) for s in meta["source_dataset"]])
        sets[name] = {"X": X[mask], "Y": Y[mask],
                      "clips": [(fp, c["labels"]) for fp, c in clips.items() if keep_src(c["source_dataset"]) and c["labels"]]}
    return sets


def _synthetic_test_set() -> dict | None:
    meta_path = SYNTHETIC_TEST_DIR / "synthetic_metadata.csv"
    if not meta_path.exists():
        return None
    with meta_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    files = [str((SYNTHETIC_TEST_DIR / r["relative_filepath"]).resolve()) for r in rows]
    build_hop_cache([(fp, "") for fp in files])
    xs, ys, clips = [], [], []
    for fp, r in zip(files, rows):
        if not hop_cache_path(fp).exists():
            continue
        x, y, _ = clip_examples(load_hop_cache(hop_cache_path(fp)), {r["label"]})
        if len(x):
            xs.append(x)
            ys.append(y)
        clips.append((fp, {r["label"]}))
    return {"X": np.concatenate(xs), "Y": np.concatenate(ys), "clips": clips}


def run(run: str | None = None) -> dict:
    run_dir = selected_run_dir(run)
    model = load_head(run_dir)
    thresholds = load_thresholds()
    sets = _real_test_sets()
    synthetic = _synthetic_test_set()
    if synthetic:
        sets["synthetic"] = synthetic
    result = {"run": run_dir.name, "split": "test", "thresholds": thresholds, "sets": {}}
    for name, data in sets.items():
        scores = head_predict(model, data["X"])
        metrics = window_metrics(data["Y"], scores, thresholds)
        metrics["clip_detection_with_app_policy"] = clip_detection(model, data["clips"], thresholds)
        metrics["source_type"] = "synthetic_mixed" if name == "synthetic" else ("self_recorded" if name == "indian_team" else "real")
        window_confusion(data["Y"], scores, run_dir / f"confusion_test_{name}",
                         threshold=np.array([thresholds.get(c, 0.5) for c in SOUND_CLASSES]))
        result["sets"][name] = metrics
        print(f"\n== {name} ({metrics['windows']} windows, {metrics['background_windows']} background); "
              f"mAP {metrics['mAP']} over {len(metrics['classes_in_mAP'])} classes")
        print("class            AP     P      R      F1    clip-detect")
        for c, m in metrics["per_class"].items():
            cd = metrics["clip_detection_with_app_policy"].get(c, {})
            print(f"{c:16s} {m['ap']:.3f}  {m['precision']:.3f}  {m['recall']:.3f}  {m['f1']:.3f}  "
                  f"{cd.get('detected', 0)}/{cd.get('clips', 0)}")
    (run_dir / "test_metrics.json").write_text(json.dumps(result, indent=2, default=list))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", help="Run folder name or path (default: models/logs/selected_run.txt)")
    run(parser.parse_args().run)
