"""Does home calibration help in homes the model never heard?

Session-held-out folds as in `team_cv.py --group-by session`: per fold, the shipped recipe is trained on public
data plus the team recordings of the other sessions. For each held-out session and class with at least
CALIBRATION_RECORDINGS + 1 clips, CALIBRATION_RECORDINGS clips play the user's calibration recordings and the
rest are the sounds to catch, over DRAWS random draws. Reported per (session, class): share of the other clips
that alert at the shipped threshold and after calibration, and the false alarms per hour that class adds on the
public test background (3.4 h, played as one stream) at the calibrated threshold.
Writes models/logs/<run>/home_calibration.json.

    python scripts/eval_home_calibration.py [--run <run folder>]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import METADATA_DIR, RANDOM_SEED, SOUND_CLASSES  # noqa: E402
from src.evaluation import policy_tuning as pt  # noqa: E402
from src.evaluation.home_calibration import CALIBRATION_RECORDINGS, calibrate, fire_level  # noqa: E402
from src.evaluation.streaming import HOP_MS, hop_scores, load_hits, load_thresholds, selected_run_dir  # noqa: E402
from src.evaluation.team_cv import _fit, assign_session_folds, session_groups, team_clips  # noqa: E402
from src.models.classifier_head import HEAD_DEVICE, build_classifier_head  # noqa: E402
from src.preprocessing.features import hop_cache_path, load_hop_cache  # noqa: E402
from src.preprocessing.windowing import clip_examples  # noqa: E402
from src.training.train import load_split  # noqa: E402

DRAWS = 20


def floors() -> dict[str, float]:
    """Each class's threshold in the most permissive sensitivity step (alert_profiles.csv)."""
    with (METADATA_DIR / "alert_profiles.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    top = max(float(r["budget_per_hour"]) for r in rows)
    return {r["class"]: float(r["threshold"]) for r in rows if float(r["budget_per_hour"]) == top}


def public_background() -> list[str]:
    with (METADATA_DIR / "test_manifest.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    labelled = {r["filepath"] for r in rows if r["label"] in SOUND_CLASSES}
    return sorted({r["filepath"] for r in rows if r["source_type"] == "real" and r["filepath"] not in labelled})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    run_dir = selected_run_dir(args.run)
    recipe = json.loads((run_dir / "config.json").read_text())["args"]
    epochs = int(json.loads((run_dir / "metrics.json").read_text()).get("best_epoch") or 5)
    thr, hits, floor = load_thresholds(), load_hits(), floors()

    X, Y, meta = load_split("train")
    public = meta["source_type"] == "real"
    clips = team_clips()
    folds = assign_session_folds(clips, args.folds)
    groups = session_groups()
    bg_paths = public_background()
    rng = np.random.default_rng(RANDOM_SEED)
    results: dict[tuple[str, str], dict] = {}

    for f in range(args.folds):
        train_c = [c for c in clips if folds[c["filepath"]] != f]
        xs, ys = zip(*[clip_examples(load_hop_cache(hop_cache_path(c["filepath"])), c["labels"])[:2] for c in train_c])
        Xf, Yf = np.concatenate([*xs, X[public]]), np.concatenate([*ys, Y[public]])
        mask = np.arange(len(Xf)) < sum(len(x) for x in xs)
        tf.keras.utils.set_random_seed(RANDOM_SEED + f)
        with tf.device(HEAD_DEVICE):
            model = build_classifier_head()
        model = _fit(model, Xf, Yf, recipe, epochs, recipe.get("learning_rate", 3e-4), mask, recipe.get("team_weight", 10))
        bg = np.concatenate([hop_scores(model, load_hop_cache(hop_cache_path(p))) for p in bg_paths
                             if hop_cache_path(p).exists()])
        hours = len(bg) * HOP_MS / 3_600_000
        held = defaultdict(list)
        for c in clips:
            if folds[c["filepath"]] == f:
                for label in c["labels"]:
                    held[(groups[Path(c["filepath"]).name], label)].append(c)
        for (session, label), cs in sorted(held.items()):
            if len(cs) <= CALIBRATION_RECORDINGS:
                continue
            i = SOUND_CLASSES.index(label)
            k = int(hits[label])
            s = [hop_scores(model, load_hop_cache(hop_cache_path(c["filepath"])))[:, i] for c in cs]
            base_fa = pt.count_alerts(bg[:, i], float(thr[label]), k) / hours
            det0, det1, fa1, moved = [], [], [], []
            for _ in range(DRAWS):
                pick = set(rng.choice(len(cs), CALIBRATION_RECORDINGS, replace=False).tolist())
                t = calibrate(float(thr[label]), floor.get(label, float(thr[label])),
                              [fire_level(s[j], k) for j in pick])
                rest = [s[j] for j in range(len(cs)) if j not in pick]
                det0.append(np.mean([pt.count_alerts(x, float(thr[label]), k) > 0 for x in rest]))
                det1.append(np.mean([pt.count_alerts(x, t, k) > 0 for x in rest]))
                fa1.append(pt.count_alerts(bg[:, i], t, k) / hours)
                moved.append(t)
            results[(session, label)] = {
                "fold": f, "clips": len(cs), "shipped_threshold": float(thr[label]), "floor": floor.get(label),
                "mean_calibrated_threshold": round(float(np.mean(moved)), 4),
                "detection_shipped": round(float(np.mean(det0)), 3), "detection_calibrated": round(float(np.mean(det1)), 3),
                "fa_per_hour_shipped": round(base_fa, 2), "fa_per_hour_calibrated": round(float(np.mean(fa1)), 2),
            }
        print(f"fold {f}: {len(train_c)} team clips in training, {hours:.2f} h background", flush=True)

    rows = list(results.values())
    w = np.array([r["clips"] - CALIBRATION_RECORDINGS for r in rows], float)
    summary = {
        "sessions_x_classes": len(rows),
        "detection_shipped": round(float(np.average([r["detection_shipped"] for r in rows], weights=w)), 3),
        "detection_calibrated": round(float(np.average([r["detection_calibrated"] for r in rows], weights=w)), 3),
        "added_fa_per_hour_mean": round(float(np.mean([r["fa_per_hour_calibrated"] - r["fa_per_hour_shipped"] for r in rows])), 2),
    }
    out = {"run": run_dir.name, "folds": args.folds, "draws": DRAWS, "calibration_recordings": CALIBRATION_RECORDINGS,
           "summary": summary, "per_session_class": {f"{s}|{c}": v for (s, c), v in results.items()}}
    (run_dir / "home_calibration.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"{'session':34s} {'class':15s} clips  shipped -> calibrated   FA/h shipped -> calibrated")
    for (s, c), r in results.items():
        print(f"{s:34s} {c:15s} {r['clips']:5d}  {r['detection_shipped']:.2f} -> {r['detection_calibrated']:.2f}"
              f"            {r['fa_per_hour_shipped']:.2f} -> {r['fa_per_hour_calibrated']:.2f}")
    print("summary", summary)


if __name__ == "__main__":
    main()
