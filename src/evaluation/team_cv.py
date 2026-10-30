"""Indian-context evaluation by 5-fold cross-validation over all team recordings.

The fixed team test side has only 2-6 clips per class, too few to judge. Here every team clip
(train and test side) is tested exactly once by a model that never heard it:

  1. base   = the selected run's recipe trained on public training data only (no team clips)
  2. folds  = team clips split into K folds per label (seeded; identical audio kept together)
  3. per fold: fine-tune the base on the other folds' team windows + a replay of public windows
     (as train.py --finetune-from does), then score the held-out fold's clips

Reported for "base" (no Indian data) and "fine-tuned": per class window AP and clip detection
with the app policy (thresholds and hits from class_thresholds.json), plus false alarms per hour
on the team background clips played back to back. Writes models/logs/<run>/team_cv.json.

    python src/evaluation/team_cv.py [--run <run folder>] [--folds 5]
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import METADATA_DIR, RANDOM_SEED, SOUND_CLASSES
from src.evaluation.policy import DetectionPolicy
from src.evaluation.streaming import load_hits, load_thresholds, run_policy, selected_run_dir
from src.models.classifier_head import HEAD_DEVICE, build_classifier_head, head_predict
from src.preprocessing.features import hop_cache_path, load_hop_cache
from src.preprocessing.windowing import clip_examples, device_window_vectors
from src.training.losses import weighted_binary_crossentropy
from src.training.train import FINETUNE_REPLAY_RATIO, balanced_sample_weights, load_split


def team_clips() -> list[dict]:
    clips: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        with (METADATA_DIR / f"{split}_manifest.csv").open(newline="", encoding="utf-8") as handle:
            for r in csv.DictReader(handle):
                if r["source_type"] != "self_recorded":
                    continue
                c = clips.setdefault(r["filepath"], {"filepath": r["filepath"], "labels": set(), "group": r["recording_id"]})
                if r["label"] in SOUND_CLASSES:
                    c["labels"].add(r["label"])
    return sorted(clips.values(), key=lambda c: c["filepath"])


def session_groups() -> dict[str, str]:
    """Recording filename -> inferred session (scripts/organise_recordings.py: session_group)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from organise_recordings import load_label_map, session_group

    return {r["filename"]: session_group(r) for r in load_label_map() if r["dataset"] == "ambient"}


def assign_session_folds(clips: list[dict], k: int) -> dict[str, int]:
    """Whole sessions per fold (largest first, into the emptiest fold), so no sitting is on both sides."""
    groups = session_groups()
    by_group: dict[str, list[dict]] = defaultdict(list)
    for c in clips:
        by_group[groups[Path(c["filepath"]).name]].append(c)
    load, folds = [0] * k, {}
    for g in sorted(by_group, key=lambda g: (-len(by_group[g]), g)):
        f = load.index(min(load))
        load[f] += len(by_group[g])
        for c in by_group[g]:
            folds[c["filepath"]] = f
    return folds


def assign_folds(clips: list[dict], k: int, seed: int = RANDOM_SEED) -> dict[str, int]:
    """Per label, deal clips round-robin into k folds after a seeded shuffle."""
    by_label: dict[str, list[dict]] = defaultdict(list)
    for c in clips:
        by_label[min(c["labels"]) if c["labels"] else "background"].append(c)
    folds = {}
    for label in sorted(by_label):
        group = by_label[label]
        random.Random(f"{seed}-{label}").shuffle(group)
        for i, c in enumerate(group):
            folds[c["filepath"]] = i % k
    return folds


def _fit(model, X, Y, recipe: dict, epochs: int, lr: float, team_mask=None, team_weight: float = 1.0):
    with tf.device(HEAD_DEVICE):
        model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=weighted_binary_crossentropy)
        weights = balanced_sample_weights(Y, recipe.get("balance_power", 0.5), recipe.get("background_weight", 1.0),
                                          hard=team_mask, hard_weight=team_weight)
        model.fit(X, Y, sample_weight=weights, batch_size=recipe.get("batch_size", 128), epochs=epochs, verbose=0)
    return model


def _clone(model):
    with tf.device(HEAD_DEVICE):
        copy = build_classifier_head()  # the base is a plain head; copy its weights into a fresh one
        copy.set_weights(model.get_weights())
    return copy


def evaluate(model_for, clips: list[dict], thresholds: dict, hits: dict) -> dict:
    """Score each clip with ``model_for(clip)`` and compute the metrics once over all clips."""
    from sklearn.metrics import average_precision_score

    win_scores, win_labels = [], []
    detected, total, wrong = defaultdict(int), defaultdict(int), defaultdict(int)
    background_stream = []
    for c in clips:
        model = model_for(c)
        hops = load_hop_cache(hop_cache_path(c["filepath"]))
        scores = np.zeros((len(hops["vad"]), len(SOUND_CLASSES)), np.float32)
        win = device_window_vectors(hops)
        if len(win["vectors"]):
            scores[win["hop_index"]] = head_predict(model, win["vectors"])
        x, y, _ = clip_examples(hops, c["labels"])
        if len(x):
            win_scores.append(head_predict(model, x))
            win_labels.append(y)
        if not c["labels"]:
            background_stream.append(scores)
            continue
        fired = {d.label for d in run_policy(DetectionPolicy(SOUND_CLASSES, thresholds, hits_per_class=hits), scores)[0]}
        for lbl in c["labels"]:
            total[lbl] += 1
            detected[lbl] += lbl in fired
        for lbl in fired - c["labels"]:
            wrong[lbl] += 1
    S, Y = np.concatenate(win_scores), np.concatenate(win_labels)
    bg = np.concatenate(background_stream) if background_stream else np.zeros((0, len(SOUND_CLASSES)))
    bg_alerts, t_end = run_policy(DetectionPolicy(SOUND_CLASSES, thresholds, hits_per_class=hits), bg)
    hours = t_end / 3_600_000
    per_class = {}
    for i, c in enumerate(SOUND_CLASSES):
        if Y[:, i].sum() == 0:
            continue
        per_class[c] = {"ap": round(float(average_precision_score(Y[:, i], S[:, i])), 4),
                        "clips": total[c], "detected": detected[c],
                        "detection": round(detected[c] / total[c], 4) if total[c] else None}
    return {
        "mAP": round(float(np.mean([v["ap"] for v in per_class.values()])), 4),
        "per_class": per_class,
        "wrong_class_alerts_on_target_clips": dict(wrong),
        "background_clips": len(background_stream),
        "background_hours": round(hours, 4),
        "background_false_alarms": len(bg_alerts),
        "background_false_alarms_per_hour": round(len(bg_alerts) / hours, 2) if hours else None,
    }


def run(run: str | None = None, k: int = 5, mode: str = "finetune", team_weight: float = 1.0,
        finetune_epochs: int | None = None, group_by: str = "clip") -> dict:
    """mode="finetune": base + per-fold fine-tune; mode="joint": per fold, train from scratch on public
    + other-fold team windows. Team windows are weighted x``team_weight`` in both modes."""
    run_dir = out_dir = selected_run_dir(run)
    cfg = json.loads((run_dir / "config.json").read_text())
    if cfg.get("ensemble_of"):
        # An ensemble's members share one recipe; cross-validate that recipe with a single model.
        run_dir = run_dir.parent / cfg["ensemble_of"][0]
        cfg = json.loads((run_dir / "config.json").read_text())
    recipe = cfg["args"]
    base_dir = run_dir if not recipe.get("finetune_from") else Path(recipe["finetune_from"])
    base_recipe = json.loads((base_dir / "config.json").read_text())["args"] if (base_dir / "config.json").exists() else recipe
    thresholds, hits = load_thresholds(), load_hits()

    X, Y, meta = load_split("train")
    st = meta["source_type"]
    public = st == "real" if not base_recipe.get("include_synthetic") else st != "self_recorded"
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    base_epochs = int(json.loads((base_dir / "metrics.json").read_text()).get("best_epoch") or 5)
    with tf.device(HEAD_DEVICE):
        base = build_classifier_head()
    base = _fit(base, X[public], Y[public], base_recipe, base_epochs, base_recipe.get("learning_rate", 3e-4))
    print(f"base: {int(public.sum())} public windows, {base_epochs} epochs")

    clips = team_clips()
    folds = assign_session_folds(clips, k) if group_by == "session" else assign_folds(clips, k)
    rng = np.random.default_rng(RANDOM_SEED)
    public_idx = np.flatnonzero(public)
    finetune_epochs = finetune_epochs or int(json.loads((run_dir / "metrics.json").read_text()).get("best_epoch") or 2)
    models = {}
    for f in range(k):
        train_c = [c for c in clips if folds[c["filepath"]] != f]
        xs, ys = [], []
        for c in train_c:
            x, y, _ = clip_examples(load_hop_cache(hop_cache_path(c["filepath"])), c["labels"])
            xs.append(x)
            ys.append(y)
        Xt, Yt = np.concatenate(xs), np.concatenate(ys)
        if mode == "joint":
            Xf, Yf = np.concatenate([Xt, X[public_idx]]), np.concatenate([Yt, Y[public_idx]])
            mask = np.arange(len(Xf)) < len(Xt)
            tf.keras.utils.set_random_seed(RANDOM_SEED + f)
            with tf.device(HEAD_DEVICE):
                fresh = build_classifier_head()
            models[f] = _fit(fresh, Xf, Yf, base_recipe, base_epochs, base_recipe.get("learning_rate", 3e-4), mask, team_weight)
        else:
            replay = rng.choice(public_idx, size=min(len(public_idx), FINETUNE_REPLAY_RATIO * len(Xt)), replace=False)
            Xf, Yf = np.concatenate([Xt, X[replay]]), np.concatenate([Yt, Y[replay]])
            mask = np.arange(len(Xf)) < len(Xt)
            models[f] = _fit(_clone(base), Xf, Yf, recipe, finetune_epochs, recipe.get("learning_rate", 1e-4), mask,
                             team_weight)
        print(f"fold {f}: fine-tuned on {len(train_c)} team clips, held out {len(clips) - len(train_c)}")

    # Every team clip is scored by a model that never heard it.
    results = {
        "base": evaluate(lambda c: base, clips, thresholds, hits),
        "finetuned": evaluate(lambda c: models[folds[c["filepath"]]], clips, thresholds, hits),
    }
    out = {"run": out_dir.name, "folds": k, "group_by": group_by, "team_clips": len(clips), "mode": mode,
           "team_weight": team_weight,
           "finetune_epochs": finetune_epochs if mode == "finetune" else None,
           "note": "Base model trained on public data only; fine-tuned scores are out-of-fold.", **results}
    out["recipe_from"] = run_dir.name
    (out_dir / ("team_cv.json" if group_by == "clip" else f"team_cv_{group_by}.json")).write_text(json.dumps(out, indent=2))
    print(f"mode {mode}, team weight x{team_weight}")
    for name in ("base", "finetuned"):
        r = results[name]
        print(f"\n{name}: mAP {r['mAP']}, background {r['background_false_alarms']} alerts in {r['background_hours']} h")
        for c, v in r["per_class"].items():
            print(f"  {c:15s} AP {v['ap']:.3f}  detected {v['detected']}/{v['clips']}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--mode", choices=("finetune", "joint"), default="finetune")
    parser.add_argument("--team-weight", type=float, default=1.0)
    parser.add_argument("--finetune-epochs", type=int)
    parser.add_argument("--group-by", choices=("clip", "session"), default="clip",
                        help="session: hold out whole recording sessions, so no sitting is on both sides.")
    args = parser.parse_args()
    run(args.run, args.folds, args.mode, args.team_weight, args.finetune_epochs, args.group_by)
