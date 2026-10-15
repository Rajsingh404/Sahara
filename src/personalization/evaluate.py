"""Evaluate few-shot personalization on the team's "Aryan" name-call recordings.

Protocol (repeated for ROUNDS random enrolments, for K = 5 and 10 enrolment samples):
  enrol  K train-side "Aryan" clips -> prototype (prototypes.build_prototype)
  tune   on the train side only: the lowest threshold whose alerts on validation background audio
         stay <= PERSONAL_FA_BUDGET_PER_HOUR (other train-side "Aryan" clips and the train-side
         other-name clips are reported for the dev curve)
  test   test-side "Aryan" clips (detection), test-side other names (false accepts) and test-split
         background clips played back to back (false alerts per hour); clip score = best window
An alert = a window at/above the threshold, at most one per 10 s (the app's cooldown).
Second experiment (``run_custom_sounds``): the same method on distinctive household sounds that are
not among the 8 classes (FSD50K clips of e.g. toilet flush, hiss, drill, water tap): enrol 5 train-split
clips, tune on validation background without that sound, test on test-split clips and background.
Writes models/logs/personalization/aryan_eval.json and custom_sounds_eval.json.

    python src/personalization/evaluate.py
"""
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import METADATA_DIR, MODELS_DIR, RANDOM_SEED, RAW_DATA_DIR
from src.evaluation.policy_tuning import count_alerts
from src.evaluation.streaming import HOP_MS
from src.config import PROCESSED_DATA_DIR
from src.personalization.prototypes import background_center, build_prototype, set_center, window_similarities

from src.preprocessing.build_features_cache import build_hop_cache
from src.preprocessing.features import hop_cache_path, load_hop_cache

PERSONAL_FA_BUDGET_PER_HOUR = 1.0
ROUNDS = 20
SHOTS = (5, 10)
GRID = np.round(np.arange(-0.20, 0.995, 0.005), 3)  # centred cosine can be low or negative
ARYAN_DIR = RAW_DATA_DIR / "personalization" / "aryan"


def use_center(enabled: bool = True) -> np.ndarray | None:
    """Centre embeddings on the mean training background window (or switch centring off)."""
    if not enabled:
        set_center(None)
        return None
    center = background_center(np.load(PROCESSED_DATA_DIR / "train_embeddings.npy", mmap_mode="r"),
                               np.load(PROCESSED_DATA_DIR / "train_labels.npy"))
    set_center(center)
    return center


def aryan_clips() -> list[dict]:
    clips = []
    for sub, positive in (("positive", True), ("negative", False)):
        with (ARYAN_DIR / sub / "metadata.csv").open(newline="", encoding="utf-8") as handle:
            for r in csv.DictReader(handle):
                clips.append({"path": str((ARYAN_DIR / sub / r["filename"]).resolve()), "positive": positive,
                              "split": r["split"], "recorder": r["recorder"]})
    return clips


def background_paths(split: str) -> list[str]:
    with (METADATA_DIR / f"{split}_manifest.csv").open(newline="", encoding="utf-8") as handle:
        return sorted({r["filepath"] for r in csv.DictReader(handle)
                       if r["label"] == "background" and r["source_type"] != "synthetic_mixed"})


def clip_score(hops, proto) -> float:
    sims, _ = window_similarities(hops, proto)
    return float(sims.max()) if len(sims) else -1.0


def stream_similarities(hop_list, proto) -> np.ndarray:
    """Per-hop similarity over clips played back to back (silent hops = -1, never an alert)."""
    parts = []
    for hops in hop_list:
        s = np.full(len(hops["vad"]), -1.0, np.float32)
        sims, idx = window_similarities(hops, proto)
        s[idx] = sims
        parts.append(s)
    return np.concatenate(parts) if parts else np.zeros(0, np.float32)


def alerts_per_hour(stream: np.ndarray, threshold: float) -> float:
    hours = len(stream) * HOP_MS / 3_600_000
    return count_alerts(stream, threshold, 1) / hours if hours else 0.0


def eer(pos: np.ndarray, neg: np.ndarray) -> float:
    """Equal error rate between positive and negative clip scores."""
    best = 1.0
    for t in np.unique(np.concatenate([pos, neg])):
        frr, far = float((pos < t).mean()), float((neg >= t).mean())
        best = min(best, max(frr, far))
    return best


def run() -> dict:
    clips = aryan_clips()
    bg_val, bg_test = background_paths("val"), background_paths("test")
    build_hop_cache([(c["path"], "") for c in clips])
    hops = {c["path"]: load_hop_cache(hop_cache_path(c["path"])) for c in clips}
    val_bg = [load_hop_cache(hop_cache_path(p)) for p in bg_val if hop_cache_path(p).exists()]
    test_bg = [load_hop_cache(hop_cache_path(p)) for p in bg_test if hop_cache_path(p).exists()]
    train_pos = [c for c in clips if c["positive"] and c["split"] == "train"]
    train_neg = [c for c in clips if not c["positive"] and c["split"] == "train"]
    test_pos = [c for c in clips if c["positive"] and c["split"] == "test"]
    test_neg = [c for c in clips if not c["positive"] and c["split"] == "test"]

    results = {"budget_per_hour": PERSONAL_FA_BUDGET_PER_HOUR, "rounds": ROUNDS,
               "clips": {"train_pos": len(train_pos), "train_neg": len(train_neg),
                         "test_pos": len(test_pos), "test_neg": len(test_neg)},
               "background_hours": {"val": round(sum(len(h["vad"]) for h in val_bg) * HOP_MS / 3.6e6, 2),
                                    "test": round(sum(len(h["vad"]) for h in test_bg) * HOP_MS / 3.6e6, 2)},
               "shots": {}}
    for k in SHOTS:
        rng = random.Random(f"{RANDOM_SEED}-{k}")
        rounds = []
        for _ in range(ROUNDS):
            enrol = rng.sample(train_pos, k)
            proto = build_prototype([hops[c["path"]] for c in enrol])
            # Threshold on the train side only: lowest one meeting the background budget.
            val_stream = stream_similarities(val_bg, proto)
            ok = [t for t in GRID if alerts_per_hour(val_stream, t) <= PERSONAL_FA_BUDGET_PER_HOUR]
            thr = float(min(ok)) if ok else float(GRID[-1])
            dev_pos = np.array([clip_score(hops[c["path"]], proto) for c in train_pos if c not in enrol])
            dev_neg = np.array([clip_score(hops[c["path"]], proto) for c in train_neg])
            # Test.
            pos = np.array([clip_score(hops[c["path"]], proto) for c in test_pos])
            neg = np.array([clip_score(hops[c["path"]], proto) for c in test_neg])
            bg_clip_scores = np.array([clip_score(h, proto) for h in test_bg])
            rounds.append({
                "threshold": thr,
                "dev_detection": float((dev_pos >= thr).mean()),
                "dev_other_name_accepts": float((dev_neg >= thr).mean()),
                "test_detection": float((pos >= thr).mean()),
                "test_other_name_accepts": float((neg >= thr).mean()),
                "test_background_alerts_per_hour": alerts_per_hour(stream_similarities(test_bg, proto), thr),
                "test_eer_name_vs_other_names": eer(pos, neg),
                "test_eer_name_vs_background_clips": eer(pos, bg_clip_scores),
                "mean_score": {"aryan": float(pos.mean()), "other_names": float(neg.mean()),
                               "background_clips": float(bg_clip_scores.mean())},
            })

        def agg(key):
            vals = np.array([r[key] for r in rounds], dtype=float)
            return {"mean": round(float(vals.mean()), 4), "min": round(float(vals.min()), 4),
                    "max": round(float(vals.max()), 4)}

        results["shots"][str(k)] = {
            "summary": {key: agg(key) for key in ("threshold", "test_detection", "test_other_name_accepts",
                                                   "test_background_alerts_per_hour", "test_eer_name_vs_other_names",
                                                   "test_eer_name_vs_background_clips", "dev_detection",
                                                   "dev_other_name_accepts")},
            "mean_scores": {g: round(float(np.mean([r["mean_score"][g] for r in rounds])), 4)
                            for g in ("aryan", "other_names", "background_clips")},
            "rounds": rounds,
        }
        s = results["shots"][str(k)]["summary"]
        print(f"K={k:2d}: threshold {s['threshold']['mean']:.3f} | Aryan detected {s['test_detection']['mean']:.0%} "
              f"(range {s['test_detection']['min']:.0%}-{s['test_detection']['max']:.0%}) | other names accepted "
              f"{s['test_other_name_accepts']['mean']:.0%} | background {s['test_background_alerts_per_hour']['mean']:.2f}/h | "
              f"EER vs other names {s['test_eer_name_vs_other_names']['mean']:.2f}, vs background "
              f"{s['test_eer_name_vs_background_clips']['mean']:.2f}")
        print(f"      mean similarity: Aryan {results['shots'][str(k)]['mean_scores']['aryan']:.3f}, "
              f"other names {results['shots'][str(k)]['mean_scores']['other_names']:.3f}, "
              f"background {results['shots'][str(k)]['mean_scores']['background_clips']:.3f}")
    out = MODELS_DIR / "logs" / "personalization"
    out.mkdir(parents=True, exist_ok=True)
    (out / "aryan_eval.json").write_text(json.dumps(results, indent=2))
    return results


CUSTOM_SOUNDS = ("Toilet_flush", "Hiss", "Drill", "Water_tap_and_faucet", "Keys_jangling", "Printer", "Cough",
                 "Zipper_(clothing)")


def _fsd_labels() -> dict[str, list[str]]:
    from src.config import INTERIM_DATA_DIR

    labels: dict[str, list[str]] = {}
    for split in ("dev", "eval"):
        with (INTERIM_DATA_DIR / "fsd50k_ground_truth" / f"{split}.csv").open(newline="", encoding="utf-8") as handle:
            for r in csv.DictReader(handle):
                labels[r["fname"]] = r["labels"].split(",")
    return labels


def run_custom_sounds(shots: int = 5) -> dict:
    """Few-shot prototypes for household sounds outside the 8 classes (FSD50K background clips)."""
    labels = _fsd_labels()
    split_clips: dict[str, list[tuple[str, list[str]]]] = {}
    for split in ("train", "val", "test"):
        rows = []
        for path in background_paths(split):
            if hop_cache_path(path).exists():
                rows.append((path, labels.get(Path(path).stem, []) if "FSD50K" in path else []))
        split_clips[split] = rows
    cache = {p: load_hop_cache(hop_cache_path(p)) for rows in split_clips.values() for p, _ in rows}
    results = {"shots": shots, "rounds": ROUNDS, "budget_per_hour": PERSONAL_FA_BUDGET_PER_HOUR, "sounds": {}}
    for sound in CUSTOM_SOUNDS:
        enrol_pool = [p for p, lbls in split_clips["train"] if sound in lbls]
        test_pos = [p for p, lbls in split_clips["test"] if sound in lbls]
        val_bg = [cache[p] for p, lbls in split_clips["val"] if sound not in lbls]
        test_bg = [cache[p] for p, lbls in split_clips["test"] if sound not in lbls]
        rng = random.Random(f"{RANDOM_SEED}-{sound}")
        rounds = []
        for _ in range(ROUNDS):
            proto = build_prototype([cache[p] for p in rng.sample(enrol_pool, shots)])
            val_stream = stream_similarities(val_bg, proto)
            ok = [t for t in GRID if alerts_per_hour(val_stream, t) <= PERSONAL_FA_BUDGET_PER_HOUR]
            thr = float(min(ok)) if ok else float(GRID[-1])
            pos = np.array([clip_score(cache[p], proto) for p in test_pos])
            bg_scores = np.array([clip_score(h, proto) for h in test_bg])
            rounds.append({"threshold": thr, "detection": float((pos >= thr).mean()),
                           "background_alerts_per_hour": alerts_per_hour(stream_similarities(test_bg, proto), thr),
                           "eer_vs_background_clips": eer(pos, bg_scores)})
        summary = {k: round(float(np.mean([r[k] for r in rounds])), 4) for k in rounds[0]}
        results["sounds"][sound] = {"enrol_pool": len(enrol_pool), "test_clips": len(test_pos), **summary}
        print(f"{sound:24s} test clips {len(test_pos):3d}: detected {summary['detection']:.0%}, background "
              f"{summary['background_alerts_per_hour']:.2f}/h, EER vs background {summary['eer_vs_background_clips']:.2f}",
              flush=True)
    det = [v["detection"] for v in results["sounds"].values()]
    results["mean_detection"] = round(float(np.mean(det)), 4)
    out = MODELS_DIR / "logs" / "personalization"
    out.mkdir(parents=True, exist_ok=True)
    (out / "custom_sounds_eval.json").write_text(json.dumps(results, indent=2))
    print(f"mean detection over {len(det)} household sounds: {results['mean_detection']:.0%}")
    return results


if __name__ == "__main__":
    use_center(True)  # the app centres too (assets/personal_center.bin)
    run()
    run_custom_sounds()
