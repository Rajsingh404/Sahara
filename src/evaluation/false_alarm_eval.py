"""False alarms per hour on continuous background audio, with the app's exact alert policy.

Test-split background clips are played back to back: every 0.5 s hop goes through VAD, the
2-embedding pool, the head and the ported DetectionPolicy (2-of-3 windows, 1 hit for
safety-critical classes, 10 s cooldown, thresholds from class_thresholds.json). Policy and pool
state carry over from clip to clip, as on a phone that keeps listening. Every alert on
background audio is a false alarm. Reported separately for public background (DESED, ESC-50,
FSD50K) and the team's Indian background recordings.

    python src/evaluation/false_alarm_eval.py [--run <run folder>]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import METADATA_DIR, SOUND_CLASSES, WINDOW_POOLS
from src.evaluation.policy import DetectionPolicy
from src.models.classifier_head import head_predict
from src.evaluation.streaming import HOP_MS, load_head, load_hits, load_thresholds, run_policy, selected_run_dir
from src.preprocessing.features import hop_cache_path, load_hop_cache
from src.preprocessing.windowing import device_window_vectors


def _background_clips(split: str) -> dict[str, list[str]]:
    with (METADATA_DIR / f"{split}_manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r["label"] == "background"]
    groups: dict[str, list[str]] = {"public": [], "indian_team": []}
    for r in sorted(rows, key=lambda r: r["filepath"]):
        groups["indian_team" if r["source_type"] == "self_recorded" else "public"].append(r["filepath"])
    return groups


def stream_false_alarms(model, filepaths: list[str], thresholds: dict[str, float]) -> dict:
    policy = DetectionPolicy(SOUND_CLASSES, thresholds, hits_per_class=load_hits())
    pool_carry: dict | None = None
    t_ms, alerts, clips = 0, [], 0
    for fp in filepaths:
        path = hop_cache_path(fp)
        if not path.exists():
            continue
        hops = load_hop_cache(path)
        carried = 0
        if pool_carry is not None:
            # Carry the phone's pool across the clip boundary: prepend the last non-silent hops.
            carried = len(pool_carry["vad"])
            hops = {k: np.concatenate([pool_carry[k], hops[k]]) for k in hops}
        scores = np.zeros((len(hops["vad"]), len(SOUND_CLASSES)), np.float32)
        win = device_window_vectors(hops)
        if len(win["vectors"]):
            scores[win["hop_index"]] = head_predict(model, win["vectors"])
        scores = scores[carried:]  # carried hops were already scored in the previous clip
        detections, t_ms = run_policy(policy, scores, t_ms)
        alerts.extend((d.label, fp, d.timestamp_ms) for d in detections)
        loud = np.flatnonzero(hops["vad"])
        keep = loud[-(max(WINDOW_POOLS) - 1):] if max(WINDOW_POOLS) > 1 else loud[:0]
        pool_carry = {k: v[keep] for k, v in hops.items()} if keep.size else None
        clips += 1
    hours = t_ms / 3_600_000
    per_class = Counter(label for label, _, _ in alerts)
    return {
        "clips": clips,
        "hours": round(hours, 3),
        "false_alarms": len(alerts),
        "false_alarms_per_hour": round(len(alerts) / hours, 2) if hours else None,
        "per_class": {c: {"count": per_class.get(c, 0),
                          "per_hour": round(per_class.get(c, 0) / hours, 2) if hours else None} for c in SOUND_CLASSES},
        "examples": [{"class": c, "file": Path(f).name, "t_s": round(t / 1000, 1)} for c, f, t in alerts[:25]],
        "hop_ms": HOP_MS,
    }


def run(run: str | None = None, split: str = "test") -> dict:
    run_dir = selected_run_dir(run)
    model = load_head(run_dir)
    thresholds = load_thresholds()
    result = {"run": run_dir.name, "split": split, "thresholds": thresholds, "sets": {}}
    for name, files in _background_clips(split).items():
        result["sets"][name] = stream_false_alarms(model, files, thresholds)
        r = result["sets"][name]
        print(f"{name:12s} {r['clips']:5d} clips, {r['hours']:.2f} h: {r['false_alarms']} false alarms "
              f"= {r['false_alarms_per_hour']} / hour")
        busiest = sorted(r["per_class"].items(), key=lambda kv: -kv[1]["count"])[:3]
        print("             top classes:", ", ".join(f"{c} {v['count']}" for c, v in busiest if v["count"]))
    (run_dir / "false_alarms.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", help="Run folder name or path (default: models/logs/selected_run.txt)")
    parser.add_argument("--split", default="test", choices=("val", "test"))
    args = parser.parse_args()
    run(args.run, args.split)
