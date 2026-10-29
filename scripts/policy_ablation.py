"""Same model, different alert rules, on the real test split: what the policy tuning buys.

Rules compared (all with the app's 10 s cooldown and 3-window memory):
  A  threshold 0.5, alert on 1 window                                  (no smoothing, no tuning)
  B  threshold 0.5, 1 window for safety classes, 2 of 3 for the rest   (the original fixed design)
  C  per-class threshold and windows tuned on validation at 4 false alarms/h (shipped)

Per class: share of test clips that raise their alert, and false alarms per hour on the test audio
that does not contain the class (background and other classes' clips, played back to back).
Writes models/logs/<run>/policy_ablation.json.

    python scripts/policy_ablation.py [--run <run folder>]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import SAFETY_CRITICAL_CLASSES, SOUND_CLASSES  # noqa: E402
from src.evaluation import policy_tuning as pt  # noqa: E402
from src.evaluation.streaming import HOP_MS, load_head, load_hits, load_thresholds, selected_run_dir  # noqa: E402


def score(clips, rule: dict[str, tuple[float, int]]) -> dict:
    per_class, total_fa = {}, 0.0
    for i, c in enumerate(SOUND_CLASSES):
        t, k = rule[c]
        pos = [cl.scores[:, i] for cl in clips if c in cl.labels]
        neg = np.concatenate([cl.scores[:, i] for cl in clips if c not in cl.labels])
        hours = len(neg) * HOP_MS / 3_600_000
        det = float(np.mean([pt.count_alerts(s, t, k) > 0 for s in pos])) if pos else None
        fa = pt.count_alerts(neg, t, k) / hours
        total_fa += fa
        per_class[c] = {"threshold": t, "hits": k, "clips": len(pos), "detection": det, "fa_per_hour": round(fa, 2)}
    dets = [v["detection"] for v in per_class.values() if v["detection"] is not None]
    safety = [per_class[c]["detection"] for c in SAFETY_CRITICAL_CLASSES]
    return {"mean_detection": round(float(np.mean(dets)), 3), "safety_detection": round(float(np.mean(safety)), 3),
            "total_fa_per_hour": round(total_fa, 1), "per_class": per_class}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run")
    args = ap.parse_args()
    run_dir = selected_run_dir(args.run)
    clips = pt.load_split_clip_scores(load_head(run_dir), "test")
    thr, hits = load_thresholds(), load_hits()
    rules = {
        "A_threshold_0.5_one_window": {c: (0.5, 1) for c in SOUND_CLASSES},
        "B_threshold_0.5_fixed_k": {c: (0.5, 1 if c in SAFETY_CRITICAL_CLASSES else 2) for c in SOUND_CLASSES},
        "C_tuned_shipped": {c: (float(thr[c]), int(hits[c])) for c in SOUND_CLASSES},
    }
    hours = sum(len(cl.scores) for cl in clips) * HOP_MS / 3_600_000
    out = {"run": run_dir.name, "split": "test", "clips": len(clips), "audio_hours": round(hours, 2),
           "rules": {name: score(clips, r) for name, r in rules.items()}}
    (run_dir / "policy_ablation.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"{run_dir.name}: {len(clips)} real test clips, {hours:.2f} h")
    print(f"{'rule':30s} {'mean clips alerted':>18s} {'safety':>7s} {'false alarms/h':>15s}")
    for name, r in out["rules"].items():
        print(f"{name:30s} {r['mean_detection']:18.3f} {r['safety_detection']:7.3f} {r['total_fa_per_hour']:15.1f}")


if __name__ == "__main__":
    main()
