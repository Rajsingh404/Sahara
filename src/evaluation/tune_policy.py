"""Tune each class's alert threshold and hits-required against a false-alarm budget (validation only).

Writes data/metadata/class_thresholds.json (the app reads `threshold` and `hits_required` per class),
data/metadata/alert_profiles.csv (the same tuning at each SENSITIVITY_PROFILE_BUDGETS budget: one row per
budget and class, read by the app's sensitivity slider) and models/logs/<run>/policy_tuning.json with the
full detection / false-alarm trade-off.
See src/evaluation/policy_tuning.py for the method.

    python src/evaluation/tune_policy.py [--run <run folder>] [--budget 2.0]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import FALSE_ALARM_BUDGET_PER_HOUR, METADATA_DIR, SENSITIVITY_PROFILE_BUDGETS
from src.evaluation import policy_tuning as pt
from src.evaluation.streaming import load_head, selected_run_dir


def run(run: str | None = None, budget: float = FALSE_ALARM_BUDGET_PER_HOUR) -> dict:
    run_dir = selected_run_dir(run)
    clips = pt.load_split_clip_scores(load_head(run_dir), "val")
    curves = pt.class_curves(clips, pt.grid())
    settings = pt.choose_settings(curves, pt.grid(), budget)
    hours = max(cv["negative_hours"] for cv in curves.values())
    payload = {**settings, "_meta": {
        "run": run_dir.name, "split": "val", "method": "policy-aware (src/evaluation/policy_tuning.py)",
        "false_alarm_budget_per_hour": budget,
        "val_total_false_alarms_per_hour": round(sum(s["val_false_alarms_per_hour"] for s in settings.values()), 3),
        "val_clips": len(clips), "val_negative_hours_max": round(hours, 3)}}
    text = json.dumps(payload, indent=2) + "\n"
    (METADATA_DIR / "class_thresholds.json").write_text(text, encoding="utf-8")
    (run_dir / "class_thresholds.json").write_text(text, encoding="utf-8")
    lines = ["budget_per_hour,val_false_alarms_per_hour,val_mean_detection,class,threshold,hits_required"]
    for b in SENSITIVITY_PROFILE_BUDGETS:
        s = pt.choose_settings(curves, pt.grid(), b)
        fa = round(sum(v["val_false_alarms_per_hour"] for v in s.values()), 3)
        det = round(sum(v["val_clip_detection"] for v in s.values()) / len(s), 4)
        lines += [f"{b},{fa},{det},{c},{v['threshold']},{v['hits_required']}" for c, v in s.items()]
    for out in (METADATA_DIR / "alert_profiles.csv", run_dir / "alert_profiles.csv"):
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (run_dir / "policy_tuning.json").write_text(json.dumps({"budget_per_hour": budget, "tradeoff": pt.tradeoff(curves)},
                                                           indent=2, default=list))
    print(f"{run_dir.name}: budget {budget}/h on validation -> total {payload['_meta']['val_total_false_alarms_per_hour']}/h")
    print("class            thr     hits  val clip detection  val FA/h")
    for c, s in settings.items():
        print(f"{c:16s} {s['threshold']:.3f}   {s['hits_required']}     {s['val_clip_detection']:.3f} ({s['val_clips']} clips)   "
              f"{s['val_false_alarms_per_hour']:.2f}")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run")
    parser.add_argument("--budget", type=float, default=FALSE_ALARM_BUDGET_PER_HOUR)
    args = parser.parse_args()
    run(args.run, args.budget)
