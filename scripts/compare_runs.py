"""Compare recorded training runs on validation and pick one (never on test).

Rule: highest validation *policy score* among runs where every safety-critical class reaches
precision >= SAFETY_PRECISION_FLOOR at some threshold with recall > 0. The policy score is the
weighted share of validation clips that raise the right alert when the thresholds and hits are
tuned to FALSE_ALARM_BUDGET_PER_HOUR (src/evaluation/policy_tuning.py; weights =
POSITIVE_CLASS_WEIGHTS, safety classes 3). It replaced validation mAP in Phase 4 because it measures
what the app does: alerts per clip under a false-alarm limit, not window ranking.
Only runs scored on the current validation set (same val_sha256) are compared; runs from an
earlier dataset build are listed but never selected.
Writes models/logs/runs_summary.csv and models/logs/selected_run.txt.

    python scripts/compare_runs.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import (
    FALSE_ALARM_BUDGET_PER_HOUR,
    MODELS_DIR,
    POSITIVE_CLASS_WEIGHTS,
    SAFETY_CRITICAL_CLASSES,
    SOUND_CLASSES,
)
from src.training.train import validation_fingerprint


def policy_score(run_dir: Path, current_val: str, budget: float = FALSE_ALARM_BUDGET_PER_HOUR) -> dict:
    """Cached in <run>/policy_score.json; recomputed if the validation set or the budget changed."""
    cache = run_dir / "policy_score.json"
    if cache.exists():
        cached = json.loads(cache.read_text())
        if cached.get("val_sha256") == current_val and cached.get("budget_per_hour") == budget:
            return cached
    from src.evaluation import policy_tuning as pt
    from src.evaluation.streaming import load_head

    curves = pt.class_curves(pt.load_split_clip_scores(load_head(run_dir), "val"), pt.grid())
    sel = pt.choose_settings(curves, pt.grid(), budget)
    w = {c: POSITIVE_CLASS_WEIGHTS.get(c, 1.0) for c in SOUND_CLASSES}
    result = {
        "val_sha256": current_val, "budget_per_hour": budget,
        "weighted_detection": round(sum(w[c] * sel[c]["val_clip_detection"] for c in sel) / sum(w.values()), 4),
        "mean_detection": round(sum(sel[c]["val_clip_detection"] for c in sel) / len(sel), 4),
        "safety_detection": round(sum(sel[c]["val_clip_detection"] for c in SAFETY_CRITICAL_CLASSES) / 3, 4),
        "per_class": {c: sel[c] for c in sel},
    }
    cache.write_text(json.dumps(result, indent=2))
    return result


def main() -> Path | None:
    logs = MODELS_DIR / "logs"
    current_val = validation_fingerprint()
    rows = []
    for metrics_path in sorted(logs.glob("*/metrics.json")):
        m = json.loads(metrics_path.read_text())
        cfg = json.loads((metrics_path.parent / "config.json").read_text())
        row = {
            "run": m["run"],
            "val_mAP": round(m["val_mAP"], 4),
            "safety_floor_met": m["safety_floor_met"],
            "best_epoch": m["best_epoch"],
            "train_windows": m["train_windows"],
            "lr": cfg["args"]["learning_rate"],
            "batch": cfg["args"]["batch_size"],
            "synthetic": cfg["args"]["include_synthetic"],
            "finetune_from": Path(cfg["args"]["finetune_from"]).name if cfg["args"]["finetune_from"] else "",
            "features": Path(cfg["args"]["features_dir"]).name,
            "same_val_data": m.get("val_sha256") == current_val,
        }
        if row["same_val_data"] and (metrics_path.parent / "model.keras").exists():
            ps = policy_score(metrics_path.parent, current_val)
            row.update({"policy_score": ps["weighted_detection"], "mean_detection": ps["mean_detection"],
                        "safety_detection": ps["safety_detection"]})
        else:
            row.update({"policy_score": None, "mean_detection": None, "safety_detection": None})
        for c in SOUND_CLASSES:
            row[f"AP_{c}"] = round(m["per_class_at_0.5"].get(c, {}).get("ap", float("nan")), 3)
        for c in SAFETY_CRITICAL_CLASSES:
            row[f"floorR_{c}"] = round(m["safety_floor"][c]["recall"], 3)
        rows.append(row)
    if not rows:
        print("No runs with metrics.json under", logs)
        return None
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (logs / "runs_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    eligible = [r for r in rows if r["safety_floor_met"] and r["policy_score"] is not None]
    best = max(eligible, key=lambda r: r["policy_score"]) if eligible else None
    print(f"policy score = weighted clip detection on validation at {FALSE_ALARM_BUDGET_PER_HOUR} false alarms/hour")
    for r in sorted(rows, key=lambda r: (r["policy_score"] is None, -(r["policy_score"] or 0), -r["val_mAP"])):
        mark = "  <= selected" if best and r["run"] == best["run"] else ""
        if not r["same_val_data"]:
            mark = "  (different validation data, not compared)"
        ps = f"{r['policy_score']:.3f}" if r["policy_score"] is not None else "  -  "
        print(f"{r['run']:45s} policy {ps}  val mAP {r['val_mAP']:.3f}  floor {'ok' if r['safety_floor_met'] else 'MISSED'}{mark}")
    if best:
        (logs / "selected_run.txt").write_text(best["run"] + "\n")
        return logs / best["run"]
    return None


if __name__ == "__main__":
    main()
