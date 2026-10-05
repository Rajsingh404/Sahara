"""Compare recorded training runs on validation and pick one (never on test).

Rule (build plan, Phase 3): highest validation mAP among runs where every safety-critical
class reaches precision >= SAFETY_PRECISION_FLOOR at some threshold with recall > 0.
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
from src.config import MODELS_DIR, SAFETY_CRITICAL_CLASSES, SOUND_CLASSES
from src.training.train import validation_fingerprint


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
        for c in SOUND_CLASSES:
            row[f"AP_{c}"] = round(m["per_class_at_0.5"].get(c, {}).get("ap", float("nan")), 3)
        for c in SAFETY_CRITICAL_CLASSES:
            row[f"floorR_{c}"] = round(m["safety_floor"][c]["recall"], 3)
        rows.append(row)
    if not rows:
        print("No runs with metrics.json under", logs)
        return None
    with (logs / "runs_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    eligible = [r for r in rows if r["safety_floor_met"] and r["same_val_data"]]
    best = max(eligible, key=lambda r: r["val_mAP"]) if eligible else None
    for r in sorted(rows, key=lambda r: (not r["same_val_data"], -r["val_mAP"])):
        mark = "  <= selected" if best and r["run"] == best["run"] else ""
        if not r["same_val_data"]:
            mark = "  (different validation data, not compared)"
        print(f"{r['run']:45s} val mAP {r['val_mAP']:.3f}  floor {'ok' if r['safety_floor_met'] else 'MISSED'}{mark}")
    if best:
        (logs / "selected_run.txt").write_text(best["run"] + "\n")
        return logs / best["run"]
    return None


if __name__ == "__main__":
    main()
