#!/usr/bin/env python3
"""SAHARA ML pipeline, one command: data -> synthetic -> features -> train -> eval -> export -> report.

Run from the repo root with the venv active and SAHARA_DRIVE_DATASETS set (FSD50K on Drive):

    python scripts/run_full_pipeline.py                     # everything
    python scripts/run_full_pipeline.py --from train        # reuse data and features
    python scripts/run_full_pipeline.py --to features       # data + features only
    python scripts/run_full_pipeline.py --sync-assets       # also copy the final models into app/assets/

Steps (each is an existing module, run as a subprocess and stopped on the first failure):
  data       organise team recordings (from Drive, if present), ESC-50, manifests
  synthetic  train-only mixes, manifests again (to add them), synthetic test set
  features   per-hop YAMNet cache -> window arrays (and the clip-mean arrays for comparison)
  train      runs a (real only), b (+synthetic), c (team fine-tune of a and of b); select on validation
  eval       thresholds on validation; test metrics (public / Indian / synthetic); false alarms per hour
  export     FP32 + INT8 head, validation against Keras, YAMNet TFLite
  report     docs/evaluation_report.md from the run files
Downloads (AudioSet, FSD50K, DESED, iNoise) are separate: see src/data/download*.py.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VENV_PYTHON = REPO / ".venv" / "bin" / "python"
PYTHON = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable
STEPS = ("data", "synthetic", "features", "train", "eval", "export", "report")
RECIPE = ["--learning-rate", "3e-4", "--batch-size", "128", "--epochs", "40", "--patience", "8"]
FINETUNE = ["--learning-rate", "1e-4", "--batch-size", "64", "--epochs", "20", "--patience", "5"]


def sh(*args: str, desc: str) -> None:
    print(f"\n=== {desc}\n$ {' '.join(args)}", flush=True)
    if subprocess.run([PYTHON, *args], cwd=REPO).returncode != 0:
        sys.exit(f"Step failed: {desc}")


def newest_run(name: str) -> str:
    runs = sorted((REPO / "models" / "logs").glob(f"*_{name}"))
    if not runs:
        sys.exit(f"No run named *_{name}")
    return str(runs[-1])


def step_data() -> None:
    drive = os.environ.get("SAHARA_DRIVE_DATASETS")
    team = Path(drive) / "Indian_Ambient" if drive else None
    if team and team.is_dir():
        sh("scripts/organise_recordings.py", "--source", str(team), "--dest", "data/raw", desc="Organise team recordings")
    if (REPO / "data/interim/ESC-50-master.zip").exists() or not (REPO / "data/raw/esc50/meta/esc50.csv").exists():
        sh("-m", "src.data.download_esc50", desc="ESC-50")
    sh("src/data/manifest_builder.py", desc="Manifests (real data)")


def step_synthetic() -> None:
    sh("src/preprocessing/synthetic_mixing.py", "--split", "train", desc="Synthetic mixes (train only)")
    sh("src/data/manifest_builder.py", desc="Manifests (with synthetic)")
    sh("src/preprocessing/synthetic_mixing.py", "--split", "test", desc="Synthetic test set (reported separately)")


def step_features() -> None:
    sh("src/preprocessing/build_features_cache.py", "--io-workers", "6", desc="YAMNet hop cache + window arrays")
    sh("src/preprocessing/build_features_cache.py", "--mode", "clip_mean", desc="Clip-mean arrays (comparison)")


def step_train() -> None:
    sh("src/training/train.py", "--name", "a_baseline", "--no-synthetic", *RECIPE, desc="Run a: real data")
    sh("src/training/train.py", "--name", "b_synthetic", *RECIPE, desc="Run b: + synthetic")
    sh("src/training/train.py", "--name", "c_finetune_team_from_a", "--no-synthetic",
       "--finetune-from", newest_run("a_baseline"), *FINETUNE, desc="Run c: team fine-tune of a")
    sh("src/training/train.py", "--name", "c_finetune_team_from_b",
       "--finetune-from", newest_run("b_synthetic"), *FINETUNE, desc="Run c: team fine-tune of b")
    sh("src/training/train.py", "--name", "z_clipmean", "--features-dir", "data/processed/clip_mean", "--no-synthetic",
       *RECIPE, desc="Comparison: old clip-mean features")
    sh("scripts/compare_runs.py", desc="Select a run on validation")


def step_eval() -> None:
    sh("src/evaluation/threshold_search.py", desc="Thresholds (validation)")
    sh("src/evaluation/evaluate.py", desc="Test metrics")
    sh("src/evaluation/false_alarm_eval.py", desc="False alarms per hour")


def step_export(sync_assets: bool) -> None:
    sh("src/export/quantize.py", desc="FP32 + INT8 head")
    sh("src/export/validate_tflite.py", desc="Validate TFLite heads")
    if not (REPO / "models/exported/yamnet.tflite").exists():
        sh("app/tool/export_yamnet_tflite.py", desc="YAMNet TFLite")
    if sync_assets:
        sh("app/tool/sync_assets.py", "--yamnet", "models/exported/yamnet.tflite", desc="Copy models into app/assets")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="start", choices=STEPS, default=STEPS[0])
    parser.add_argument("--to", dest="end", choices=STEPS, default=STEPS[-1])
    parser.add_argument("--sync-assets", action="store_true", help="Copy the exported models into app/assets/.")
    args = parser.parse_args()
    for step in STEPS[STEPS.index(args.start): STEPS.index(args.end) + 1]:
        if step == "export":
            step_export(args.sync_assets)
        elif step == "report":
            sh("scripts/write_evaluation_report.py", desc="Evaluation report")
        else:
            globals()[f"step_{step}"]()
    print("\nPipeline finished.")


if __name__ == "__main__":
    main()
