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
  features   per-hop YAMNet cache -> window arrays
  train      a (real data), b (+ synthetic), e (5-seed ensemble of a); select on validation policy score
  eval       alert settings tuned on validation to the false-alarm budget; test metrics (public / Indian /
             synthetic); false alarms per hour; 5-fold cross-validation over the team recordings
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
# Phase 4 recipe (chosen on validation and team cross-validation, see docs/evaluation_report.md):
# background windows x8 (fewer false alarms), team recordings x10 (Indian context), real data only.
RECIPE = ["--learning-rate", "3e-4", "--batch-size", "128", "--epochs", "40", "--patience", "8",
          "--background-weight", "8", "--team-weight", "10"]
ENSEMBLE_SEEDS = (42, 1, 2, 3, 4)


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
    sh("src/preprocessing/build_features_cache.py", "--io-workers", "12", desc="YAMNet hop cache + window arrays")


def step_train() -> None:
    sh("src/training/train.py", "--name", "a_real", "--no-synthetic", *RECIPE, desc="Run a: real data")
    sh("src/training/train.py", "--name", "b_synthetic", *RECIPE, desc="Run b: + synthetic mixes")
    members = [newest_run("a_real")]
    for seed in ENSEMBLE_SEEDS[1:]:
        sh("src/training/train.py", "--name", f"a_real_s{seed}", "--seed", str(seed), "--no-synthetic", *RECIPE,
           desc=f"Ensemble member, seed {seed}")
        members.append(newest_run(f"a_real_s{seed}"))
    sh("scripts/make_ensemble.py", "--name", "e_ens5_real", *members, desc="Run e: 5-seed ensemble of a")
    sh("scripts/compare_runs.py", desc="Select a run on validation (policy score)")


def step_eval() -> None:
    sh("src/evaluation/tune_policy.py", desc="Alert settings tuned to the false-alarm budget (validation)")
    sh("src/evaluation/evaluate.py", desc="Test metrics")
    sh("src/evaluation/false_alarm_eval.py", desc="False alarms per hour")
    sh("src/evaluation/team_cv.py", "--mode", "joint", "--team-weight", "10", desc="Team recordings: 5-fold CV")


def step_export(sync_assets: bool) -> None:
    sh("src/export/quantize.py", desc="FP32 + INT8 head")
    sh("src/export/validate_tflite.py", desc="Validate TFLite heads")
    sh("src/personalization/export_center.py", desc="Custom-sound centring vector")
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
