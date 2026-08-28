#!/usr/bin/env python3
"""
SAHARA Phase 2-3 End-to-End Pipeline Runner
============================================
Executes: manifest_builder → build_features_cache → train → evaluate
Handles the indian_ambient hot-swap validation test.

Run from repo root:
    python scripts/run_full_pipeline.py [--augment] [--skip-audioset-download]
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VENV_PYTHON = REPO / ".venv" / "bin" / "python"
PYTHON = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable


def run(cmd: list[str], desc: str, cwd: Path = REPO) -> int:
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"  ⚠ '{desc}' exited with code {result.returncode}")
    return result.returncode


def step_manifest():
    run([PYTHON, "-m", "src.data.manifest_builder"], "Step 1: Build manifests")


def step_features():
    run([PYTHON, "-m", "src.preprocessing.build_features_cache"], "Step 2: Extract YAMNet embeddings")


def step_train(augment: bool = False):
    cmd = [PYTHON, "-m", "src.training.train"]
    if augment:
        cmd.append("--augment")
    run(cmd, "Step 3: Train classifier head")


def step_evaluate():
    run([PYTHON, "-m", "src.evaluation.confusion_matrix"], "Step 4: Evaluate + confusion matrix")


def step_indian_ambient_hotswap_test():
    """
    Create dummy files in indian_ambient/, re-run manifest, confirm they appear,
    then delete them and confirm the pipeline handles an empty folder gracefully.
    """
    print(f"\n{'='*60}")
    print("  Step 5: indian_ambient hot-swap design validation")
    print(f"{'='*60}")

    ia_root = REPO / "data" / "raw" / "indian_ambient"
    tc_dir = ia_root / "target_classes"
    bg_dir = ia_root / "background"
    meta_path = ia_root / "metadata.csv"

    # Back up existing metadata if present.
    meta_backup = None
    if meta_path.exists():
        meta_backup = meta_path.with_suffix(".csv.bak")
        shutil.copy2(meta_path, meta_backup)
        print(f"  Backed up existing metadata.csv → {meta_backup.name}")

    # Create dummy WAV files (silent, 1s, 16kHz mono).
    import wave, struct, math
    def _make_silent_wav(path: Path, duration_sec: float = 1.0, sr: int = 16000):
        path.parent.mkdir(parents=True, exist_ok=True)
        n_frames = int(duration_sec * sr)
        with wave.open(str(path), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(b"\x00\x00" * n_frames)

    dummy_target = tc_dir / "_test_smoke_alarm.wav"
    dummy_bg = bg_dir / "_test_background.wav"
    _make_silent_wav(dummy_target)
    _make_silent_wav(dummy_bg)
    print(f"  Created: {dummy_target.relative_to(REPO)}")
    print(f"  Created: {dummy_bg.relative_to(REPO)}")

    # Write a minimal metadata.csv.
    import csv
    fields = ["filename", "class", "is_background", "location_type",
              "recording_device", "distance_m", "timestamp", "notes"]
    rows = [
        {
            "filename": "_test_smoke_alarm.wav", "class": "smoke_alarm",
            "is_background": "false", "location_type": "test_room",
            "recording_device": "test_device", "distance_m": "1.0",
            "timestamp": "2026-08-28T00:00:00", "notes": "hotswap validation dummy",
        },
        {
            "filename": "_test_background.wav", "class": "background",
            "is_background": "true", "location_type": "test_room",
            "recording_device": "test_device", "distance_m": "0.0",
            "timestamp": "2026-08-28T00:00:00", "notes": "hotswap validation dummy",
        },
    ]
    with meta_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Written: {meta_path.relative_to(REPO)} (2 rows)")

    print("\n  --- Re-running manifest builder WITH dummy files ---")
    result = subprocess.run(
        [PYTHON, "-m", "src.data.manifest_builder"],
        cwd=REPO, capture_output=True, text=True,
    )
    output = result.stdout + result.stderr

    # Check that indian_ambient rows appear.
    manifest_path = REPO / "data" / "metadata" / "full_manifest.csv"
    ia_count = 0
    if manifest_path.exists():
        with manifest_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("source_dataset") == "indian_ambient":
                    ia_count += 1

    if ia_count >= 2:
        print(f"  ✅ HOT-SWAP TEST PASSED: {ia_count} indian_ambient rows in manifest")
    else:
        print(f"  ❌ HOT-SWAP TEST FAILED: only {ia_count} indian_ambient rows found")
        print("  --- Manifest builder output ---")
        print(output)

    # Clean up dummy files.
    dummy_target.unlink(missing_ok=True)
    dummy_bg.unlink(missing_ok=True)
    meta_path.unlink(missing_ok=True)
    print(f"  Removed dummy files and metadata.csv")

    # Restore backup if it existed, else leave metadata absent.
    if meta_backup and meta_backup.exists():
        shutil.move(str(meta_backup), str(meta_path))
        print(f"  Restored backup metadata.csv")

    print("\n  --- Re-running manifest builder WITHOUT indian_ambient (empty folder) ---")
    result2 = subprocess.run(
        [PYTHON, "-m", "src.data.manifest_builder"],
        cwd=REPO, capture_output=True, text=True,
    )
    output2 = result2.stdout + result2.stderr
    if "indian_ambient: skipped" in output2:
        print("  ✅ EMPTY FOLDER TEST PASSED: 'indian_ambient: skipped' logged correctly")
    else:
        print("  ⚠  Expected 'indian_ambient: skipped' in output — checking manifest instead:")
    ia_count2 = 0
    if manifest_path.exists():
        with manifest_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("source_dataset") == "indian_ambient":
                    ia_count2 += 1
    if ia_count2 == 0:
        print(f"  ✅ EMPTY FOLDER TEST PASSED: 0 indian_ambient rows in manifest after cleanup")
    else:
        print(f"  ❌ EMPTY FOLDER TEST FAILED: {ia_count2} rows remain after cleanup")
    print("\n  Manifest output (empty-folder run):")
    print(output2[:2000])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--augment", action="store_true", help="Enable embedding-space augmentation during training")
    parser.add_argument("--skip-audioset-download", action="store_true", help="Skip AudioSet download (use existing clips)")
    parser.add_argument("--hotswap-only", action="store_true", help="Only run the indian_ambient hot-swap test")
    args = parser.parse_args()

    start = time.time()
    sys.path.insert(0, str(REPO))

    if args.hotswap_only:
        step_indian_ambient_hotswap_test()
        return

    step_manifest()
    step_features()
    step_train(args.augment)
    step_evaluate()
    step_indian_ambient_hotswap_test()

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"  Full pipeline completed in {elapsed/60:.1f} minutes")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
