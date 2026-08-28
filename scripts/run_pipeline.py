#!/usr/bin/env python3
"""
SAHARA pipeline runner — runs Steps 4-9 in order after downloads complete.
Usage:
    python scripts/run_pipeline.py [--augment] [--skip-cache]

Steps:
    1. Rebuild manifest (all sources present)
    2. Build YAMNet embedding cache
    3. Train classifier head
    4. Evaluate + generate confusion matrix
    5. Indian ambient hot-swap test
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import METADATA_DIR, MODELS_DIR, PROCESSED_DATA_DIR, REPO_ROOT
from src.data.manifest_builder import run as build_manifest
from src.evaluation.confusion_matrix import evaluate_and_plot
from src.preprocessing.build_features_cache import build_cache
from src.training.train import train


def step(name: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")


def hot_swap_test() -> bool:
    """Create dummy indian_ambient data, verify manifest picks it up, clean up."""
    import csv
    import wave
    import struct

    ia_root = REPO_ROOT / "data" / "raw" / "indian_ambient"
    tc_dir = ia_root / "target_classes"
    bg_dir = ia_root / "background"
    meta_path = ia_root / "metadata.csv"
    tc_dir.mkdir(parents=True, exist_ok=True)
    bg_dir.mkdir(parents=True, exist_ok=True)

    def _write_silence(path: Path, duration_sec: float = 0.5, sr: int = 16000) -> None:
        n_frames = int(sr * duration_sec)
        with wave.open(str(path), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(struct.pack(f"<{n_frames}h", *([0] * n_frames)))

    dummy_files = [
        (tc_dir / "test_smoke_alarm_001.wav", "smoke_alarm", False),
        (tc_dir / "test_dog_bark_001.wav", "dog_bark", False),
        (bg_dir / "test_background_001.wav", "background", True),
    ]

    print("\n[Hot-swap test] Creating dummy indian_ambient files...")
    for path, cls, is_bg in dummy_files:
        _write_silence(path)
        print(f"  Created: {path.name}  class={cls}")

    with meta_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["filename", "class", "is_background", "location_type",
                        "recording_device", "distance_m", "timestamp", "notes"],
        )
        writer.writeheader()
        for path, cls, is_bg in dummy_files:
            writer.writerow({
                "filename": path.name,
                "class": cls,
                "is_background": str(is_bg).lower(),
                "location_type": "test_environment",
                "recording_device": "synthetic",
                "distance_m": "1.0",
                "timestamp": "2026-08-28T00:00:00",
                "notes": "hot-swap design validation — not real data",
            })

    print("\n[Hot-swap test] Re-running manifest builder WITH dummy files...")
    counts_with = build_manifest()
    print(f"  Manifest totals WITH indian_ambient: {counts_with}")

    # Verify indian_ambient rows appear
    import csv as csv_mod
    full_manifest = METADATA_DIR / "full_manifest.csv"
    ia_rows = []
    with full_manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv_mod.DictReader(handle):
            if row["source_dataset"] == "indian_ambient":
                ia_rows.append(row)
    print(f"  indian_ambient rows in manifest: {len(ia_rows)}")
    for row in ia_rows:
        print(f"    {row['label']:20s}  {Path(row['filepath']).name}")

    passed = len(ia_rows) == len(dummy_files)
    if not passed:
        print(f"  ❌ FAIL: expected {len(dummy_files)} rows, got {len(ia_rows)}")

    print("\n[Hot-swap test] Deleting dummy files...")
    for path, _, _ in dummy_files:
        path.unlink(missing_ok=True)
    meta_path.unlink(missing_ok=True)

    print("\n[Hot-swap test] Re-running manifest builder WITHOUT indian_ambient files...")
    counts_without = build_manifest()
    print(f"  Manifest totals WITHOUT indian_ambient: {counts_without}")

    ia_rows_after = []
    with full_manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv_mod.DictReader(handle):
            if row["source_dataset"] == "indian_ambient":
                ia_rows_after.append(row)
    print(f"  indian_ambient rows after cleanup: {len(ia_rows_after)}")

    passed = passed and len(ia_rows_after) == 0
    print(f"\n[Hot-swap test] Result: {'✅ PASSED' if passed else '❌ FAILED'}")
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--augment", action="store_true", help="Apply embedding-space augmentation during training")
    parser.add_argument("--skip-cache", action="store_true", help="Skip feature cache rebuild (use existing .npz files)")
    args = parser.parse_args()

    total_start = time.time()

    # ── Step 1: Rebuild manifest ────────────────────────────────────────────
    step("Step 1: Build manifest (all available sources)")
    build_manifest()

    # ── Step 2: Feature cache ───────────────────────────────────────────────
    if not args.skip_cache:
        step("Step 2: Build YAMNet embedding cache")
        build_cache()
    else:
        print("\n[skip] Feature cache rebuild skipped (--skip-cache)")

    # ── Step 3: Train ───────────────────────────────────────────────────────
    step("Step 3: Train classifier head")
    results = train(use_augmentation=args.augment)
    print(f"\n  Train loss={results['train_loss']:.4f}  acc={results['train_acc']:.4f}")
    print(f"  Val   loss={results['val_loss']:.4f}  acc={results['val_acc']:.4f}")

    # ── Step 4: Evaluate ────────────────────────────────────────────────────
    step("Step 4: Evaluate on test set + generate confusion matrix")
    cm_path = evaluate_and_plot()
    print(f"  Confusion matrix → {cm_path}")

    # ── Step 5: Hot-swap test ───────────────────────────────────────────────
    step("Step 5: Indian-ambient hot-swap design validation")
    hot_swap_passed = hot_swap_test()

    # ── Summary ─────────────────────────────────────────────────────────────
    elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {elapsed/60:.1f} min")
    print(f"  Confusion matrix: {cm_path}")
    print(f"  Hot-swap test:    {'PASSED ✅' if hot_swap_passed else 'FAILED ❌'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
