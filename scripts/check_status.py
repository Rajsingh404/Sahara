#!/usr/bin/env python3
"""
SAHARA Phase 2-3 Status Check
Shows current dataset coverage, manifests, embeddings, and model state.
Run: python scripts/check_status.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.config import METADATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, SOUND_CLASSES, RAW_DATA_DIR

def hr(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print('─'*60)

def check_raw_data():
    hr("Raw Dataset Inventory")
    datasets = {
        "fsd50k": "FSD50K (Google Drive)",
        "audioset": "AudioSet (yt-dlp)",
        "desed": "DESED real soundscapes",
        "indian_ambient": "Indian Ambient (Phase 4)",
    }
    for folder, label in datasets.items():
        root = RAW_DATA_DIR / folder
        if not root.exists():
            print(f"  {label:40s} ✗ not found")
            continue
        wav_count = sum(1 for p in root.rglob("*") if p.suffix.lower() in {".wav", ".flac"} and p.is_file())
        print(f"  {label:40s} {wav_count:6d} audio files")
        if folder == "audioset":
            for cls in SOUND_CLASSES:
                cls_dir = root / cls
                if cls_dir.is_dir():
                    n = sum(1 for p in cls_dir.glob("*.wav"))
                    print(f"    {cls:30s} {n:4d} clips")

def check_manifests():
    hr("Manifest Status")
    for name in ("full", "train", "val", "test"):
        path = METADATA_DIR / f"{name}_manifest.csv"
        if not path.exists():
            print(f"  {name}_manifest.csv         ✗ not built yet")
            continue
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        labels = defaultdict(int)
        sources = defaultdict(int)
        for row in rows:
            labels[row.get("label","?")] += 1
            sources[row.get("source_dataset","?")] += 1
        print(f"  {name}_manifest.csv: {len(rows):5d} rows | " + 
              " | ".join(f"{s}={n}" for s,n in sorted(sources.items())))

def check_coverage():
    full_path = METADATA_DIR / "full_manifest.csv"
    if not full_path.exists():
        print("\n  full_manifest.csv not found — run manifest_builder.py first")
        return
    hr("Per-class Coverage Table")
    with full_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    
    sources = sorted({row["source_dataset"] for row in rows})
    counts = defaultdict(lambda: defaultdict(int))
    for row in rows:
        counts[row["label"]][row["source_dataset"]] += 1
    
    # Header
    header_parts = ["Class".ljust(20)] + [s[:10].ljust(12) for s in sources] + ["TOTAL".ljust(8), ""]
    print("  " + " ".join(header_parts))
    print("  " + "─" * 70)
    
    for label in SOUND_CLASSES + ["background"]:
        total = sum(counts[label].values())
        flag = " ◀ LOW" if label in SOUND_CLASSES and total < 100 else ""
        parts = [label.ljust(20)] + [str(counts[label].get(s,0)).ljust(12) for s in sources] + [str(total).ljust(8), flag]
        print("  " + " ".join(parts))

def check_embeddings():
    hr("Embedding Cache Status")
    import numpy as np
    for name in ("full", "train", "val", "test"):
        path = PROCESSED_DATA_DIR / f"embeddings_{name}.npz"
        if not path.exists():
            print(f"  embeddings_{name}.npz      ✗ not built yet")
            continue
        data = np.load(path)
        print(f"  embeddings_{name}.npz      X={data['X'].shape} y={data['y'].shape}")

def check_model():
    hr("Model State")
    ckpt = MODELS_DIR / "checkpoints" / "best_classifier.keras"
    if ckpt.exists():
        size = ckpt.stat().st_size / 1024
        print(f"  best_classifier.keras      ✓ ({size:.0f} KB)")
    else:
        print(f"  best_classifier.keras      ✗ not trained yet")
    
    cm = REPO / "docs" / "confusion_matrix.png"
    if (REPO / "docs" / "confusion_matrix.png").exists():
        print(f"  confusion_matrix.png       ✓")
    else:
        print(f"  confusion_matrix.png       ✗ not generated yet")

REPO = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    print("=" * 60)
    print("  SAHARA Phase 2-3 Pipeline Status")
    print("=" * 60)
    check_raw_data()
    check_manifests()
    check_coverage()
    try:
        check_embeddings()
    except Exception as e:
        print(f"  [embedding check error: {e}]")
    check_model()
    print("\n")
