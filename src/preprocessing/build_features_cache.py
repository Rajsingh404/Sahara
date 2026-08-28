"""Build cached YAMNet embedding arrays from the full manifest."""
from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import METADATA_DIR, PROCESSED_DATA_DIR, SOUND_CLASSES
from src.preprocessing.audio_utils import load_audio
from src.preprocessing.features import embedding_cache_path, extract_yamnet_embedding_cached


def _load_manifest(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _label_to_index(label: str) -> int:
    if label in SOUND_CLASSES:
        return SOUND_CLASSES.index(label)
    return -1  # background / unknown


def _is_local_file(filepath: str) -> bool:
    """Return True if the file is safe to read without blocking indefinitely.

    Behaviour by path type:

    * **Google Drive CloudStorage path** (``~/Library/CloudStorage/GoogleDrive-*``):
      These use the modern Google Drive for Desktop native integration and
      stream reliably on-demand.  Always allowed — streaming is the intent.

    * **Local disk** (no Drive xattr): always True.

    * **Old FUSE-mounted Drive paths** (data/raw/fsd50k/ pointing to a
      Drive FUSE mount that is NOT under CloudStorage): these previously
      blocked indefinitely on cold reads.  We gate those via xattr check.
    """
    import subprocess
    # CloudStorage = modern Drive for Desktop streaming — allow unconditionally.
    if "CloudStorage" in filepath and "GoogleDrive" in filepath:
        return True
    # For other paths, check for Drive stub xattr to catch stale FUSE mounts.
    try:
        result = subprocess.run(
            ["xattr", "-l", filepath],
            capture_output=True, timeout=2,
        )
        # Use raw bytes to avoid UTF-8 decode errors on binary xattr values.
        xattrs = result.stdout
        has_stub = b"drivefs.item-id#S" in xattrs
        has_local = b"drivefs.item-id#PS" in xattrs
        if has_stub and not has_local:
            return False
        return True
    except Exception:
        return True


def build_cache(
    manifest_path: Path | None = None,
    splits: tuple[str, ...] = ("train", "val", "test"),
    io_workers: int = 4,
) -> None:
    manifest_path = manifest_path or METADATA_DIR / "full_manifest.csv"
    rows = _load_manifest(manifest_path)
    if not rows:
        print("No rows in manifest — nothing to cache.")
        return

    start = time.time()
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Only cache target-class clips for baseline training.
    target_rows = [row for row in rows if row["label"] in SOUND_CLASSES]

    # Split into already-cached and needs-loading — skip the FUSE read for cached clips.
    cached_rows = [r for r in target_rows if embedding_cache_path(r["filepath"]).exists()]
    uncached_rows = [r for r in target_rows if not embedding_cache_path(r["filepath"]).exists()]

    # Filter uncached rows to only locally-available files (skip Google Drive stubs).
    # A Google Drive File Stream stub has xattr com.google.drivefs.item-id#S (not #PS).
    # Reading a stub triggers a cloud fetch which may hang — we skip those here.
    locally_available = []
    stub_skipped = 0
    for r in uncached_rows:
        if _is_local_file(r["filepath"]):
            locally_available.append(r)
        else:
            stub_skipped += 1
    if stub_skipped:
        print(f"  Skipped {stub_skipped} Google Drive stubs (not locally synced) — re-run after Drive syncs them.")
    uncached_rows = locally_available

    print(f"  {len(cached_rows)} embeddings already cached, {len(uncached_rows)} to load from disk.")

    embeddings: list[np.ndarray] = []
    labels: list[int] = []
    filepaths: list[str] = []
    skipped = 0

    # --- Fast path: load already-cached embeddings directly (no audio I/O) ---
    for row in tqdm(cached_rows, desc="Loading cached embeddings", unit="clip"):
        cache = embedding_cache_path(row["filepath"])
        emb = np.load(cache)
        embeddings.append(emb)
        labels.append(_label_to_index(row["label"]))
        filepaths.append(row["filepath"])

    # --- Slow path: batched parallel audio I/O → serial YAMNet inference ---
    # Process uncached clips in batches: load a batch of audio files in parallel
    # (threads help on FUSE/network mounts where I/O is the bottleneck), then
    # run YAMNet inference serially on the main thread (YAMNet is not thread-safe).
    if uncached_rows:
        batch_size = max(io_workers * 2, 16)

        def _load_one(row: dict) -> tuple[dict, np.ndarray | None]:
            return row, load_audio(row["filepath"])

        with tqdm(total=len(uncached_rows), desc="YAMNet embeddings (new)", unit="clip") as pbar:
            for batch_start in range(0, len(uncached_rows), batch_size):
                batch = uncached_rows[batch_start : batch_start + batch_size]
                with ThreadPoolExecutor(max_workers=io_workers) as pool:
                    loaded = list(pool.map(_load_one, batch))
                for row, waveform in loaded:
                    pbar.update(1)
                    if waveform is None:
                        skipped += 1
                        continue
                    emb = extract_yamnet_embedding_cached(row["filepath"], waveform)
                    embeddings.append(emb)
                    labels.append(_label_to_index(row["label"]))
                    filepaths.append(row["filepath"])

    if not embeddings:
        print(f"No embeddings extracted ({skipped} skipped).")
        return

    X = np.stack(embeddings)
    y = np.array(labels, dtype=np.int32)
    np.savez_compressed(PROCESSED_DATA_DIR / "embeddings_full.npz", X=X, y=y, filepaths=filepaths)

    split_map = {name: _load_manifest(METADATA_DIR / f"{name}_manifest.csv") for name in splits}
    filepath_set = {row["filepath"] for row in target_rows}
    idx_by_path = {fp: i for i, fp in enumerate(filepaths)}

    for split_name, split_rows in split_map.items():
        indices = [idx_by_path[row["filepath"]] for row in split_rows if row["filepath"] in idx_by_path]
        if not indices:
            continue
        np.savez_compressed(
            PROCESSED_DATA_DIR / f"embeddings_{split_name}.npz",
            X=X[indices],
            y=y[indices],
            filepaths=[filepaths[i] for i in indices],
        )

    elapsed = time.time() - start
    print(f"\nCached {len(embeddings)} embeddings in {elapsed:.1f}s (skipped {skipped})")
    print(f"X shape: {X.shape}  y shape: {y.shape}")
    for split_name in splits:
        split_path = PROCESSED_DATA_DIR / f"embeddings_{split_name}.npz"
        if split_path.exists():
            data = np.load(split_path)
            print(f"  {split_name}: X={data['X'].shape} y={data['y'].shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=METADATA_DIR / "full_manifest.csv")
    parser.add_argument("--io-workers", type=int, default=4, help="Parallel threads for audio I/O (default: 4)")
    args = parser.parse_args()
    build_cache(args.manifest, io_workers=args.io_workers)


if __name__ == "__main__":
    main()
