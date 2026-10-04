"""Build the YAMNet feature cache and the per-split training arrays.

Stage 1 (cache): every clip in the manifest (targets, background, synthetic; all splits) is
loaded at its natural level (no peak normalisation, like the phone's raw microphone PCM) and
embedded hop by hop exactly as the phone does (features.extract_hop_embeddings). One
``data/processed/hop_cache/<key>.npz`` per clip; reruns only embed new clips.

Stage 2 (arrays), per split, from the cache:
- ``--mode window`` (default): device window vectors with event-window selection
  (windowing.clip_examples). Written to data/processed/:
  ``{split}_embeddings.npy`` [n, 1024], ``{split}_labels.npy`` [n, 8] multi-hot,
  ``{split}_windows_meta.npz`` (filepath, source_dataset, source_type, recording_id, hop_index).
- ``--mode clip_mean``: the old one-vector-per-clip mean, for comparison, written to
  data/processed/clip_mean/ with the same file names.

``--augment`` adds AUGMENTATION_VARIANTS_PER_CLIP raw-waveform variants of each real train
target clip (train only; validation and test are never augmented).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import (
    AUGMENTATION_ENABLED,
    AUGMENTATION_VARIANTS_PER_CLIP,
    METADATA_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    SOUND_CLASSES,
)
from src.preprocessing.audio_utils import load_audio
from src.preprocessing.features import extract_hop_embeddings, hop_cache_path, load_hop_cache, save_hop_cache
from src.preprocessing.windowing import clip_examples, clip_mean_example

SPLITS = ("train", "val", "test")
META_FIELDS = ("filepath", "source_dataset", "source_type", "recording_id")


def _load_manifest(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _clips(rows: list[dict]) -> dict[str, dict]:
    """One entry per audio file: its target labels (empty = background) and metadata."""
    clips: dict[str, dict] = {}
    for row in rows:
        clip = clips.setdefault(row["filepath"], {**{k: row[k] for k in META_FIELDS}, "labels": set()})
        if row["label"] in SOUND_CLASSES:
            clip["labels"].add(row["label"])
    return clips


def _augment_variants(clips: dict[str, dict]) -> list[tuple[str, str]]:
    return [(fp, f"aug{i}") for fp, c in clips.items() if c["labels"] and c["source_type"] != "synthetic_mixed"
            for i in range(AUGMENTATION_VARIANTS_PER_CLIP)]


def build_hop_cache(jobs: list[tuple[str, str]], io_workers: int = 4, background_paths: list[str] | None = None) -> int:
    """Embed every (filepath, variant) job whose cache file is missing. Returns the failure count."""
    todo = [(fp, v) for fp, v in jobs if not hop_cache_path(fp, v).exists()]
    print(f"Hop cache: {len(jobs) - len(todo)} cached, {len(todo)} to embed")
    if not todo:
        return 0
    augmenter = None
    if any(v for _, v in todo):
        from src.preprocessing.augmentation import SaharaAugmenter

        augmenter = SaharaAugmenter(background_paths or None)

    def load(job: tuple[str, str]):
        fp, variant = job
        wave = load_audio(fp, peak_normalize=False)
        if wave is not None and variant:
            wave = augmenter.augment(wave)
        return job, wave

    failed = 0
    with ThreadPoolExecutor(max_workers=io_workers) as pool, tqdm(total=len(todo), unit="clip") as bar:
        # Audio I/O in threads (Drive streaming is the bottleneck); YAMNet on this thread.
        for job, wave in pool.map(load, todo):
            bar.update(1)
            if wave is None or wave.size == 0:
                failed += 1
                continue
            save_hop_cache(hop_cache_path(*job), extract_hop_embeddings(wave))
    return failed


def assemble_split(clips: dict[str, dict], mode: str, variants: dict[str, list[str]] | None = None):
    xs, ys, meta = [], [], defaultdict(list)
    missing = 0
    for fp, clip in clips.items():
        for variant in [""] + (variants or {}).get(fp, []):
            path = hop_cache_path(fp, variant)
            if not path.exists():
                missing += 1
                continue
            hops = load_hop_cache(path)
            if mode == "clip_mean":
                x, y = clip_mean_example(hops, clip["labels"])
                hop_index = np.full(1, -1, dtype=np.int32)
            else:
                x, y, hop_index = clip_examples(hops, clip["labels"])
            if len(x) == 0:
                continue
            xs.append(x)
            ys.append(y)
            for key in META_FIELDS:
                meta[key].extend([clip[key]] * len(x))
            meta["hop_index"].extend(hop_index.tolist())
            meta["variant"].extend([variant] * len(x))
    X = np.concatenate(xs).astype(np.float32) if xs else np.zeros((0, 1024), np.float32)
    Y = np.concatenate(ys).astype(np.float32) if ys else np.zeros((0, len(SOUND_CLASSES)), np.float32)
    return X, Y, {k: np.asarray(v) for k, v in meta.items()}, missing


def build_cache(
    manifest_path: Path | None = None,
    splits: tuple[str, ...] = SPLITS,
    io_workers: int = 4,
    force: bool = False,
    augment: bool = AUGMENTATION_ENABLED,
    mode: str = "window",
) -> dict:
    start = time.time()
    np.random.seed(RANDOM_SEED)
    split_clips = {s: _clips(_load_manifest(METADATA_DIR / f"{s}_manifest.csv")) for s in splits}
    all_clips = {fp: c for clips in split_clips.values() for fp, c in clips.items()}
    if not all_clips:
        print("No rows in the split manifests — run manifest_builder.py first.")
        return {}

    jobs = [(fp, "") for fp in all_clips]
    aug_variants: dict[str, list[str]] = defaultdict(list)
    if augment and "train" in split_clips:
        for fp, v in _augment_variants(split_clips["train"]):
            jobs.append((fp, v))
            aug_variants[fp].append(v)
    beds = [fp for fp, c in split_clips.get("train", {}).items() if not c["labels"] and c["source_type"] != "synthetic_mixed"]
    failed = build_hop_cache(jobs, io_workers, beds)

    out_dir = PROCESSED_DATA_DIR if mode == "window" else PROCESSED_DATA_DIR / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {"mode": mode, "augment": augment, "failed_to_load": failed, "splits": {}}
    for split, clips in split_clips.items():
        X, Y, meta, missing = assemble_split(clips, mode, aug_variants if split == "train" else None)
        np.save(out_dir / f"{split}_embeddings.npy", X)
        np.save(out_dir / f"{split}_labels.npy", Y)
        np.savez(out_dir / f"{split}_windows_meta.npz", **meta)
        positives = {c: int(Y[:, i].sum()) for i, c in enumerate(SOUND_CLASSES)}
        background = int((Y.sum(axis=1) == 0).sum())
        by_type = dict(Counter(meta.get("source_type", np.array([])).tolist()))
        summary["splits"][split] = {"clips": len(clips), "windows": len(X), "background_windows": background,
                                    "positive_windows": positives, "windows_by_source_type": by_type,
                                    "clips_missing_from_cache": missing}
        print(f"{split}: {len(clips)} clips -> {len(X)} windows ({background} background) {positives}")
    summary["seconds"] = round(time.time() - start, 1)
    (out_dir / "features_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=None, help="Unused; split manifests are read directly.")
    parser.add_argument("--io-workers", type=int, default=4, help="Parallel threads for audio I/O (default: 4)")
    parser.add_argument("--force", action="store_true", help="Kept for compatibility; the arrays are always rebuilt.")
    parser.add_argument("--augment", action="store_true", help="Add raw-waveform variants of train target clips.")
    parser.add_argument("--mode", choices=("window", "clip_mean"), default="window")
    args = parser.parse_args()
    build_cache(io_workers=args.io_workers, force=args.force, augment=args.augment, mode=args.mode)


if __name__ == "__main__":
    main()
