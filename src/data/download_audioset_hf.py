"""Fetch AudioSet balanced-train and eval clips from the Hugging Face mirror (no YouTube).

Mirror: https://huggingface.co/datasets/agkphysics/AudioSet (CC BY 4.0, no login).
Each parquet file holds 500 clips in row groups of 100 (~140 MB of FLAC per group). Only the
label columns are read to find matching clips; audio is fetched one row group at a time and
nothing but the matching clips is kept. Clips are written as 16 kHz mono WAV to
data/raw/audioset/<class>/<video_id>_<start>_<end>.wav, the layout the AudioSet scanner reads;
start/end come from the segment CSVs because the mirror stores only the video id.

    python -m src.data.download_audioset_hf --classes smoke_alarm baby_cry
"""
from __future__ import annotations

import argparse
import io
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import HfFileSystem

from src.config import RAW_DATA_DIR, SAMPLE_RATE, SOUND_CLASSES
from src.data.download_audioset import _read_segments
from src.data.label_map import audioset_mids_for_class

REPO = "datasets/agkphysics/AudioSet/data"
SPLITS = {
    "eval": "eval_segments.csv",
    "bal_train": "balanced_train_segments.csv",
    "unbal_train": "unbalanced_train_segments.csv",  # 870 files, 2.4 TB: only matching row groups are fetched
}


def _segment_times(csv_root: Path, splits: list[str]) -> dict[str, tuple[str, str]]:
    times: dict[str, tuple[str, str]] = {}
    for name in (SPLITS[s] for s in splits):
        for row in _read_segments(csv_root / name):
            times[row["YTID"]] = (row["start_seconds"], row["end_seconds"])
    return times


def _to_wav(flac: bytes, out: Path) -> None:
    audio, sr = sf.read(io.BytesIO(flac), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=SAMPLE_RATE)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, np.clip(mono, -1.0, 1.0), SAMPLE_RATE, subtype="PCM_16")


def run(classes: list[str], splits: list[str] | None = None, target_per_class: int | None = None,
        class_targets: dict[str, int] | None = None) -> dict[str, int]:
    """Add clips of ``classes``; stop a class once its target (``class_targets`` overrides
    ``target_per_class``) is reached on disk."""
    root = RAW_DATA_DIR / "audioset"
    splits = splits or ["eval", "bal_train"]
    times = _segment_times(root / "metadata", splits)
    mids = {name: audioset_mids_for_class(name) for name in classes}
    fs = HfFileSystem()
    added: dict[str, int] = defaultdict(int)

    def on_disk(c: str) -> int:
        return sum(1 for _ in (root / c).glob("*.wav")) if (root / c).is_dir() else 0

    for split in splits:
        files = sorted(p for p in fs.ls(f"{REPO}/{split}", detail=False) if p.endswith(".parquet"))
        for i, path in enumerate(files, 1):
            if target_per_class is not None or class_targets:
                limits = {c: (class_targets or {}).get(c, target_per_class) for c in classes}
                classes = [c for c in classes if limits[c] is None or on_disk(c) < limits[c]]
                if not classes:
                    print("All classes reached their targets on disk; stopping.")
                    return dict(added)
            with fs.open(path, "rb") as handle:
                pf = pq.ParquetFile(handle)
                labels = pf.read(columns=["video_id", "labels"]).to_pylist()
                # (row group, row in group) -> (video id, classes) for clips we still need
                wanted: dict[int, list[tuple[int, str, list[str]]]] = defaultdict(list)
                offset = 0
                for group in range(pf.metadata.num_row_groups):
                    n = pf.metadata.row_group(group).num_rows
                    for j, row in enumerate(labels[offset:offset + n]):
                        vid = row["video_id"]
                        hit = [c for c in classes if set(row["labels"]) & mids[c]]
                        if not hit or vid not in times:
                            continue
                        start, end = times[vid]
                        stem = f"{vid}_{start.replace('.', '_')}_{end.replace('.', '_')}.wav"
                        hit = [c for c in hit if not (root / c / stem).exists()]
                        if hit:
                            wanted[group].append((j, stem, hit))
                    offset += n
                for group, items in wanted.items():
                    audio = pf.read_row_group(group, columns=["audio"]).column("audio").to_pylist()
                    for j, stem, hit in items:
                        for c in hit:
                            try:
                                _to_wav(audio[j]["bytes"], root / c / stem)
                                added[c] += 1
                            except Exception as exc:  # undecodable clip: skip, keep going
                                print(f"skip {stem}: {exc!r}")
            print(f"{split} {i}/{len(files)} added so far: {dict(added)}", flush=True)
    return dict(added)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--classes", nargs="+", choices=SOUND_CLASSES, default=["smoke_alarm", "baby_cry"])
    parser.add_argument("--splits", nargs="+", choices=list(SPLITS), default=["eval", "bal_train"])
    parser.add_argument("--target-per-class", type=int, help="Stop a class once this many clips are on disk.")
    parser.add_argument("--class-target", action="append", default=[], metavar="CLASS=N",
                        help="Per-class override of --target-per-class, e.g. baby_cry=105")
    args = parser.parse_args()
    overrides = {k: int(v) for k, v in (item.split("=") for item in args.class_target)}
    print("Added:", run(args.classes, args.splits, args.target_per_class, overrides))
