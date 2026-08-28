"""Download target-class AudioSet clips with a persistent attempt manifest."""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import shutil

from src.config import RAW_DATA_DIR, SOUND_CLASSES
from src.data.common import download_file, write_rows
from src.data.label_map import AUDISET_DISPLAY_NAMES, audioset_mids_for_class

CSV_BASE = "https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv"
CSV_FILES = ("class_labels_indices.csv", "balanced_train_segments.csv", "eval_segments.csv", "unbalanced_train_segments.csv")
MIN_FREE_GB = 100


def _read_segments(path: Path):
    with path.open(encoding="utf-8") as handle:
        def csv_lines():
            for line in handle:
                if line.startswith("# YTID"):
                    yield line.removeprefix("# ")
                elif not line.startswith("#"):
                    yield line
        rows = csv_lines()
        for row in csv.DictReader(rows, skipinitialspace=True):
            # AudioSet's header is emitted as `# YTID, start_seconds, ...`.
            yield {str(key).strip(): (value or "").strip() for key, value in row.items()}


def _check_disk_space() -> None:
    usage = shutil.disk_usage(RAW_DATA_DIR)
    free_gb = usage.free / (1024**3)
    print(f"Free disk space: {free_gb:.1f} GiB")
    if free_gb < MIN_FREE_GB:
        print(f"WARNING: less than {MIN_FREE_GB} GiB free — AudioSet download may fail.")


def _target_mids(labels_path: Path) -> dict[str, set[str]]:
    found = {name: set(audioset_mids_for_class(name)) for name in SOUND_CLASSES}
    with labels_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            display = row["display_name"]
            for target, names in AUDISET_DISPLAY_NAMES.items():
                if display in names:
                    found[target].add(row["mid"])
    return found


def _load_manifest(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["clip_key"]: row["status"] for row in csv.DictReader(handle)}


def run(limit_per_class: int | None = None, retry_failed: bool = False, dry_run: bool = False) -> dict[str, int]:
    _check_disk_space()
    root = RAW_DATA_DIR / "audioset"
    csv_root = root / "metadata"
    for filename in CSV_FILES:
        # A smoke test must remain small; full runs fetch unbalanced only if needed.
        if filename == "unbalanced_train_segments.csv" and limit_per_class is not None:
            continue
        download_file(f"{CSV_BASE}/{filename}", csv_root / filename, dry_run)
    labels_path = csv_root / "class_labels_indices.csv"
    if dry_run or not labels_path.exists():
        print("DRY RUN resolve target AudioSet ontology labels and download segmented 16 kHz mono WAV files via yt-dlp.")
        return {name: 0 for name in SOUND_CLASSES}
    mids = _target_mids(labels_path)
    print("AudioSet mids:", {key: sorted(value) for key, value in mids.items()})
    manifest_path = root / "download_manifest.csv"
    previous = _load_manifest(manifest_path)
    attempts: list[dict] = []
    counts = defaultdict(int)
    sources = [csv_root / "balanced_train_segments.csv", csv_root / "eval_segments.csv"]
    # Include unbalanced only for a full run and only after balanced/eval are exhausted.
    if limit_per_class is None:
        sources.append(csv_root / "unbalanced_train_segments.csv")
    for source in sources:
        if not source.exists():
            continue
        for row in _read_segments(source):
            labels = set(row["positive_labels"].strip('"').split(","))
            classes = [name for name, values in mids.items() if labels & values]
            for target in classes:
                cap = limit_per_class if limit_per_class is not None else 100
                if source.name.startswith("unbalanced") and counts[target] >= cap:
                    continue
                if limit_per_class is not None and counts[target] >= limit_per_class:
                    continue
                video_id = row["YTID"].strip()
                start, end = row["start_seconds"].strip(), row["end_seconds"].strip()
                key = f"{target}:{video_id}:{start}:{end}"
                old = previous.get(key)
                if old == "success" or (old == "failed" and not retry_failed):
                    continue
                output = root / target / f"{video_id}_{start.replace('.', '_')}_{end.replace('.', '_')}.wav"
                output.parent.mkdir(parents=True, exist_ok=True)
                command = [
                    sys.executable, "-m", "yt_dlp",
                    f"https://www.youtube.com/watch?v={video_id}",
                    "--download-sections", f"*{start}-{end}",
                    "-x", "--audio-format", "wav",
                    "--postprocessor-args", "ffmpeg:-ac 1 -ar 16000",
                    "-o", str(output.with_suffix(".%(ext)s")),
                    "--no-playlist", "--quiet", "--no-warnings",
                    "--retries", "3",
                ]
                try:
                    result = subprocess.run(command, capture_output=True, text=True)
                    error = result.stderr[-500:]
                    status = "success" if result.returncode == 0 and output.exists() else "failed"
                except OSError as exc:
                    error, status = repr(exc), "failed"
                attempts.append({"clip_key": key, "video_id": video_id, "class": target, "start": start, "end": end, "status": status, "error": error})
                previous[key] = status
                if status == "success":
                    counts[target] += 1
                else:
                    print(f"FAILED {key}: {error[-160:]}")
    merged = [{"clip_key": key, "video_id": "", "class": "", "start": "", "end": "", "status": state, "error": ""} for key, state in previous.items()]
    # Preserve rich details for this run; the key gives idempotency across runs.
    by_key = {row["clip_key"]: row for row in merged}
    by_key.update({row["clip_key"]: row for row in attempts})
    rows = list(by_key.values())
    fields = ["clip_key", "video_id", "class", "start", "end", "status", "error"]
    write_rows(manifest_path, rows, fields)
    write_rows(root / "failed_downloads.csv", [row for row in rows if row["status"] == "failed"], fields)
    return {name: counts[name] for name in SOUND_CLASSES}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-per-class", type=int)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(args.limit_per_class, args.retry_failed, args.dry_run)
