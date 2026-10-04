"""Download target-class AudioSet clips with a persistent attempt manifest."""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import shutil
from threading import Lock

from src.config import AUDIOSET_TARGET_PER_CLASS, RAW_DATA_DIR, SOUND_CLASSES
from src.data.common import download_file, write_rows
from src.data.label_map import audioset_mids_for_class

CSV_BASE = "https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv"
CSV_FILES = ("class_labels_indices.csv", "balanced_train_segments.csv", "eval_segments.csv", "unbalanced_train_segments.csv")
MIN_FREE_GB = 20


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


def _target_mids() -> dict[str, set[str]]:
    # Exact mids only. Display-name matching pulled in "Glass" (clinking) and "Dog".
    return {name: audioset_mids_for_class(name) for name in SOUND_CLASSES}


def _load_manifest(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["clip_key"]: row for row in csv.DictReader(handle)}


def _clip_path(root: Path, target: str, video_id: str, start: str, end: str) -> Path:
    return root / target / f"{video_id}_{start.replace('.', '_')}_{end.replace('.', '_')}.wav"


def _fetch(output: Path, video_id: str, start: str, end: str) -> tuple[str, str]:
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
        result = subprocess.run(command, capture_output=True, text=True, timeout=180)
        error = result.stderr[-500:]
        status = "success" if result.returncode == 0 and output.exists() else "failed"
    except (OSError, subprocess.TimeoutExpired) as exc:
        error, status = repr(exc), "failed"
    return status, error


def run(
    limit_per_class: int | None = None,
    retry_failed: bool = False,
    dry_run: bool = False,
    target_per_class: int = AUDIOSET_TARGET_PER_CLASS,
    classes: list[str] | None = None,
    workers: int = 4,
) -> dict[str, int]:
    """Top every class up to ``target_per_class`` clips (``limit_per_class`` = smoke-test cap).

    Clips already on disk count toward the target. Sources are read in the order
    eval -> balanced -> unbalanced, so the curated segments are used first.
    """
    _check_disk_space()
    root = RAW_DATA_DIR / "audioset"
    csv_root = root / "metadata"
    for filename in CSV_FILES:
        if filename == "unbalanced_train_segments.csv" and limit_per_class is not None:
            continue
        download_file(f"{CSV_BASE}/{filename}", csv_root / filename, dry_run)
    if dry_run or not (csv_root / "class_labels_indices.csv").exists():
        print("DRY RUN resolve target AudioSet mids and download segmented 16 kHz mono WAV files via yt-dlp.")
        return {name: 0 for name in SOUND_CLASSES}
    target = limit_per_class if limit_per_class is not None else target_per_class
    wanted = classes or list(SOUND_CLASSES)
    mids = {k: v for k, v in _target_mids().items() if k in wanted}
    print("AudioSet mids:", {key: sorted(value) for key, value in mids.items()})
    manifest_path = root / "download_manifest.csv"
    previous = _load_manifest(manifest_path)

    counts: dict[str, int] = defaultdict(int)
    for name in mids:
        counts[name] = sum(1 for _ in (root / name).glob("*.wav")) if (root / name).is_dir() else 0
    print("Already on disk:", dict(counts))

    jobs: list[tuple[str, str, str, str, str, str]] = []
    queued: dict[str, int] = defaultdict(int)
    for source_name in ("eval_segments.csv", "balanced_train_segments.csv", "unbalanced_train_segments.csv"):
        source = csv_root / source_name
        if not source.exists() or (source_name.startswith("unbalanced") and limit_per_class is not None):
            continue
        for row in _read_segments(source):
            labels = set(row["positive_labels"].strip('"').split(","))
            for name, values in mids.items():
                # Over-queue by 2x to absorb unavailable videos; stop once the target is met.
                if not labels & values or counts[name] + queued[name] >= 2 * target:
                    continue
                video_id = row["YTID"].strip()
                start, end = row["start_seconds"].strip(), row["end_seconds"].strip()
                key = f"{name}:{video_id}:{start}:{end}"
                old = previous.get(key, {}).get("status")
                if old == "success" or (old == "failed" and not retry_failed):
                    continue
                if _clip_path(root, name, video_id, start, end).exists():
                    continue
                jobs.append((key, name, video_id, start, end, ",".join(sorted(labels))))
                queued[name] += 1
    print(f"Queued {len(jobs)} downloads:", dict(queued))

    fields = ["clip_key", "video_id", "class", "start", "end", "labels", "status", "error"]
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for job in jobs:
            futures[pool.submit(_maybe_fetch, root, job, counts, target)] = job
        for future in as_completed(futures):
            key, name, video_id, start, end, labels = futures[future]
            status, error = future.result()
            if status == "skipped":
                continue
            previous[key] = {"clip_key": key, "video_id": video_id, "class": name, "start": start, "end": end,
                             "labels": labels, "status": status, "error": error}
            done += 1
            if done % 25 == 0:
                print(f"{done} attempts, on disk: {dict(counts)}", flush=True)
                write_rows(manifest_path, previous.values(), fields)
    rows = [{f: row.get(f, "") for f in fields} for row in previous.values()]
    write_rows(manifest_path, rows, fields)
    write_rows(root / "failed_downloads.csv", [row for row in rows if row["status"] == "failed"], fields)
    print("Final on disk:", dict(counts))
    return {name: counts[name] for name in SOUND_CLASSES}


_count_lock = Lock()


def _maybe_fetch(root: Path, job: tuple, counts: dict[str, int], target: int) -> tuple[str, str]:
    _, name, video_id, start, end, _ = job
    with _count_lock:
        if counts[name] >= target:
            return "skipped", ""
    status, error = _fetch(_clip_path(root, name, video_id, start, end), video_id, start, end)
    if status == "success":
        with _count_lock:
            counts[name] += 1
    return status, error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-per-class", type=int, help="Smoke-test cap (skips the unbalanced CSV).")
    parser.add_argument("--target-per-class", type=int, default=AUDIOSET_TARGET_PER_CLASS)
    parser.add_argument("--classes", nargs="+", choices=SOUND_CLASSES)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(args.limit_per_class, args.retry_failed, args.dry_run, args.target_per_class, args.classes, args.workers)
