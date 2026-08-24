"""Download target-class AudioSet clips with a persistent attempt manifest."""
from __future__ import annotations

import argparse
import csv
import subprocess
from collections import defaultdict
from pathlib import Path

from src.config import RAW_DATA_DIR, SOUND_CLASSES
from src.data.common import download_file, write_rows

CSV_BASE = "https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv"
CSV_FILES = ("class_labels_indices.csv", "balanced_train_segments.csv", "eval_segments.csv", "unbalanced_train_segments.csv")
# Names are resolved against class_labels_indices.csv, so this remains resilient to ID changes.
TARGET_LABEL_NAMES = {
    "smoke_alarm": {"Smoke detector, smoke alarm"}, "doorbell": {"Doorbell"},
    "siren": {"Siren"}, "knocking": {"Knock"}, "dog_bark": {"Bark", "Dog"},
    "baby_cry": {"Baby cry, infant cry"}, "glass_break": {"Glass", "Breaking"},
    "appliance_beep": {"Beep, bleep", "Microwave oven", "Alarm clock"},
}


def _read_segments(path: Path):
    with path.open(encoding="utf-8") as handle:
        rows = (line for line in handle if not line.startswith("#"))
        yield from csv.DictReader(rows, skipinitialspace=True)


def _target_mids(labels_path: Path) -> dict[str, set[str]]:
    found = {name: set() for name in SOUND_CLASSES}
    with labels_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            display = row["display_name"]
            for target, names in TARGET_LABEL_NAMES.items():
                if display in names:
                    found[target].add(row["mid"])
    return found


def _load_manifest(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["clip_key"]: row["status"] for row in csv.DictReader(handle)}


def run(limit_per_class: int | None = None, retry_failed: bool = False, dry_run: bool = False) -> dict[str, int]:
    root = RAW_DATA_DIR / "audioset"
    csv_root = root / "metadata"
    for filename in CSV_FILES:
        # unbalanced is fetched now but only used for classes below the requested limit/100 clips.
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
    # Include unbalanced only when a class needs more coverage, avoiding an unnecessary huge pass.
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
                command = ["yt-dlp", f"https://www.youtube.com/watch?v={video_id}", "--download-sections", f"*{start}-{end}", "-x", "--audio-format", "wav", "--postprocessor-args", "ffmpeg:-ac 1 -ar 16000", "-o", str(output.with_suffix(".%(ext)s")), "--no-playlist", "--quiet"]
                result = subprocess.run(command, capture_output=True, text=True)
                status = "success" if result.returncode == 0 and output.exists() else "failed"
                attempts.append({"clip_key": key, "video_id": video_id, "class": target, "start": start, "end": end, "status": status, "error": result.stderr[-500:]})
                previous[key] = status
                if status == "success":
                    counts[target] += 1
                else:
                    print(f"FAILED {key}: {result.stderr[-160:]}")
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
