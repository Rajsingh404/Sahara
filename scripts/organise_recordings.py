"""Organise the team's own recordings into the layouts the pipeline expects.

The four team members (Advika, Mahi, Prachi, Raj) recorded two things. The final
Drive folder (Datasets/Indian_Ambient) groups clips by class; the recorder of each
clip comes from the earlier per-person upload (``original_path``):

* Indian ambient audio: target-class sounds (doorbell, knocking, ...) and
  background scenes (kitchen, temple, hawker, rain, TV).
* Personalization clips: people saying the name "Aryan" (positive) and other,
  often similar-sounding names (negative).

``data/recordings/label_map.csv`` is the reviewed source of truth: one row per
Drive file with its recorder, label and new filename. This script copies the
audio into

    <dest>/indian_ambient/{target_classes,background}/ + metadata.csv
    <dest>/personalization/aryan/{positive,negative}/ + metadata.csv

and assigns a ``split`` column by recorder, so no person's voice, phone or room
appears in both train and test.

Usage (from the repo root):

    # Download from Drive by file id, then organise into data/raw/
    python scripts/organise_recordings.py --download --source data/interim/drive_recordings --dest data/raw

    # Organise an existing download (files named by Drive id, or the Drive tree)
    python scripts/organise_recordings.py --source /path/to/download --dest data/raw
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import urllib.request
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LABEL_MAP = REPO_ROOT / "data" / "recordings" / "label_map.csv"
DRIVE_URL = (
    "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"
)
DEFAULT_TEST_RECORDERS = ("mahi", "prachi")

AMBIENT_FIELDS = [
    "filename",
    "class",
    "is_background",
    "location_type",
    "recording_device",
    "distance_m",
    "timestamp",
    "notes",
    "recorder",
    "split",
    "duration_sec",
    "source_path",
]
PERSONAL_FIELDS = [
    "filename",
    "label",
    "recorder",
    "split",
    "recording_device",
    "timestamp",
    "duration_sec",
    "notes",
    "source_path",
]


def load_label_map(path: Path = LABEL_MAP) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def find_source(row: dict, source: Path) -> Path | None:
    """Locate a clip by Drive id, final Drive path or per-recorder Drive path."""
    for candidate in (
        source / row["drive_file_id"],
        source / row["source_path"],
        source / row["original_path"],
    ):
        if candidate.is_file():
            return candidate
    return None


def download(rows: list[dict], source: Path) -> None:
    source.mkdir(parents=True, exist_ok=True)
    for row in rows:
        out = source / row["drive_file_id"]
        if out.is_file() and out.stat().st_size > 0:
            continue
        print(f"downloading {row['source_path']}")
        urllib.request.urlretrieve(DRIVE_URL.format(row["drive_file_id"]), out)


def target_dir(row: dict, dest: Path) -> Path:
    if row["dataset"] == "ambient":
        return dest / "indian_ambient" / row["subset"]
    return dest / "personalization" / "aryan" / row["subset"]


def session_tag(row: dict) -> str:
    # Used by manifest_builder as recording_id, so clips from one sitting stay together.
    return f"{row['recorded_on'] or 'undated'}_{row['recorder']}"


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def organise(
    rows: list[dict],
    source: Path,
    dest: Path,
    test_recorders: set[str],
    copy_audio: bool = True,
) -> int:
    ambient: list[dict] = []
    personal: dict[str, list[dict]] = {}
    missing = 0
    for row in rows:
        split = "test" if row["recorder"] in test_recorders else "train"
        if copy_audio:
            src = find_source(row, source)
            if src is None:
                print(f"missing: {row['source_path']}", file=sys.stderr)
                missing += 1
                continue
            out_dir = target_dir(row, dest)
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out_dir / row["filename"])
        if row["dataset"] == "ambient":
            ambient.append(
                {
                    "filename": row["filename"],
                    "class": row["label"],
                    "is_background": str(row["subset"] == "background").lower(),
                    "location_type": row["location_type"],
                    "recording_device": row["recording_device"],
                    "distance_m": "",
                    "timestamp": session_tag(row),
                    "notes": row["notes"],
                    "recorder": row["recorder"],
                    "split": split,
                    "duration_sec": row["duration_sec"],
                    "source_path": row["source_path"],
                }
            )
        else:
            personal.setdefault(row["subset"], []).append(
                {
                    "filename": row["filename"],
                    "label": row["label"],
                    "recorder": row["recorder"],
                    "split": split,
                    "recording_device": row["recording_device"],
                    "timestamp": session_tag(row),
                    "duration_sec": row["duration_sec"],
                    "notes": row["notes"],
                    "source_path": row["source_path"],
                }
            )

    write_csv(dest / "indian_ambient" / "metadata.csv", AMBIENT_FIELDS, ambient)
    for subset, subset_rows in personal.items():
        write_csv(
            dest / "personalization" / "aryan" / subset / "metadata.csv",
            PERSONAL_FIELDS,
            subset_rows,
        )
    return missing


def summarise(rows: list[dict], test_recorders: set[str]) -> None:
    by_label = Counter((r["dataset"], r["label"]) for r in rows)
    by_recorder = Counter(r["recorder"] for r in rows)
    by_split = Counter(
        (
            r["dataset"],
            r["label"],
            "test" if r["recorder"] in test_recorders else "train",
        )
        for r in rows
    )
    print("\nclips per label (train/test):")
    for dataset, label in sorted(by_label):
        tr, te = by_split[(dataset, label, "train")], by_split[(dataset, label, "test")]
        print(
            f"  {dataset:8s} {label:15s} {by_label[(dataset, label)]:4d}  ({tr}/{te})"
        )
    print("clips per recorder:")
    for rec, n in sorted(by_recorder.items()):
        print(f"  {rec:8s} {n:4d}  {'test' if rec in test_recorders else 'train'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--source", type=Path, required=True, help="Folder holding the Drive download."
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=REPO_ROOT / "data" / "raw",
        help="Output root (default data/raw).",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Fetch every clip in the label map into --source first.",
    )
    parser.add_argument(
        "--test-recorders",
        default=",".join(DEFAULT_TEST_RECORDERS),
        help="Comma-separated recorders held out as test (default: %(default)s).",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Write metadata.csv files without copying audio.",
    )
    args = parser.parse_args()

    rows = load_label_map()
    test_recorders = {
        r.strip().lower() for r in args.test_recorders.split(",") if r.strip()
    }
    if args.download:
        download(rows, args.source)
    missing = organise(
        rows, args.source, args.dest, test_recorders, copy_audio=not args.metadata_only
    )
    summarise(rows, test_recorders)
    if missing:
        sys.exit(f"{missing} clips were missing from {args.source}")


if __name__ == "__main__":
    main()
