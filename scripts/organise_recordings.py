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

and assigns a ``split`` column. Default (``--split-by label``): each label is split on its own,
~60 % train / 40 % test, with the test quota shared across the label's recording conditions
(Drive folder x recorder) so both sides see every condition; identical clips stay together.
``--split-by recorder`` keeps the earlier rule (whole recorders held out as test).

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
TEST_FRACTION = 0.4
# Share of each ambient label moved from train to validation (test is never touched), so the alert thresholds can be
# tuned on Indian audio. Off by default: tried 2026-10-30, it alerted on fewer Indian test clips (see the report).
VAL_FRACTION = 0.0
SPLIT_SEED = 42

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
    # Drive folder names can carry stray trailing spaces ("dog barking ").
    folder, _, name = row["source_path"].rpartition("/")
    if folder:
        for child in source.iterdir() if source.is_dir() else ():
            if child.is_dir() and child.name.strip() == folder.strip() and (child / name).is_file():
                return child / name
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


def condition(row: dict) -> str:
    """Recording condition: Drive folder + recorder (e.g. 'Doorbell|prachi')."""
    return f"{row['source_path'].split('/')[0].strip()}|{row['recorder']}"


_GENERIC_NAMES = {"", "new recording", "recording", "audio", "voice"}


def session_group(row: dict) -> str:
    """Best guess at the sitting (person, phone, place) a clip came from, for splits that never put one
    sitting on both sides. Named recorders are one group each. Clips with no recorder are grouped by their
    file-name series: phone recorders number takes in sequence ("Juhu Tara Road 61", "New Recording 7",
    Voice Memos naming them after the place), so one series is one phone at one place. A generic name is
    split further by Drive folder. Inferred from names only: it can merge two people or split one."""
    import re

    if row["recorder"] != "team":
        return row["recorder"]
    name = row["source_path"].split("/")[-1].rsplit(".", 1)[0].strip()
    name = re.sub(r"^copy of\s+", "", name, flags=re.I)
    name = re.sub(r"[\W\d_]+$", "", name).strip().lower()
    folder = row["source_path"].split("/")[0].strip().lower()
    return f"u{row.get('upload') or '1'}:{name}" + (f"@{folder}" if name in _GENERIC_NAMES else "")


def _duplicate_key(row: dict) -> str:
    # Clips with identical audio are marked in label_map.csv (`same_audio_as`, e.g. 'door bang 2'
    # is 'door bang' re-saved); they must land on the same side.
    return row.get("same_audio_as") or row["filename"]


def label_splits(rows: list[dict], test_fraction: float = TEST_FRACTION, seed: int = SPLIT_SEED) -> dict[str, str]:
    """filename -> 'train' | 'test', split per label and spread across recording conditions."""
    import random

    splits: dict[str, str] = {}
    # Each upload is split on its own, so adding clips never moves an earlier clip between train and test.
    by_label: dict[tuple, list[dict]] = {}
    for row in rows:
        by_label.setdefault((row["dataset"], row["label"], row.get("upload") or "1"), []).append(row)
    for (dataset, label, upload), label_rows in sorted(by_label.items()):
        n = len(label_rows)
        target = int(test_fraction * n + 0.5)
        if n >= 2:
            target = min(max(target, 1), n - 1)
        strata: dict[str, list[list[dict]]] = {}
        groups: dict[tuple, list[dict]] = {}
        for row in sorted(label_rows, key=lambda r: r["filename"]):
            groups.setdefault(_duplicate_key(row), []).append(row)
        for group in groups.values():
            strata.setdefault(condition(group[0]), []).append(group)
        # Largest-remainder share of the label's test quota per condition.
        quota = {c: test_fraction * sum(len(g) for g in gs) for c, gs in strata.items()}
        alloc = {c: int(q) for c, q in quota.items()}
        for c in sorted(quota, key=lambda c: (-(quota[c] - alloc[c]), c))[: max(0, target - sum(alloc.values()))]:
            alloc[c] += 1
        rng = random.Random(f"{seed}-{dataset}-{label}" + ("" if upload == "1" else f"-{upload}"))
        for c in sorted(strata):
            gs = strata[c]
            rng.shuffle(gs)
            taken = 0
            for group in gs:
                side = "test" if taken < alloc[c] else "train"
                taken += len(group) if side == "test" else 0
                for row in group:
                    splits[row["filename"]] = side
    return splits


def val_splits(rows: list[dict], splits: dict[str, str], val_fraction: float = VAL_FRACTION,
               seed: int = SPLIT_SEED) -> dict[str, str]:
    """Move ~val_fraction of each ambient label (per upload and condition) from train to val; test clips stay put."""
    import random

    out = dict(splits)
    strata: dict[tuple, dict[str, list[dict]]] = {}
    for row in sorted(rows, key=lambda r: r["filename"]):
        if row["dataset"] != "ambient" or splits[row["filename"]] != "train":
            continue
        key = (row["label"], row.get("upload") or "1", condition(row))
        strata.setdefault(key, {}).setdefault(_duplicate_key(row), []).append(row)
    for key in sorted(strata):
        groups = list(strata[key].values())
        n_label = sum(1 for r in rows if r["dataset"] == "ambient" and r["label"] == key[0]
                      and (r.get("upload") or "1") == key[1] and condition(r) == key[2])
        want = int(val_fraction * n_label + 0.5)
        random.Random(f"{seed}-val-{'-'.join(key)}").shuffle(groups)
        taken = 0
        for group in groups[:-1]:  # the condition always keeps at least one clip in train
            if taken >= want:
                break
            for row in group:
                out[row["filename"]] = "val"
            taken += len(group)
    return out


def recorder_splits(rows: list[dict], test_recorders: set[str]) -> dict[str, str]:
    return {r["filename"]: "test" if r["recorder"] in test_recorders else "train" for r in rows}


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
    splits: dict[str, str],
    copy_audio: bool = True,
) -> int:
    ambient: list[dict] = []
    personal: dict[str, list[dict]] = {}
    missing = 0
    for row in rows:
        split = splits[row["filename"]]
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


def summarise(rows: list[dict], splits: dict[str, str]) -> None:
    by_label = Counter((r["dataset"], r["label"]) for r in rows)
    by_split = Counter((r["dataset"], r["label"], splits[r["filename"]]) for r in rows)
    by_condition = Counter((r["dataset"], r["label"], condition(r), splits[r["filename"]]) for r in rows)
    print("\nclips per label (train/val/test), and per recording condition:")
    for dataset, label in sorted(by_label):
        tr, va, te = (by_split[(dataset, label, s)] for s in ("train", "val", "test"))
        print(f"  {dataset:8s} {label:15s} {by_label[(dataset, label)]:4d}  ({tr}/{va}/{te})")
        for cond in sorted({c for d, lbl, c, _ in by_condition if (d, lbl) == (dataset, label)}):
            ctr, cva, cte = (by_condition[(dataset, label, cond, s)] for s in ("train", "val", "test"))
            print(f"      {cond:32s} {ctr:3d}/{cva}/{cte}")


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
        "--split-by",
        choices=("label", "recorder"),
        default="label",
        help="label (default): per-label split across conditions; recorder: hold out --test-recorders.",
    )
    parser.add_argument(
        "--test-recorders",
        default=",".join(DEFAULT_TEST_RECORDERS),
        help="With --split-by recorder: comma-separated recorders held out as test (default: %(default)s).",
    )
    parser.add_argument(
        "--val-fraction", type=float, default=VAL_FRACTION,
        help="Share of each ambient label moved from train to validation (default %(default)s; 0 = none).",
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
    splits = label_splits(rows) if args.split_by == "label" else recorder_splits(rows, test_recorders)
    if args.val_fraction > 0:
        splits = val_splits(rows, splits, args.val_fraction)
    missing = organise(rows, args.source, args.dest, splits, copy_audio=not args.metadata_only)
    summarise(rows, splits)
    if missing:
        sys.exit(f"{missing} clips were missing from {args.source}")


if __name__ == "__main__":
    main()
