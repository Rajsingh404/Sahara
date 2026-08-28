"""Resumable FSD50K downloader and target-label coverage report."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path
from urllib.request import urlopen

from src.config import RAW_DATA_DIR, SOUND_CLASSES
from src.data.common import count_audio_files, download_file

# Default Google Drive sync path (override with SAHARA_FSD50K_GDRIVE or --gdrive-path).
DEFAULT_GDRIVE_FSD50K = Path.home() / (
    "Library/CloudStorage/GoogleDrive-singhraj04.jan@gmail.com/My Drive/SAHARA/Datasets/FSD50K"
)
FSD50K_ITEMS = ("FSD50K.dev_audio", "FSD50K.eval_audio", "FSD50K.ground_truth", "FSD50K.metadata")

ZENODO_RECORD = "https://zenodo.org/api/records/4060432"
FSD_LABELS = {
    "smoke_alarm": ["Smoke_detector_smoke_alarm"], "doorbell": ["Doorbell"],
    "siren": ["Siren"], "knocking": ["Knock"], "dog_bark": ["Bark"],
    "baby_cry": ["Baby_cry_infant_cry"], "glass_break": ["Glass"],
    "appliance_beep": ["Beep_buzzer", "Alarm", "Microwave_oven"],
}


def _record_files(dry_run: bool) -> list[dict]:
    if dry_run:
        print(f"DRY RUN query Zenodo record 4060432 ({ZENODO_RECORD})")
        return []
    with urlopen(ZENODO_RECORD) as response:
        return json.load(response)["files"]


def _extract_archives(root: Path, dry_run: bool) -> None:
    for archive in root.glob("*.zip"):
        # Split archives are joined with zip(1), preserving the original parts.
        split_parts = list(root.glob(archive.stem + ".z*"))
        combined = archive.with_name(archive.stem + ".combined.zip")
        if split_parts and (not combined.exists() or combined.stat().st_size == 0):
            if dry_run:
                print(f"DRY RUN reassemble: {archive.name} -> {combined.name}")
            elif shutil.which("zip"):
                if combined.exists():
                    # A prior interrupted zip(1) invocation leaves an empty output.
                    combined.unlink()
                subprocess.run(["zip", "-s", "0", str(archive), "--out", str(combined)], check=True)
        source = combined if combined.exists() else archive
        marker = root / (source.name + ".extracted")
        if source.exists() and not marker.exists():
            if dry_run:
                print(f"DRY RUN extract: {source.name}")
            else:
                with zipfile.ZipFile(source) as zf:
                    zf.extractall(root)
                marker.touch()


def coverage_report(root: Path) -> dict[str, int]:
    counts = {name: 0 for name in SOUND_CLASSES}
    csvs = list(root.rglob("dev.csv")) + list(root.rglob("eval.csv"))
    for path in csvs:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                labels = row.get("labels", "").split(",")
                for target, source_labels in FSD_LABELS.items():
                    if any(label in source_labels for label in labels):
                        counts[target] += 1
    print("\nFSD50K target-label coverage")
    print("class\tclips\tstatus")
    for name, count in counts.items():
        print(f"{name}\t{count}\t{'LOW COVERAGE' if count < 100 else 'OK'}")
    if not csvs:
        print("Metadata CSVs not yet present; coverage will run after extraction.")
    return counts


def link_from_gdrive(gdrive_path: Path | None = None, dry_run: bool = False) -> bool:
    """Symlink extracted FSD50K folders from a local Google Drive sync into data/raw/fsd50k/."""
    source = Path(os.environ.get("SAHARA_FSD50K_GDRIVE", gdrive_path or DEFAULT_GDRIVE_FSD50K))
    dest = RAW_DATA_DIR / "fsd50k"
    if not source.exists():
        print(f"Google Drive FSD50K not found at {source}")
        return False
    dest.mkdir(parents=True, exist_ok=True)
    linked = 0
    for item in FSD50K_ITEMS:
        src_item = source / item
        if not src_item.exists():
            print(f"SKIP missing: {src_item}")
            continue
        dest_item = dest / item
        if dry_run:
            print(f"DRY RUN link {src_item} -> {dest_item}")
        else:
            if dest_item.is_symlink() or dest_item.exists():
                dest_item.unlink(missing_ok=True)
            dest_item.symlink_to(src_item, target_is_directory=True)
            print(f"LINKED {dest_item.name} -> {src_item}")
        linked += 1
    return linked >= 3


def run(dry_run: bool = False, gdrive_path: Path | None = None, use_gdrive: bool = False) -> dict[str, int]:
    if use_gdrive or (DEFAULT_GDRIVE_FSD50K / "FSD50K.ground_truth").exists():
        if link_from_gdrive(gdrive_path, dry_run):
            root = RAW_DATA_DIR / "fsd50k"
            if (root / "FSD50K.ground_truth").exists() or (root / "FSD50K.ground_truth").is_symlink():
                return coverage_report(root)
    root = RAW_DATA_DIR / "fsd50k"
    root.mkdir(parents=True, exist_ok=True)
    files = _record_files(dry_run)
    if dry_run:
        print("DRY RUN expected Zenodo assets: FSD50K.dev_audio.z01-z05/.zip, FSD50K.eval_audio parts/.zip, ground_truth.zip, metadata.zip")
    for item in files:
        name = item["key"]
        if any(token in name.lower() for token in ("audio", "ground_truth", "metadata")):
            destination = root / name
            # A terminated process can leave a shorter file under its final name.
            # Restore it to the resumable `.part` path after validating against
            # Zenodo's published byte count.
            if (not dry_run and destination.exists() and
                    destination.stat().st_size < item["size"]):
                partial = destination.with_suffix(destination.suffix + ".part")
                if not partial.exists():
                    print(f"INCOMPLETE {name}; resuming from {destination.stat().st_size:,} bytes")
                    destination.replace(partial)
            download_file(item["links"]["self"], destination, dry_run)
    _extract_archives(root, dry_run)
    return coverage_report(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(args.dry_run)
