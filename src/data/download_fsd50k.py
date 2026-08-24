"""Resumable FSD50K downloader and target-label coverage report."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import zipfile
from pathlib import Path
from urllib.request import urlopen

from src.config import RAW_DATA_DIR, SOUND_CLASSES
from src.data.common import count_audio_files, download_file

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
        if split_parts and not combined.exists():
            if dry_run:
                print(f"DRY RUN reassemble: {archive.name} -> {combined.name}")
            elif shutil.which("zip"):
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


def run(dry_run: bool = False) -> dict[str, int]:
    root = RAW_DATA_DIR / "fsd50k"
    root.mkdir(parents=True, exist_ok=True)
    files = _record_files(dry_run)
    if dry_run:
        print("DRY RUN expected Zenodo assets: FSD50K.dev_audio.z01-z05/.zip, FSD50K.eval_audio parts/.zip, ground_truth.zip, metadata.zip")
    for item in files:
        name = item["key"]
        if any(token in name.lower() for token in ("audio", "ground_truth", "metadata")):
            download_file(item["links"]["self"], root / name, dry_run)
    _extract_archives(root, dry_run)
    return coverage_report(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(args.dry_run)
