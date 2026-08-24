"""Shared primitives for resumable dataset downloads."""
from __future__ import annotations

import csv
import shutil
import urllib.request
from pathlib import Path
from typing import Iterable

AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def download_file(url: str, destination: Path, dry_run: bool = False) -> bool:
    """Download once; a non-empty existing file is treated as complete."""
    if destination.exists() and destination.stat().st_size > 0:
        print(f"SKIP existing: {destination.name}")
        return False
    if dry_run:
        print(f"DRY RUN download: {url} -> {destination}")
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"DOWNLOAD {url}")
    urllib.request.urlretrieve(url, temporary)
    temporary.replace(destination)
    return True


def count_audio_files(path: Path) -> int:
    return sum(1 for item in path.rglob("*") if item.is_file() and item.suffix.lower() in AUDIO_SUFFIXES)


def disk_usage(path: Path) -> str:
    if not path.exists():
        return "0 B"
    total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if total < 1024 or unit == "TiB":
            return f"{total:.1f} {unit}"
        total /= 1024
    return "0 B"


def write_rows(path: Path, rows: Iterable[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
