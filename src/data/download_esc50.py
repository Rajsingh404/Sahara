"""Download ESC-50 (Piczak 2015, CC BY-NC 3.0; fine for academic use): mapped categories as targets, the rest as background.

The archive (~600 MB) is fetched once from GitHub and extracted to data/raw/esc50/ as
    audio/*.wav  (5 s, 44.1 kHz mono; all 2,000 clips: mapped categories are targets,
                  the other 44 categories are background)
    meta/esc50.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import INTERIM_DATA_DIR, SOUND_CLASSES, get_raw_dir
from src.data.common import download_file
from src.data.label_map import esc50_categories_for_class

ESC50_URL = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"


def mapped_categories() -> dict[str, str]:
    return {category: name for name in SOUND_CLASSES for category in esc50_categories_for_class(name)}


def run(dry_run: bool = False) -> dict[str, int]:
    root = get_raw_dir("esc50")
    archive = INTERIM_DATA_DIR / "ESC-50-master.zip"
    download_file(ESC50_URL, archive, dry_run)
    if dry_run:
        return {}
    categories = mapped_categories()
    counts = {name: 0 for name in SOUND_CLASSES}
    with zipfile.ZipFile(archive) as zf:
        meta = zf.read("ESC-50-master/meta/esc50.csv").decode("utf-8")
        (root / "meta").mkdir(parents=True, exist_ok=True)
        (root / "meta" / "esc50.csv").write_text(meta, encoding="utf-8")
        (root / "audio").mkdir(parents=True, exist_ok=True)
        for row in csv.DictReader(meta.splitlines()):
            out = root / "audio" / row["filename"]
            if not out.exists():
                out.write_bytes(zf.read(f"ESC-50-master/audio/{row['filename']}"))
            target = categories.get(row["category"], "background")
            counts[target] = counts.get(target, 0) + 1
    print("ESC-50 clips per class:", counts)
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    run(parser.parse_args().dry_run)
