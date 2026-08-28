"""Inspect removable local dataset artifacts without deleting them by default.

Use this before removing raw archives after feature extraction.  The command
prints candidate paths and only performs deletion with explicit --apply.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.config import RAW_DATA_DIR

ARCHIVE_SUFFIXES = (".zip", ".z01", ".z02", ".z03", ".z04", ".z05")


def candidates(root: Path = RAW_DATA_DIR / "fsd50k") -> list[Path]:
    return sorted(path for path in root.iterdir() if path.is_file() and path.name.endswith(ARCHIVE_SUFFIXES))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Delete listed archive files after review.")
    args = parser.parse_args()
    paths = candidates()
    total = sum(path.stat().st_size for path in paths)
    action = "Would remove" if not args.apply else "Removing"
    print(f"{action} {len(paths)} archive files ({total / 1024**3:.1f} GiB):")
    for path in paths:
        print(path)
    if args.apply:
        for path in paths:
            path.unlink()


if __name__ == "__main__":
    main()
