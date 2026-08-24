"""SAHARA dataset-download entrypoint."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Supports both `python src/data/download.py` and `python -m src.data.download`.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import RAW_DATA_DIR, SOUND_CLASSES
from src.data import download_audioset, download_desed, download_fsd50k
from src.data.common import disk_usage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("fsd50k", "audioset", "desed", "all"), required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit-per-class", type=int, help="AudioSet smoke-test cap per target class.")
    parser.add_argument("--retry-failed", action="store_true", help="Retry failed AudioSet manifest entries.")
    parser.add_argument("--soundbank", action="store_true", help="Also download optional DESED synthetic soundbank.")
    args = parser.parse_args()
    reports: dict[str, dict] = {}
    if args.dataset in ("fsd50k", "all"):
        reports["fsd50k"] = download_fsd50k.run(args.dry_run)
    if args.dataset in ("audioset", "all"):
        reports["audioset"] = download_audioset.run(args.limit_per_class, args.retry_failed, args.dry_run)
    if args.dataset in ("desed", "all"):
        reports["desed"] = download_desed.run(args.soundbank, args.dry_run)
    print("\nDataset disk usage")
    for name in ("fsd50k", "audioset", "desed"):
        print(f"{name}: {disk_usage(RAW_DATA_DIR / name)}")
    if args.dataset == "all":
        print("\nCombined target-class counts (metadata-derived; DESED is soundscape-level and not class-annotated here)")
        print("class\tfsd50k\taudioset\tcombined")
        for target in SOUND_CLASSES:
            fsd = reports.get("fsd50k", {}).get(target, 0)
            audio = reports.get("audioset", {}).get(target, 0)
            total = fsd + audio
            print(f"{target}\t{fsd}\t{audio}\t{total}{'  LOW COVERAGE' if total < 100 else ''}")


if __name__ == "__main__":
    main()
