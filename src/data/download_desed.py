"""Download DESED real soundscapes; the soundbank is deliberately opt-in."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.config import RAW_DATA_DIR
from src.data.common import count_audio_files


def _call_download(function, destination: Path) -> None:
    """Cope with minor desed API keyword variations between releases."""
    for kwargs in ({"output_folder": str(destination)}, {"download_folder": str(destination)}, {"path": str(destination)}):
        try:
            function(**kwargs)
            return
        except TypeError:
            continue
    function(str(destination))


def run(soundbank: bool = False, dry_run: bool = False) -> dict[str, int]:
    root = RAW_DATA_DIR / "desed"
    real, bank = root / "real", root / "soundbank"
    if dry_run:
        print(f"DRY RUN desed.download_real -> {real}")
        if soundbank:
            print(f"DRY RUN desed.download_desed_soundbank -> {bank}")
        return {"real": 0, "soundbank": 0}
    import desed
    real.mkdir(parents=True, exist_ok=True)
    _call_download(desed.download_real, real)
    if soundbank:
        bank.mkdir(parents=True, exist_ok=True)
        _call_download(desed.download_desed_soundbank, bank)
    subsets = {"unlabeled_in_domain": 0, "weakly_labeled_train": 0, "strongly_labeled_validation": 0}
    for path in real.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".wav", ".flac", ".ogg"}:
            lower = str(path).lower()
            for subset in subsets:
                if all(word in lower for word in subset.split("_")):
                    subsets[subset] += 1
    print("DESED real subsets:", subsets)
    print(f"DESED soundbank files: {count_audio_files(bank)}")
    if not all(subsets.values()):
        print("WARNING: one or more expected DESED subsets is empty; some remote real-world files are known to fail resolution.")
    return {"real": count_audio_files(real), "soundbank": count_audio_files(bank)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--soundbank", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(args.soundbank, args.dry_run)
