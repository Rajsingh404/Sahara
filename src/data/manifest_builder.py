"""Build unified train/val/test manifests from whatever raw dataset folders exist."""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import (
    METADATA_DIR,
    RANDOM_SEED,
    RAW_DATA_DIR,
    SOUND_CLASSES,
    TEST_SPLIT,
    VALIDATION_SPLIT,
    get_raw_dir,
)
from src.data.label_map import BACKGROUND_LABEL, CLASS_LABEL_MAP, fsd50k_labels_for_class
from src.preprocessing.audio_utils import get_duration_sec

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MANIFEST_FIELDS = ["filepath", "label", "source_dataset", "source_type", "duration_sec", "recording_id"]
AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def _is_cloud_mount(path: Path) -> float:
    """Heuristic: return True if path is under a Google Drive / CloudStorage FUSE mount."""
    s = str(path.resolve())
    return "CloudStorage" in s or "GoogleDrive" in s or "/Volumes/" in s


def _duration_best_effort(filepath: Path, csv_duration: float | None = None) -> float:
    """Return duration without blocking on cloud-mounted files.

    Priority:
      1. Pre-computed value from a metadata CSV (free — no I/O).
      2. soundfile header read if the file is local.
      3. 0.0 fallback (duration is not used in training; it's only informational).
    """
    if csv_duration is not None and csv_duration > 0:
        return csv_duration
    if _is_cloud_mount(filepath):
        # Avoid blocking FUSE call — duration is informational only.
        return 0.0
    return get_duration_sec(filepath)


def _recording_id(source: str, filepath: Path, extra: str = "") -> str:
    """Stable group id for stratified splits (same session stays in one split)."""
    stem = filepath.stem
    if source == "audioset":
        # video_id is embedded before the timestamp tokens.
        parts = stem.split("_")
        return f"audioset:{parts[0]}" if parts else f"audioset:{stem}"
    if source == "fsd50k":
        return f"fsd50k:{stem}"
    if source == "indian_ambient":
        return f"indian_ambient:{extra or stem}"
    if source == "desed":
        return f"desed:{filepath.parent.name}:{stem}"
    if source == "inoise":
        return f"inoise:{filepath.parent.name}:{stem}"
    if source == "synthetic_mixed":
        return f"synthetic_mixed:{stem}"
    return f"{source}:{stem}"


def _load_fsd50k_durations(root: Path) -> dict[str, float]:
    """Read per-file durations from FSD50K.metadata/collection/dev_clips_info_FSD50K.json
    or fall back to empty dict (duration will be 0.0 — informational only)."""
    durations: dict[str, float] = {}
    # FSD50K metadata clip info is sometimes distributed as JSON or CSV.
    meta_dir = root / "FSD50K.metadata" / "collection"
    if not meta_dir.exists():
        meta_dir = root / "FSD50K.metadata"
    for json_path in meta_dir.glob("*clips_info*.json") if meta_dir.exists() else []:
        try:
            import json
            with json_path.open(encoding="utf-8") as handle:
                data = json.load(handle)
            for clip_id, info in data.items():
                if isinstance(info, dict) and "duration" in info:
                    durations[str(clip_id)] = float(info["duration"])
        except Exception as exc:
            logger.debug("fsd50k metadata JSON parse error: %s", exc)
    return durations


def _scan_fsd50k(root: Path) -> list[dict]:
    """Scan FSD50K ground_truth CSVs and emit one manifest row per matched clip.

    Design note: this function avoids ALL filesystem stat/exists/resolve calls
    beyond reading the two CSV files.  FSD50K has a fixed, well-known layout;
    doing per-file or per-directory stats over a Google Drive FUSE mount with
    40K+ files causes minutes of blocking I/O.  We trust the CSV as the
    authoritative file inventory.
    """
    rows: list[dict] = []

    label_to_class: dict[str, str] = {}
    for class_name in SOUND_CLASSES:
        for native in CLASS_LABEL_MAP[class_name]["fsd50k"]:
            label_to_class[native] = class_name

    # Pre-load durations from metadata JSON if present (zero filesystem overhead).
    csv_durations = _load_fsd50k_durations(root)

    # FSD50K uses a fixed directory layout.  Try both the official prefixed names
    # (FSD50K.ground_truth, FSD50K.dev_audio, …) and unprefixed fallbacks.
    # LOCAL CACHE PRIORITY: data/interim/fsd50k_ground_truth/ is checked first.
    # Copy CSVs there once with: cp data/raw/fsd50k/FSD50K.ground_truth/*.csv data/interim/fsd50k_ground_truth/
    # This avoids streaming large CSVs over Google Drive FUSE on every manifest rebuild.
    local_gt_cache = METADATA_DIR.parent / "interim" / "fsd50k_ground_truth"
    split_pairs = [
        ("dev.csv",  "FSD50K.ground_truth", "FSD50K.dev_audio",  "dev_audio"),
        ("eval.csv", "FSD50K.ground_truth", "FSD50K.eval_audio", "eval_audio"),
    ]
    found_gt = False
    for split_name, gt_prefix, audio_prefix, audio_plain in split_pairs:
        # Build candidate CSV paths: local cache first, then Drive paths.
        gt_candidates = [
            local_gt_cache / split_name,            # fast local copy
            root / gt_prefix / split_name,          # official FSD50K symlink
            root / "ground_truth" / split_name,     # unprefixed fallback
        ]
        audio_candidates = [root / audio_prefix, root / audio_plain]
        audio_dir = audio_candidates[0]  # Use prefixed by default (official FSD50K layout)

        # Try each candidate path in priority order until one opens.
        # Skip zero-byte files (e.g. an incomplete interim cache copy).
        handle = None
        for idx, csv_candidate in enumerate(gt_candidates):
            try:
                if csv_candidate.exists() and csv_candidate.stat().st_size == 0:
                    logger.debug("fsd50k: skipping empty file %s", csv_candidate)
                    continue
                handle = csv_candidate.open(newline="", encoding="utf-8")
                csv_path = csv_candidate
                logger.debug("fsd50k: opened %s", csv_candidate)
                break
            except OSError:
                continue
        if handle is None:
            logger.debug("fsd50k: %s not found in any candidate path, skipping", split_name)
            continue

        found_gt = True
        with handle:
            for row in csv.DictReader(handle):
                labels = [lbl.strip() for lbl in row.get("labels", "").split(",") if lbl.strip()]
                matched = {label_to_class[lbl] for lbl in labels if lbl in label_to_class}
                fname = row.get("fname", "").strip()
                if not fname or not matched:
                    continue
                # Construct path pointing to the Drive audio directory — no stat/exists call.
                filepath = audio_dir / f"{fname}.wav"
                csv_dur = csv_durations.get(fname)
                # _duration_best_effort returns 0.0 for cloud mounts (no I/O).
                duration = _duration_best_effort(filepath, csv_dur)
                for class_name in matched:
                    rows.append(
                        {
                            "filepath": str(filepath),
                            "label": class_name,
                            "source_dataset": "fsd50k",
                            "duration_sec": duration,
                            "recording_id": _recording_id("fsd50k", filepath),
                        }
                    )

    if not found_gt:
        logger.info("fsd50k: skipped — ground_truth CSVs not found")
    else:
        logger.info("fsd50k: %d manifest rows", len(rows))
    return rows


def _scan_audioset(root: Path) -> list[dict]:
    rows: list[dict] = []
    if not root.exists() or not any(root.iterdir()):
        logger.info("audioset: skipped — not yet downloaded")
        return rows
    found = 0
    for class_name in SOUND_CLASSES:
        class_dir = root / class_name
        if not class_dir.is_dir():
            continue
        for filepath in class_dir.rglob("*"):
            if filepath.suffix.lower() not in AUDIO_SUFFIXES:
                continue
            rows.append(
                {
                    "filepath": str(filepath.resolve()),
                    "label": class_name,
                    "source_dataset": "audioset",
                    "duration_sec": _duration_best_effort(filepath),
                    "recording_id": _recording_id("audioset", filepath),
                }
            )
            found += 1
    if found == 0:
        logger.info("audioset: skipped — folder exists but no audio files yet")
    else:
        logger.info("audioset: %d manifest rows", len(rows))
    return rows


def _scan_desed(root: Path) -> list[dict]:
    """DESED real soundscapes are background negatives unless strongly labeled."""
    rows: list[dict] = []
    real = root / "real"
    if not real.exists() or not any(real.rglob("*")):
        logger.info("desed: skipped — not yet downloaded")
        return rows

    desed_label_map: dict[str, str] = {}
    for class_name in SOUND_CLASSES:
        for native in CLASS_LABEL_MAP[class_name]["desed"]:
            desed_label_map[native.lower()] = class_name

    for filepath in real.rglob("*"):
        if filepath.suffix.lower() not in AUDIO_SUFFIXES:
            continue
        lower = str(filepath).lower()
        matched_class = None
        for native, class_name in desed_label_map.items():
            if native.lower().replace("_", " ") in lower.replace("_", " "):
                matched_class = class_name
                break
        label = matched_class or BACKGROUND_LABEL
        rows.append(
            {
                "filepath": str(filepath.resolve()),
                "label": label,
                "source_dataset": "desed",
                "duration_sec": _duration_best_effort(filepath),
                "recording_id": _recording_id("desed", filepath),
            }
        )
    logger.info("desed: %d manifest rows", len(rows))
    return rows


def _scan_indian_ambient(root: Path) -> list[dict]:
    rows: list[dict] = []
    meta_path = root / "metadata.csv"
    if not meta_path.exists():
        logger.info("indian_ambient: skipped — metadata.csv not found")
        return rows

    native_to_class = {native: cls for cls in SOUND_CLASSES for native in CLASS_LABEL_MAP[cls]["indian_ambient"]}
    with meta_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            filename = row.get("filename", "").strip()
            if not filename:
                continue
            is_bg = str(row.get("is_background", "")).lower() in {"1", "true", "yes"}
            if is_bg:
                filepath = root / "background" / filename
                label = BACKGROUND_LABEL
            else:
                filepath = root / "target_classes" / filename
                raw_class = row.get("class", "").strip()
                label = native_to_class.get(raw_class, raw_class)
            if not filepath.exists():
                logger.warning("indian_ambient: missing file %s", filepath)
                continue
            session = row.get("timestamp", "") or row.get("notes", "") or filename
            rows.append(
                {
                    "filepath": str(filepath.resolve()),
                    "label": label,
                    "source_dataset": "indian_ambient",
                    "duration_sec": _duration_best_effort(filepath),
                    "recording_id": _recording_id("indian_ambient", filepath, session),
                }
            )
    logger.info("indian_ambient: %d manifest rows", len(rows))
    return rows


def _scan_inoise(root: Path) -> list[dict]:
    rows: list[dict] = []
    if not root.exists() or not any(root.rglob("*")):
        logger.info("inoise: skipped — not yet downloaded")
        return rows

    for filepath in root.rglob("*"):
        if filepath.suffix.lower() not in AUDIO_SUFFIXES:
            continue
        rows.append(
            {
                "filepath": str(filepath.resolve()),
                "label": BACKGROUND_LABEL,
                "source_dataset": "inoise",
                "source_type": "real",
                "duration_sec": _duration_best_effort(filepath),
                "recording_id": _recording_id("inoise", filepath),
            }
        )
    logger.info("inoise: %d manifest rows", len(rows))
    return rows


def _scan_synthetic_mixed(root: Path) -> list[dict]:
    rows: list[dict] = []
    meta_path = root / "synthetic_metadata.csv"
    if meta_path.exists():
        with meta_path.open(newline="", encoding="utf-8") as handle:
            for r in csv.DictReader(handle):
                rel_p = r.get("relative_filepath") or f"{r['label']}/{r['filename']}"
                filepath = root / rel_p
                rows.append(
                    {
                        "filepath": str(filepath.resolve()),
                        "label": r["label"],
                        "source_dataset": "synthetic_mixed",
                        "source_type": "synthetic_mixed",
                        "duration_sec": _duration_best_effort(filepath),
                        "recording_id": _recording_id("synthetic_mixed", filepath),
                    }
                )
    elif root.exists():
        for filepath in root.rglob("*"):
            if filepath.suffix.lower() not in AUDIO_SUFFIXES or filepath.name == "synthetic_metadata.csv":
                continue
            label = filepath.parent.name
            if label not in SOUND_CLASSES and label != BACKGROUND_LABEL:
                continue
            rows.append(
                {
                    "filepath": str(filepath.resolve()),
                    "label": label,
                    "source_dataset": "synthetic_mixed",
                    "source_type": "synthetic_mixed",
                    "duration_sec": _duration_best_effort(filepath),
                    "recording_id": _recording_id("synthetic_mixed", filepath),
                }
            )
    if not rows:
        logger.info("synthetic_mixed: skipped — not yet generated")
    else:
        logger.info("synthetic_mixed: %d manifest rows", len(rows))
    return rows


SCANNERS = {
    "fsd50k": _scan_fsd50k,
    "audioset": _scan_audioset,
    "desed": _scan_desed,
    "indian_ambient": _scan_indian_ambient,
    "inoise": _scan_inoise,
    "synthetic_mixed": _scan_synthetic_mixed,
}


def build_manifest(raw_dir: Path | None = None) -> list[dict]:
    """Scan all known dataset sources and build a unified manifest.

    Each dataset is looked up via ``config.get_raw_dir(name)``, which
    checks SAHARA_DRIVE_DATASETS first, then falls back to data/raw/<name>.
    Unknown subdirectories under raw_dir are still logged as skipped so
    that ad-hoc folders don't cause silent failures.
    """
    rows: list[dict] = []
    # Always scan the six known datasets via get_raw_dir (Drive-aware).
    for name, scanner in SCANNERS.items():
        dataset_dir = get_raw_dir(name)
        rows.extend(scanner(dataset_dir))

    for row in rows:
        if "source_type" not in row:
            row["source_type"] = "synthetic_mixed" if row.get("source_dataset") == "synthetic_mixed" else "real"

    # Also scan any extra subdirs under raw_dir that aren't in SCANNERS
    # (future datasets, one-off folders, etc.) — log them as skipped.
    scan_root = raw_dir or RAW_DATA_DIR
    if scan_root.exists():
        for child in sorted(scan_root.iterdir()):
            if child.is_dir() and child.name not in SCANNERS:
                logger.info("%s: skipped — no scanner registered", child.name)
    return rows


def _stratified_split(rows: list[dict], train_frac: float, val_frac: float, seed: int) -> tuple[list, list, list]:
    rng = np.random.default_rng(seed)
    by_class_rec: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_class_rec[row["label"]][row["recording_id"]].append(row)

    train, val, test = [], [], []
    for label, recordings in by_class_rec.items():
        rec_ids = list(recordings.keys())
        rng.shuffle(rec_ids)
        n = len(rec_ids)
        if n == 0:
            continue
        n_val = max(1, int(round(n * val_frac))) if n > 2 else (1 if n == 3 else 0)
        n_test = max(1, int(round(n * (1.0 - train_frac - val_frac)))) if n > 2 else (1 if n == 3 else 0)
        if n_val + n_test >= n:
            n_val = 1 if n >= 2 else 0
            n_test = 1 if n >= 3 else 0
        val_ids = set(rec_ids[:n_val])
        test_ids = set(rec_ids[n_val : n_val + n_test])
        for rec_id, rec_rows in recordings.items():
            if rec_id in val_ids:
                val.extend(rec_rows)
            elif rec_id in test_ids:
                test.extend(rec_rows)
            else:
                train.extend(rec_rows)
    return train, val, test


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for r in rows:
        if "source_type" not in r:
            r["source_type"] = "synthetic_mixed" if r.get("source_dataset") == "synthetic_mixed" else "real"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def print_coverage_table(rows: list[dict]) -> None:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    sources = sorted({row["source_dataset"] for row in rows}) or ["(none)"]
    labels = list(SOUND_CLASSES) + [BACKGROUND_LABEL]
    for row in rows:
        counts[row["label"]][row["source_dataset"]] += 1

    header = ["class", *sources, "total"]
    print("\nPer-class coverage (clips)")
    print("\t".join(header))
    for label in labels:
        row_counts = [str(counts[label].get(src, 0)) for src in sources]
        total = sum(counts[label].get(src, 0) for src in sources)
        flag = "  LOW" if label in SOUND_CLASSES and total < 100 else ""
        print("\t".join([label, *row_counts, str(total)]) + flag)


def run(raw_dir: Path | None = None) -> dict[str, int]:
    rows = build_manifest(raw_dir)
    if not rows:
        logger.warning("No manifest rows found — check data/raw/ contents")
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    write_manifest(METADATA_DIR / "full_manifest.csv", rows)

    train_frac = 1.0 - VALIDATION_SPLIT - TEST_SPLIT
    train, val, test = _stratified_split(rows, train_frac, VALIDATION_SPLIT, RANDOM_SEED)
    write_manifest(METADATA_DIR / "train_manifest.csv", train)
    write_manifest(METADATA_DIR / "val_manifest.csv", val)
    write_manifest(METADATA_DIR / "test_manifest.csv", test)

    print(f"\nManifest totals: full={len(rows)} train={len(train)} val={len(val)} test={len(test)}")
    print_coverage_table(rows)
    return {"full": len(rows), "train": len(train), "val": len(val), "test": len(test)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    args = parser.parse_args()
    run(args.raw_dir)


if __name__ == "__main__":
    main()
