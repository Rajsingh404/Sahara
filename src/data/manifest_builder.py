"""Build unified train/val/test manifests from whatever raw dataset folders exist."""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import (
    FSD50K_BACKGROUND_SAMPLE,
    METADATA_DIR,
    RANDOM_SEED,
    RAW_DATA_DIR,
    SOUND_CLASSES,
    TEST_SPLIT,
    VALIDATION_SPLIT,
    get_raw_dir,
)
from src.data.label_map import BACKGROUND_LABEL, CLASS_LABEL_MAP, DESED_EXCLUDED_EVENTS, FSD50K_BACKGROUND_EXCLUDE
from src.preprocessing.audio_utils import get_duration_sec

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MANIFEST_FIELDS = ["filepath", "label", "source_dataset", "source_type", "duration_sec", "recording_id", "recorder"]
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
    """Group id for splitting: every row sharing it lands in the same split.

    AudioSet and DESED clips are both cut from YouTube videos, and FSD50K and ESC-50 clips
    both come from Freesound, so the id is the upstream video / Freesound id. A clip that
    appears in two datasets (or under two labels) therefore cannot straddle train and test.
    """
    stem = filepath.stem
    if source == "audioset":
        return f"yt:{stem[:11]}"  # YouTube ids are 11 chars and may contain "_"
    if source == "desed":
        return f"yt:{stem.removeprefix('Y')[:11]}"
    if source == "fsd50k":
        return f"fs:{stem}"
    if source == "esc50":
        return f"fs:{extra}"  # ESC-50 src_file is the Freesound clip id
    if source == "indian_ambient":
        return f"indian_ambient:{extra or stem}"
    if source == "inoise":
        return f"inoise:{filepath.parent.name}:{stem}"
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
    background_candidates: list[Path] = []
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
                if fname and not matched and not set(labels) & FSD50K_BACKGROUND_EXCLUDE:
                    background_candidates.append(audio_dir / f"{fname}.wav")
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
        return rows
    rng = np.random.default_rng(RANDOM_SEED)
    # A seeded permutation: a larger FSD50K_BACKGROUND_SAMPLE always contains every smaller one.
    picked = sorted(rng.permutation(len(background_candidates))[:FSD50K_BACKGROUND_SAMPLE]) if background_candidates else []
    for i in picked:
        filepath = background_candidates[i]
        rows.append(
            {
                "filepath": str(filepath),
                "label": BACKGROUND_LABEL,
                "source_dataset": "fsd50k",
                "duration_sec": csv_durations.get(filepath.stem, 0.0),
                "recording_id": _recording_id("fsd50k", filepath),
            }
        )
    logger.info("fsd50k: %d manifest rows (%d background sampled from %d candidates)", len(rows), len(picked),
                len(background_candidates))
    return rows


def _audioset_segment_labels(root: Path, wanted: set[str]) -> dict[tuple[str, float], set[str]]:
    """Positive labels per (video id, start) from the AudioSet segment CSVs, for wanted videos only."""
    index: dict[tuple[str, float], set[str]] = {}
    for name in ("eval_segments.csv", "balanced_train_segments.csv", "unbalanced_train_segments.csv"):
        path = root / "metadata" / name
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("#") or line[:11] not in wanted:
                    continue
                parts = [p.strip() for p in line.split(",", 3)]
                index[(parts[0], float(parts[1]))] = set(parts[3].strip().strip('"').split(","))
    return index


def _scan_audioset(root: Path) -> list[dict]:
    """One row per clip and per target class its AudioSet labels actually carry.

    The class folder a clip was downloaded into is not trusted on its own: the clip's
    positive labels are looked up in the segment CSVs and matched against label_map mids.
    """
    rows: list[dict] = []
    if not root.exists() or not any(root.iterdir()):
        logger.info("audioset: skipped — not yet downloaded")
        return rows
    files = [f for c in SOUND_CLASSES if (root / c).is_dir() for f in (root / c).glob("*.wav")]
    index = _audioset_segment_labels(root, {f.stem[:11] for f in files})
    mids = {c: set(CLASS_LABEL_MAP[c]["audioset"]) for c in SOUND_CLASSES}
    seen: set[tuple[str, str]] = set()
    dropped = 0
    for filepath in files:
        start_tokens = filepath.stem[12:].split("_")
        labels = index.get((filepath.stem[:11], float(f"{start_tokens[0]}.{start_tokens[1]}")), set())
        matched = [c for c in SOUND_CLASSES if labels & mids[c]]
        if not matched:
            dropped += 1
            continue
        # The same segment can sit in two class folders; keep one copy per class.
        for class_name in matched:
            key = (filepath.stem, class_name)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "filepath": str(filepath.resolve()),
                    "label": class_name,
                    "source_dataset": "audioset",
                    "duration_sec": _duration_best_effort(filepath),
                    "recording_id": _recording_id("audioset", filepath),
                }
            )
    logger.info("audioset: %d manifest rows (%d clips dropped: labels match no target class)", len(rows), dropped)
    return rows


def _scan_esc50(root: Path) -> list[dict]:
    rows: list[dict] = []
    meta = root / "meta" / "esc50.csv"
    if not meta.exists():
        logger.info("esc50: skipped — not yet downloaded")
        return rows
    category_to_class = {cat: c for c in SOUND_CLASSES for cat in CLASS_LABEL_MAP[c]["esc50"]}
    with meta.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            # Unmapped ESC-50 categories (rain, footsteps, typing, car horn, ...) are background.
            label = category_to_class.get(row["category"], BACKGROUND_LABEL)
            filepath = root / "audio" / row["filename"]
            if not filepath.exists():
                continue
            rows.append(
                {
                    "filepath": str(filepath.resolve()),
                    "label": label,
                    "source_dataset": "esc50",
                    "duration_sec": 5.0,
                    "recording_id": _recording_id("esc50", filepath, row["src_file"]),
                }
            )
    logger.info("esc50: %d manifest rows", len(rows))
    return rows


def _desed_events(real: Path) -> dict[str, set[str]]:
    """Event labels per DESED filename from the weak / strong metadata TSVs."""
    events: dict[str, set[str]] = defaultdict(set)
    for tsv in (real / "metadata").rglob("*.tsv"):
        if tsv.name.startswith("._"):
            continue
        with tsv.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                labels = row.get("event_labels") or row.get("event_label") or ""
                name = row.get("filename", "")
                if name:
                    events[name].update(lbl for lbl in labels.split(",") if lbl)
    return events


def _scan_desed(root: Path) -> list[dict]:
    """DESED soundscapes as background, minus clips that contain a SAHARA-like event.

    Clips with Dog or Alarm_bell_ringing (DESED_EXCLUDED_EVENTS) are dropped, and so is
    the unlabelled in-domain subset, which was chosen for containing DESED's events.
    """
    rows: list[dict] = []
    real = root / "real"
    if not real.exists() or not any(real.rglob("*")):
        logger.info("desed: skipped — not yet downloaded")
        return rows
    events = _desed_events(real)
    dropped = Counter()
    for filepath in sorted((real / "audio").rglob("*")):
        if filepath.suffix.lower() not in AUDIO_SUFFIXES or filepath.name.startswith("._"):
            continue
        if "unlabel" in filepath.parent.name:
            dropped["unlabelled"] += 1
            continue
        if filepath.name not in events:
            dropped["no metadata"] += 1
            continue
        if events[filepath.name] & DESED_EXCLUDED_EVENTS:
            dropped["dog/alarm event"] += 1
            continue
        rows.append(
            {
                "filepath": str(filepath.resolve()),
                "label": BACKGROUND_LABEL,
                "source_dataset": "desed",
                "duration_sec": _duration_best_effort(filepath),
                "recording_id": _recording_id("desed", filepath),
            }
        )
    logger.info("desed: %d background rows (dropped %s)", len(rows), dict(dropped))
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
            csv_duration = float(row["duration_sec"]) if row.get("duration_sec") else None
            rows.append(
                {
                    "filepath": str(filepath.resolve()),
                    "label": label,
                    "source_dataset": "indian_ambient",
                    "source_type": "self_recorded",
                    "duration_sec": _duration_best_effort(filepath, csv_duration),
                    "recording_id": _recording_id("indian_ambient", filepath, session),
                    "recorder": row.get("recorder", ""),
                    # organise_recordings.py writes the split (per label, across conditions); honoured as-is.
                    "split": row.get("split", "").strip(),
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
    """Synthetic mixes, grouped with the real clip they were made from (train only)."""
    rows: list[dict] = []
    meta_path = root / "synthetic_metadata.csv"
    if not meta_path.exists():
        logger.info("synthetic_mixed: skipped — not yet generated")
        return rows
    with meta_path.open(newline="", encoding="utf-8") as handle:
        for r in csv.DictReader(handle):
            filepath = root / r["relative_filepath"]
            if not r.get("source_recording_id"):
                logger.warning("synthetic_mixed: %s has no source_recording_id; regenerate the mixes", filepath.name)
                continue
            rows.append(
                {
                    "filepath": str(filepath.resolve()),
                    "label": r["label"],
                    "source_dataset": "synthetic_mixed",
                    "source_type": "synthetic_mixed",
                    "duration_sec": float(r.get("duration_sec") or 0.0),
                    "recording_id": r["source_recording_id"],
                    "split": "train",
                }
            )
    logger.info("synthetic_mixed: %d manifest rows", len(rows))
    return rows


SCANNERS = {
    "fsd50k": _scan_fsd50k,
    "audioset": _scan_audioset,
    "esc50": _scan_esc50,
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
    # Always scan the known datasets via get_raw_dir (Drive-aware).
    for name, scanner in SCANNERS.items():
        dataset_dir = get_raw_dir(name)
        rows.extend(scanner(dataset_dir))

    for row in rows:
        if "source_type" not in row:
            row["source_type"] = "real"

    # Also scan any extra subdirs under raw_dir that aren't in SCANNERS
    # (future datasets, one-off folders, etc.) — log them as skipped.
    scan_root = raw_dir or RAW_DATA_DIR
    if scan_root.exists():
        for child in sorted(scan_root.iterdir()):
            if child.is_dir() and child.name not in SCANNERS:
                logger.info("%s: skipped — no scanner registered", child.name)
    return rows


def _stratified_split(rows: list[dict], train_frac: float, val_frac: float, seed: int) -> tuple[list, list, list]:
    """Split by recording group, stratified by each group's rarest label.

    All rows of a recording_id go to one split, even when the group carries several
    labels (multi-label FSD50K clips, the same video in AudioSet and DESED).
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["recording_id"]].append(row)
    label_groups = Counter(label for rec in groups.values() for label in {r["label"] for r in rec})
    by_primary: dict[str, list[str]] = defaultdict(list)
    for rec_id, rec_rows in groups.items():
        primary = min({r["label"] for r in rec_rows}, key=lambda lbl: (label_groups[lbl], lbl))
        by_primary[primary].append(rec_id)

    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for label in sorted(by_primary):
        rec_ids = sorted(by_primary[label])
        rng.shuffle(rec_ids)
        n = len(rec_ids)
        n_val = int(round(n * val_frac)) if n >= 3 else 0
        n_test = int(round(n * (1.0 - train_frac - val_frac))) if n >= 3 else 0
        if n >= 3:
            n_val, n_test = max(1, n_val), max(1, n_test)
        for i, rec_id in enumerate(rec_ids):
            target = val if i < n_val else test if i < n_val + n_test else train
            target.extend(groups[rec_id])
    return train, val, test


def split_rows(rows: list[dict], train_frac: float, val_frac: float, seed: int) -> tuple[list, list, list]:
    """Rows with a fixed ``split`` (team recordings, synthetic mixes) keep it; the rest are split by group.

    Synthetic mixes stay only if their source clip is in train, so a mix can never carry
    a validation or test clip into training.
    """
    fixed = [r for r in rows if r.get("split")]
    free = [r for r in rows if not r.get("split")]
    train, val, test = _stratified_split([r for r in free if r["source_type"] != "synthetic_mixed"], train_frac, val_frac, seed)
    splits = {"train": train, "val": val, "test": test}
    synthetic = [r for r in fixed if r["source_type"] == "synthetic_mixed"]
    for row in fixed:
        if row["source_type"] != "synthetic_mixed":
            splits[row["split"]].append(row)
    train_ids = {r["recording_id"] for r in train}
    orphans = 0
    for row in synthetic:
        if row["recording_id"] not in train_ids:
            orphans += 1
            continue
        train.append(row)
    if orphans:
        logger.warning("dropped %d synthetic rows whose source clip is not in train", orphans)
    return train, val, test


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


MIN_REAL_CLIPS = 150
MIN_REAL_TEST_CLIPS = 20


def coverage_table(splits: dict[str, list[dict]]) -> str:
    """Clips per class x source x split, flagging classes below the real-clip minimums."""
    counts: dict[tuple[str, str, str], int] = Counter()
    for split_name, rows in splits.items():
        for row in rows:
            counts[(row["label"], row["source_dataset"], split_name)] += 1
    sources = sorted({src for _, src, _ in counts})
    lines = ["class           " + "".join(f"{s[:14]:>16s}" for s in sources) + "      real  real_test",
             "                " + "".join(f"{'tr/va/te':>16s}" for _ in sources)]
    for label in [*SOUND_CLASSES, BACKGROUND_LABEL]:
        cells = []
        for src in sources:
            t, v, e = (counts[(label, src, s)] for s in ("train", "val", "test"))
            cells.append(f"{f'{t}/{v}/{e}' if t + v + e else '-':>16s}")
        real = sum(n for (lbl, src, _), n in counts.items() if lbl == label and src != "synthetic_mixed")
        real_test = sum(n for (lbl, src, sp), n in counts.items() if lbl == label and sp == "test" and src != "synthetic_mixed")
        flag = ""
        if label in SOUND_CLASSES and (real < MIN_REAL_CLIPS or real_test < MIN_REAL_TEST_CLIPS):
            flag = "  << SHORT"
        lines.append(f"{label:16s}" + "".join(cells) + f"{real:10d}{real_test:11d}{flag}")
    return "\n".join(lines)


def print_coverage_table(rows: list[dict]) -> None:
    """Back-compatible single-table view (all rows counted as one split)."""
    print(coverage_table({"train": rows}))


def run(raw_dir: Path | None = None) -> dict[str, int]:
    rows = build_manifest(raw_dir)
    if not rows:
        logger.warning("No manifest rows found — check data/raw/ contents")
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    train_frac = 1.0 - VALIDATION_SPLIT - TEST_SPLIT
    train, val, test = split_rows(rows, train_frac, VALIDATION_SPLIT, RANDOM_SEED)
    write_manifest(METADATA_DIR / "full_manifest.csv", train + val + test)
    write_manifest(METADATA_DIR / "train_manifest.csv", train)
    write_manifest(METADATA_DIR / "val_manifest.csv", val)
    write_manifest(METADATA_DIR / "test_manifest.csv", test)

    table = coverage_table({"train": train, "val": val, "test": test})
    (METADATA_DIR / "coverage.txt").write_text(table + "\n", encoding="utf-8")
    print(f"\nManifest totals: full={len(train) + len(val) + len(test)} train={len(train)} val={len(val)} test={len(test)}")
    print(table)
    return {"full": len(train) + len(val) + len(test), "train": len(train), "val": len(val), "test": len(test)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    args = parser.parse_args()
    run(args.raw_dir)


if __name__ == "__main__":
    main()
