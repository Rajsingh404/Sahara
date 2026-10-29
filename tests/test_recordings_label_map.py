import csv
import importlib.util
from pathlib import Path

from src.config import SOUND_CLASSES
from src.data.label_map import BACKGROUND_LABEL

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "organise_recordings", REPO_ROOT / "scripts" / "organise_recordings.py"
)
organise_recordings = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(organise_recordings)

RECORDERS = {"advika", "mahi", "prachi", "raj", "team"}  # "team": second upload, recorder not given


def test_label_map_rows_are_valid():
    rows = organise_recordings.load_label_map()
    assert rows
    assert len({r["drive_file_id"] for r in rows}) == len(rows)
    assert len({(r["dataset"], r["subset"], r["filename"]) for r in rows}) == len(rows)
    for row in rows:
        assert row["recorder"] in RECORDERS
        if row["dataset"] == "ambient":
            if row["subset"] == "background":
                assert row["label"] == BACKGROUND_LABEL
            else:
                assert row["subset"] == "target_classes"
                assert row["label"] in SOUND_CLASSES
        else:
            assert row["dataset"] == "aryan"
            assert row["subset"] in {"positive", "negative"}


def test_recorder_split_keeps_each_recorder_on_one_side(tmp_path):
    rows = organise_recordings.load_label_map()
    splits = organise_recordings.recorder_splits(rows, {"mahi"})
    organise_recordings.organise(rows, tmp_path, tmp_path, splits, copy_audio=False)
    with (tmp_path / "indian_ambient" / "metadata.csv").open(newline="") as handle:
        ambient = list(csv.DictReader(handle))
    splits = {}
    for row in ambient:
        splits.setdefault(row["recorder"], set()).add(row["split"])
    assert splits["mahi"] == {"test"}
    assert all(s == {"train"} for rec, s in splits.items() if rec != "mahi")


def test_label_split_covers_every_label_and_condition():
    rows = organise_recordings.load_label_map()
    splits = organise_recordings.label_splits(rows)
    by_label, by_condition = {}, {}
    for r in rows:
        by_label.setdefault((r["dataset"], r["label"]), []).append(splits[r["filename"]])
        by_condition.setdefault((r["label"], organise_recordings.condition(r)), []).append(splits[r["filename"]])
    for key, sides in by_label.items():
        n, n_test = len(sides), sides.count("test")
        assert set(sides) == {"train", "test"}, key
        assert abs(n_test - 0.4 * n) <= 1, (key, n, n_test)  # ~40 % test per label
    # Every condition with at least 3 clips appears on both sides.
    for key, sides in by_condition.items():
        if len(sides) >= 3:
            assert set(sides) == {"train", "test"}, key


def test_label_split_keeps_identical_audio_together():
    rows = organise_recordings.load_label_map()
    splits = organise_recordings.label_splits(rows)
    dups = [r for r in rows if r["same_audio_as"]]
    assert dups
    for r in dups:
        assert splits[r["filename"]] == splits[r["same_audio_as"]]


def test_label_split_is_reproducible():
    rows = organise_recordings.load_label_map()
    assert organise_recordings.label_splits(rows) == organise_recordings.label_splits(rows)


def test_later_uploads_never_move_earlier_clips():
    rows = organise_recordings.load_label_map()
    first = [r for r in rows if (r.get("upload") or "1") == "1"]
    assert len(first) < len(rows)
    alone, together = organise_recordings.label_splits(first), organise_recordings.label_splits(rows)
    assert all(together[f] == side for f, side in alone.items())
