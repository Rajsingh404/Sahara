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

RECORDERS = {"advika", "mahi", "prachi", "raj"}


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
            assert row["subset"] in {"positive", "negative", "unverified"}


def test_split_keeps_each_recorder_on_one_side(tmp_path):
    rows = organise_recordings.load_label_map()
    organise_recordings.organise(rows, tmp_path, tmp_path, {"mahi"}, copy_audio=False)
    with (tmp_path / "indian_ambient" / "metadata.csv").open(newline="") as handle:
        ambient = list(csv.DictReader(handle))
    splits = {}
    for row in ambient:
        splits.setdefault(row["recorder"], set()).add(row["split"])
    assert splits["mahi"] == {"test"}
    assert all(s == {"train"} for rec, s in splits.items() if rec != "mahi")
