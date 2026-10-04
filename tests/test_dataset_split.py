from pathlib import Path

from src.data.manifest_builder import _scan_audioset, _scan_desed, _stratified_split, split_rows


def _ids(split):
    return {row["recording_id"] for row in split}


def _real(label, rec, **extra):
    return {"label": label, "recording_id": rec, "filepath": f"{rec}-{label}.wav",
            "source_dataset": "fsd50k", "source_type": "real", **extra}


def test_grouped_split_never_leaks_a_recording_id():
    rows = [
        {"label": "doorbell", "recording_id": f"session-{group}", "filepath": f"{group}-{clip}.wav"}
        for group in range(12) for clip in range(2)
    ]
    train, val, test = _stratified_split(rows, train_frac=0.7, val_frac=0.15, seed=42)
    ids = [_ids(split) for split in (train, val, test)]
    assert ids[0].isdisjoint(ids[1])
    assert ids[0].isdisjoint(ids[2])
    assert ids[1].isdisjoint(ids[2])
    assert len(train) + len(val) + len(test) == len(rows)


def test_multilabel_recording_stays_in_one_split():
    # One clip carrying two labels used to be split per label and could land in train and test.
    rows = [_real("doorbell", f"fs:{i}") for i in range(40)]
    rows += [_real("siren", f"fs:{i}") for i in range(0, 40, 2)]
    train, val, test = split_rows(rows, 0.7, 0.15, seed=1)
    ids = [_ids(split) for split in (train, val, test)]
    assert ids[0].isdisjoint(ids[1]) and ids[0].isdisjoint(ids[2]) and ids[1].isdisjoint(ids[2])
    assert len(train) + len(val) + len(test) == len(rows)


def test_recorder_split_is_honoured():
    rows = [_real("doorbell", f"fs:{i}") for i in range(20)]
    team = [
        {"label": "doorbell", "recording_id": f"indian_ambient:{who}", "filepath": f"{who}-{i}.wav",
         "source_dataset": "indian_ambient", "source_type": "self_recorded", "recorder": who,
         "split": "test" if who in {"mahi", "prachi"} else "train"}
        for who in ("advika", "raj", "mahi", "prachi") for i in range(5)
    ]
    train, val, test = split_rows(rows + team, 0.7, 0.15, seed=0)
    recorders = {name: {r["recorder"] for r in split if r.get("recorder")}
                 for name, split in (("train", train), ("val", val), ("test", test))}
    assert recorders == {"train": {"advika", "raj"}, "val": set(), "test": {"mahi", "prachi"}}


def test_synthetic_rows_only_in_train_and_follow_their_source():
    rows = [_real("siren", f"fs:{i}") for i in range(30)]
    synthetic = [
        {"label": "siren", "recording_id": f"fs:{i}", "filepath": f"syn_{i}_{v}.wav",
         "source_dataset": "synthetic_mixed", "source_type": "synthetic_mixed", "split": "train"}
        for i in range(30) for v in range(3)
    ]
    train, val, test = split_rows(rows + synthetic, 0.7, 0.15, seed=3)
    assert all(r["source_type"] != "synthetic_mixed" for r in val + test)
    real_train_ids = {r["recording_id"] for r in train if r["source_type"] == "real"}
    syn_train = [r for r in train if r["source_type"] == "synthetic_mixed"]
    assert syn_train and {r["recording_id"] for r in syn_train} <= real_train_ids
    # Mixes of a val/test clip are dropped, not moved into train.
    assert len(syn_train) == 3 * len(real_train_ids)


def test_synthetic_mixes_of_fixed_split_train_clips_are_kept():
    team = {"label": "knocking", "recording_id": "indian_ambient:raj", "filepath": "raj-1.wav",
            "source_dataset": "indian_ambient", "source_type": "self_recorded", "split": "train"}
    mix = {"label": "knocking", "recording_id": "indian_ambient:raj", "filepath": "syn_raj-1.wav",
           "source_dataset": "synthetic_mixed", "source_type": "synthetic_mixed", "split": "train"}
    train, _, _ = split_rows([_real("knocking", f"fs:{i}") for i in range(10)] + [team, mix], 0.7, 0.15, seed=0)
    assert mix in train


def test_real_split_unchanged_by_adding_synthetic_rows():
    rows = [_real("knocking", f"fs:{i}") for i in range(50)]
    before = split_rows([dict(r) for r in rows], 0.7, 0.15, seed=42)
    synthetic = [
        {"label": "knocking", "recording_id": f"fs:{i}", "filepath": f"syn_{i}.wav",
         "source_dataset": "synthetic_mixed", "source_type": "synthetic_mixed", "split": "train"}
        for i in range(50)
    ]
    after = split_rows([dict(r) for r in rows] + synthetic, 0.7, 0.15, seed=42)
    for b, a in zip(before, after):
        assert _ids(b) == {r["recording_id"] for r in a if r["source_type"] == "real"}


def test_audioset_scanner_drops_clips_whose_labels_do_not_match(tmp_path: Path):
    root = tmp_path / "audioset"
    (root / "metadata").mkdir(parents=True)
    (root / "metadata" / "eval_segments.csv").write_text(
        "# YTID, start_seconds, end_seconds, positive_labels\n"
        'aaaaaaaaaaa, 30.000, 40.000, "/m/07rn7sz,/m/039jq"\n'  # Shatter + Glass -> glass_break
        'b_bbbbbbbbb, 0.000, 10.000, "/m/07rwj3x"\n'  # Whoop: not glass
    )
    (root / "glass_break").mkdir()
    for name in ("aaaaaaaaaaa_30_000_40_000.wav", "b_bbbbbbbbb_0_000_10_000.wav"):
        (root / "glass_break" / name).write_bytes(b"")
    rows = _scan_audioset(root)
    assert [(Path(r["filepath"]).name, r["label"], r["recording_id"]) for r in rows] == [
        ("aaaaaaaaaaa_30_000_40_000.wav", "glass_break", "yt:aaaaaaaaaaa")
    ]


def test_desed_scanner_drops_dog_and_alarm_clips(tmp_path: Path):
    real = tmp_path / "desed" / "real"
    (real / "metadata" / "train").mkdir(parents=True)
    (real / "metadata" / "train" / "weak.tsv").write_text(
        "filename\tevent_labels\n"
        "Ydog00000000_0.000_10.000.wav\tDog,Speech\n"
        "Yvac00000000_0.000_10.000.wav\tVacuum_cleaner\n"
    )
    for sub in ("weak", "unlabel_in_domain"):
        (real / "audio" / "train" / sub).mkdir(parents=True)
    for name in ("Ydog00000000_0.000_10.000.wav", "Yvac00000000_0.000_10.000.wav"):
        (real / "audio" / "train" / "weak" / name).write_bytes(b"")
    (real / "audio" / "train" / "unlabel_in_domain" / "Yunl00000000_0.000_10.000.wav").write_bytes(b"")
    rows = _scan_desed(tmp_path / "desed")
    assert [(Path(r["filepath"]).name, r["label"]) for r in rows] == [("Yvac00000000_0.000_10.000.wav", "background")]


def test_fsd50k_background_sample_skips_target_like_labels(tmp_path: Path, monkeypatch):
    from src.data import manifest_builder

    gt = tmp_path / "FSD50K" / "FSD50K.ground_truth"
    gt.mkdir(parents=True)
    (gt / "dev.csv").write_text(
        "fname,labels,mids,split\n"
        '1,"Speech,Human_voice",x,train\n'  # background
        '2,"Telephone,Alarm",x,train\n'  # phone/alarm: never background
        '3,"Doorbell,Alarm",x,train\n'  # target
        '4,"Dog,Animal",x,train\n'  # dog without Bark: excluded, not background
    )
    monkeypatch.setattr(manifest_builder, "METADATA_DIR", tmp_path / "no_cache" / "metadata")
    rows = manifest_builder._scan_fsd50k(tmp_path / "FSD50K")
    got = sorted((Path(r["filepath"]).stem, r["label"]) for r in rows)
    assert got == [("1", "background"), ("3", "doorbell")]
