from src.data.manifest_builder import _stratified_split


def test_grouped_split_never_leaks_a_recording_id():
    rows = [
        {"label": "doorbell", "recording_id": f"session-{group}", "filepath": f"{group}-{clip}.wav"}
        for group in range(12) for clip in range(2)
    ]
    train, val, test = _stratified_split(rows, train_frac=0.7, val_frac=0.15, seed=42)
    ids = [{row["recording_id"] for row in split} for split in (train, val, test)]
    assert ids[0].isdisjoint(ids[1])
    assert ids[0].isdisjoint(ids[2])
    assert ids[1].isdisjoint(ids[2])
    assert len(train) + len(val) + len(test) == len(rows)
