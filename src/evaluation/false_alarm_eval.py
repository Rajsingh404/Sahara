"""Measure calibrated false alarms on background soundscapes."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import FALSE_ALARM_RATE_LIMIT, METADATA_DIR, SOUND_CLASSES
from src.preprocessing.audio_utils import load_audio
from src.preprocessing.features import extract_yamnet_embedding
from src.training.losses import weighted_binary_crossentropy


def run() -> dict:
    with (METADATA_DIR / "class_thresholds.json").open() as handle:
        thresholds = {key: value["threshold"] for key, value in json.load(handle).items()}
    with (METADATA_DIR / "full_manifest.csv").open(newline="", encoding="utf-8") as handle:
        background = [row for row in csv.DictReader(handle) if row["label"] == "background"]
    checkpoint = Path("models/checkpoints/best_model.keras")
    model = tf.keras.models.load_model(checkpoint, custom_objects={"weighted_binary_crossentropy": weighted_binary_crossentropy})
    triggered = np.zeros(len(SOUND_CLASSES), dtype=int)
    evaluated = 0
    for row in background:
        waveform = load_audio(row["filepath"])
        if waveform is None:
            continue
        prediction = model.predict(extract_yamnet_embedding(waveform)[None, :], verbose=0)[0]
        triggered += prediction >= np.asarray([thresholds[name] for name in SOUND_CLASSES])
        evaluated += 1
    rates = {name: float(triggered[i] / evaluated) if evaluated else 0.0 for i, name in enumerate(SOUND_CLASSES)}
    overall = float(np.any(triggered > 0))  # replaced below with clip-level aggregate in a future streaming run
    print(f"Background clips evaluated: {evaluated}")
    print("class\tfalse_alarm_rate\tstatus")
    for name in SOUND_CLASSES:
        print(f"{name}\t{rates[name]:.3f}\t{'TUNE' if rates[name] > FALSE_ALARM_RATE_LIMIT else 'OK'}")
    report = Path("docs/evaluation_report.md")
    with report.open("a", encoding="utf-8") as handle:
        handle.write("\n## False Alarm Analysis\n\n")
        handle.write(f"Background clips evaluated: {evaluated}. Per-class false-alarm rates use calibrated thresholds; rates above {FALSE_ALARM_RATE_LIMIT:.0%} require tuning.\n\n")
        handle.write("| Class | False-alarm rate | Status |\n|---|---:|---|\n")
        for name in SOUND_CLASSES:
            handle.write(f"| {name} | {rates[name]:.3f} | {'TUNE' if rates[name] > FALSE_ALARM_RATE_LIMIT else 'OK'} |\n")
    return rates


if __name__ == "__main__":
    run()
