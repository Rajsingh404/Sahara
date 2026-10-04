"""Export the selected run's classifier head (1024-d window vector -> 8 sigmoid scores) as FP32 TFLite.

    python src/export/to_tflite.py [--run <run folder>]   # default: models/logs/selected_run.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import MODELS_DIR
from src.evaluation.streaming import load_head, selected_run_dir

EXPORT_DIR = MODELS_DIR / "exported"


def export_fp32(run: str | None = None) -> Path:
    run_dir = selected_run_dir(run)
    model = load_head(run_dir)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # TF 2.16's MLIR converter can abort on some macOS/Metal Keras graphs; the legacy
    # converter produces an equivalent portable FlatBuffer for this Dense head.
    converter.experimental_new_converter = False
    target = EXPORT_DIR / "sahara_fp32.tflite"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(converter.convert())
    (EXPORT_DIR / "SOURCE_RUN.txt").write_text(run_dir.name + "\n")
    print(f"FP32 head from {run_dir.name}: {target} ({target.stat().st_size / 1024:.1f} KiB)")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run")
    export_fp32(parser.parse_args().run)
