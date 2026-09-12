"""Export the embedding-classifier head as an FP32 TensorFlow Lite model."""
from __future__ import annotations

import sys
from pathlib import Path
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import MODELS_DIR
from src.training.losses import weighted_binary_crossentropy


def checkpoint_path() -> Path:
    new = MODELS_DIR / "checkpoints" / "best_model.keras"
    return new if new.exists() else MODELS_DIR / "checkpoints" / "best_classifier.keras"


def export_fp32() -> Path:
    model = tf.keras.models.load_model(checkpoint_path(), custom_objects={"weighted_binary_crossentropy": weighted_binary_crossentropy})
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # TF 2.16's MLIR converter can abort on some macOS/Metal Keras graphs.
    # The legacy converter produces an equivalent portable FlatBuffer here.
    converter.experimental_new_converter = False
    target = MODELS_DIR / "exported" / "sahara_fp32.tflite"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(converter.convert())
    print(f"FP32 classifier TFLite: {target} ({target.stat().st_size / 1024:.1f} KiB)")
    return target


if __name__ == "__main__":
    export_fp32()
