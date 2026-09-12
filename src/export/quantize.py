"""Post-training INT8 export for the SAHARA embedding-classifier head."""
from __future__ import annotations

import time
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.export.to_tflite import checkpoint_path, export_fp32
from src.training.losses import weighted_binary_crossentropy


def _representative_dataset():
    embeddings = np.load(PROCESSED_DATA_DIR / "train_embeddings.npy")[:200].astype(np.float32)
    for embedding in embeddings:
        yield [embedding[None, :]]


def _latency(model_path: Path, samples: np.ndarray) -> float:
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details, output_details = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    elapsed = []
    for sample in samples[:100]:
        value = sample[None, :]
        if input_details["dtype"] == np.int8:
            scale, zero = input_details["quantization"]
            value = np.round(value / scale + zero).clip(-128, 127).astype(np.int8)
        else:
            value = value.astype(np.float32)
        start = time.perf_counter()
        interpreter.set_tensor(input_details["index"], value)
        interpreter.invoke()
        interpreter.get_tensor(output_details["index"])
        elapsed.append(time.perf_counter() - start)
    return float(np.mean(elapsed) * 1000)


def export_int8() -> Path:
    fp32 = export_fp32()
    model = tf.keras.models.load_model(checkpoint_path(), custom_objects={"weighted_binary_crossentropy": weighted_binary_crossentropy})
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.experimental_new_converter = False
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = _representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    target = MODELS_DIR / "exported" / "sahara_int8.tflite"
    target.write_bytes(converter.convert())
    samples = np.load(PROCESSED_DATA_DIR / "test_embeddings.npy")
    print(f"INT8 classifier TFLite: {target} ({target.stat().st_size / 1024:.1f} KiB)")
    print(f"Size reduction: {100 * (1 - target.stat().st_size / fp32.stat().st_size):.1f}%")
    print(f"Latency (development Mac, embedding head only): FP32={_latency(fp32, samples):.3f}ms INT8={_latency(target, samples):.3f}ms")
    return target


if __name__ == "__main__":
    export_int8()
