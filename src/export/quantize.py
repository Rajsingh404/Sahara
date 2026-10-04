"""Post-training INT8 export of the selected run's head.

The representative dataset is a seeded random sample of 1,000 training window vectors (the
same vectors the phone feeds the head), covering positives and background.

    python src/export/quantize.py [--run <run folder>]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import PROCESSED_DATA_DIR, RANDOM_SEED
from src.evaluation.streaming import load_head, selected_run_dir
from src.export.to_tflite import EXPORT_DIR, export_fp32

REPRESENTATIVE_SAMPLES = 1000


def representative_windows(n: int = REPRESENTATIVE_SAMPLES) -> np.ndarray:
    X = np.load(PROCESSED_DATA_DIR / "train_embeddings.npy", mmap_mode="r")
    idx = np.sort(np.random.default_rng(RANDOM_SEED).choice(len(X), size=min(n, len(X)), replace=False))
    return np.asarray(X[idx], dtype=np.float32)


def latency_ms(model_path: Path, samples: np.ndarray) -> float:
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    elapsed = []
    for sample in samples[:200]:
        value = sample[None, :]
        if inp["dtype"] == np.int8:
            scale, zero = inp["quantization"]
            value = np.round(value / scale + zero).clip(-128, 127).astype(np.int8)
        start = time.perf_counter()
        interpreter.set_tensor(inp["index"], value.astype(inp["dtype"]))
        interpreter.invoke()
        interpreter.get_tensor(out["index"])
        elapsed.append(time.perf_counter() - start)
    return float(np.mean(elapsed) * 1000)


def export_int8(run: str | None = None) -> Path:
    fp32 = export_fp32(run)
    model = load_head(selected_run_dir(run))
    samples = representative_windows()
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.experimental_new_converter = False
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: ([s[None, :]] for s in samples)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    target = EXPORT_DIR / "sahara_int8.tflite"
    target.write_bytes(converter.convert())
    print(f"INT8 head: {target} ({target.stat().st_size / 1024:.1f} KiB, "
          f"{100 * (1 - target.stat().st_size / fp32.stat().st_size):.0f}% smaller than FP32)")
    print(f"Head latency on this Mac: FP32 {latency_ms(fp32, samples):.3f} ms, INT8 {latency_ms(target, samples):.3f} ms")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run")
    export_int8(parser.parse_args().run)
