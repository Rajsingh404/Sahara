"""Check the exported heads against the Keras model on the test windows.

Pass criteria (build plan, Phase 5): FP32 TFLite vs Keras max |score diff| < 1e-4, and INT8
macro mAP no more than 0.02 below Keras. Writes models/logs/<run>/export_validation.json and
exits non-zero if a criterion fails.

    python src/export/validate_tflite.py [--run <run folder>]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import PROCESSED_DATA_DIR
from src.evaluation.metrics import compute_metrics
from src.evaluation.streaming import load_head, selected_run_dir
from src.export.to_tflite import EXPORT_DIR

FP32_TOLERANCE = 1e-4
INT8_MAX_MAP_DROP = 0.02


def predict_tflite(path: Path, values: np.ndarray) -> np.ndarray:
    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    predicted = []
    for value in values:
        batch = value[None, :]
        if inp["dtype"] == np.int8:
            scale, zero = inp["quantization"]
            batch = np.round(batch / scale + zero).clip(-128, 127).astype(np.int8)
        interpreter.set_tensor(inp["index"], batch.astype(inp["dtype"]))
        interpreter.invoke()
        result = interpreter.get_tensor(out["index"])
        if out["dtype"] == np.int8:
            scale, zero = out["quantization"]
            result = (result.astype(np.float32) - zero) * scale
        predicted.append(result[0])
    return np.asarray(predicted, dtype=np.float32)


def run(run: str | None = None) -> dict:
    run_dir = selected_run_dir(run)
    x = np.load(PROCESSED_DATA_DIR / "test_embeddings.npy")
    y = np.load(PROCESSED_DATA_DIR / "test_labels.npy")
    keras_scores = load_head(run_dir).predict(x, batch_size=1024, verbose=0)
    fp32 = predict_tflite(EXPORT_DIR / "sahara_fp32.tflite", x)
    int8 = predict_tflite(EXPORT_DIR / "sahara_int8.tflite", x)
    m_keras, m_int8 = compute_metrics(y, keras_scores)["macro"]["mAP"], compute_metrics(y, int8)["macro"]["mAP"]
    result = {
        "run": run_dir.name,
        "test_windows": int(len(x)),
        "fp32_max_abs_diff": float(np.abs(fp32 - keras_scores).max()),
        "fp32_pass": bool(np.abs(fp32 - keras_scores).max() < FP32_TOLERANCE),
        "keras_mAP": round(m_keras, 4),
        "int8_mAP": round(m_int8, 4),
        "int8_mAP_drop": round(m_keras - m_int8, 4),
        "int8_max_abs_diff": float(np.abs(int8 - keras_scores).max()),
        "int8_pass": bool(m_keras - m_int8 <= INT8_MAX_MAP_DROP),
        "sizes_kib": {p.name: round(p.stat().st_size / 1024, 1) for p in sorted(EXPORT_DIR.glob("*.tflite"))},
    }
    (run_dir / "export_validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run")
    r = run(parser.parse_args().run)
    sys.exit(0 if r["fp32_pass"] and r["int8_pass"] else 1)
