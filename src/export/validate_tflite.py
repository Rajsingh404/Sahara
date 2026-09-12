"""Compare the exported INT8 embedding head with the Keras checkpoint."""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.evaluation.metrics import compute_metrics
from src.export.to_tflite import checkpoint_path
from src.training.losses import weighted_binary_crossentropy


def _predict_tflite(path, values: np.ndarray) -> np.ndarray:
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
    return np.asarray(predicted)


def run() -> dict:
    x = np.load(PROCESSED_DATA_DIR / "test_embeddings.npy")
    y = np.load(PROCESSED_DATA_DIR / "test_labels.npy")
    model = tf.keras.models.load_model(checkpoint_path(), custom_objects={"weighted_binary_crossentropy": weighted_binary_crossentropy})
    keras = compute_metrics(y, model.predict(x, verbose=0))
    tflite = compute_metrics(y, _predict_tflite(MODELS_DIR / "exported" / "sahara_int8.tflite", x))
    print("metric\tKeras\tINT8 TFLite")
    for metric in ("precision", "recall", "f1", "mAP"):
        print(f"{metric}\t{keras['macro'][metric]:.3f}\t{tflite['macro'][metric]:.3f}")
    for name in keras["per_class"]:
        base, quantized = keras["per_class"][name]["f1"], tflite["per_class"][name]["f1"]
        if base > 0 and (base - quantized) / base > 0.05:
            print(f"WARNING meaningful INT8 F1 drop: {name} ({base:.3f} -> {quantized:.3f})")
    return {"keras": keras, "int8": tflite}


if __name__ == "__main__":
    run()
