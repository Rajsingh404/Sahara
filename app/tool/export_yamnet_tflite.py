"""Export the frozen YAMNet backbone as a TFLite model that outputs 1024-d embeddings.

The classifier head exported by src/export/to_tflite.py takes a YAMNet embedding as input, so
the phone needs YAMNet too. This wraps the same TF Hub model used for training
(src/preprocessing/features.py) with a fixed 15,600-sample input (0.975 s at 16 kHz, one
YAMNet patch) and an `embeddings` output of shape [1, 1024].

Run from the repo root inside the ML venv (needs tensorflow + tensorflow_hub and network for
the first Hub download):

    python app/tool/export_yamnet_tflite.py
    python app/tool/sync_assets.py --yamnet models/exported/yamnet.tflite
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from src.config import MODELS_DIR  # noqa: E402

YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"
INPUT_SAMPLES = 15_600


def main() -> Path:
    yamnet = hub.load(YAMNET_HANDLE)

    class Embedder(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec([INPUT_SAMPLES], tf.float32, name="waveform")])
        def embed(self, waveform):
            _, embeddings, _ = yamnet(waveform)
            return {"embeddings": embeddings}

    module = Embedder()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([module.embed.get_concrete_function()], module)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite = converter.convert()

    target = MODELS_DIR / "exported" / "yamnet.tflite"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(tflite)

    # Sanity check against the Hub model on random audio.
    wave = np.random.default_rng(0).uniform(-0.5, 0.5, INPUT_SAMPLES).astype(np.float32)
    interpreter = tf.lite.Interpreter(model_content=tflite)
    interpreter.allocate_tensors()
    interpreter.set_tensor(interpreter.get_input_details()[0]["index"], wave)
    interpreter.invoke()
    out = next(d for d in interpreter.get_output_details() if d["shape"][-1] == 1024)
    lite = interpreter.get_tensor(out["index"])
    ref = yamnet(wave)[1].numpy()
    print(f"yamnet.tflite: {target} ({target.stat().st_size / 1024:.0f} KiB), embeddings {lite.shape}")
    print(f"max |tflite - hub| = {np.abs(lite - ref).max():.2e}")
    return target


if __name__ == "__main__":
    main()
