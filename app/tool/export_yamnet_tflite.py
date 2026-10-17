"""Export the frozen YAMNet backbone as a TFLite model with embeddings and class scores.

The classifier head exported by src/export/to_tflite.py takes a YAMNet embedding as input, so
the phone needs YAMNet too. This wraps the same TF Hub model used for training
(src/preprocessing/features.py) with a fixed 15,600-sample input (0.975 s at 16 kHz, one
YAMNet patch) and two outputs: `embeddings` [1, 1024] for the head, and `scores` [1, 521],
YAMNet's own AudioSet classes, for the open-set "other sounds" layer and the speech gate in
front of name calls. It also writes YAMNet's class map next to the model.

Run from the repo root inside the ML venv (needs tensorflow + tensorflow_hub and network for
the first Hub download):

    python app/tool/export_yamnet_tflite.py
    python app/tool/sync_assets.py --yamnet models/exported/yamnet.tflite
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from src.config import MODELS_DIR  # noqa: E402

# Same TF-Hub cache as training (src/preprocessing/features.py): the default lives in a temp folder
# that macOS cleans up, which leaves a broken half-deleted model behind.
os.environ.setdefault("TFHUB_CACHE_DIR", str(MODELS_DIR / "tfhub_cache"))

YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"
INPUT_SAMPLES = 15_600


def main() -> Path:
    yamnet = hub.load(YAMNET_HANDLE)

    class Embedder(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec([INPUT_SAMPLES], tf.float32, name="waveform")])
        def embed(self, waveform):
            scores, embeddings, _ = yamnet(waveform)
            return {"embeddings": embeddings, "scores": scores}

    module = Embedder()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([module.embed.get_concrete_function()], module)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite = converter.convert()

    target = MODELS_DIR / "exported" / "yamnet.tflite"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(tflite)
    class_map = target.parent / "yamnet_class_map.csv"
    class_map.write_bytes(tf.io.gfile.GFile(yamnet.class_map_path().numpy().decode(), "rb").read())

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
    print(f"yamnet_class_map.csv: {class_map}")
    check_against_pipeline(target)
    return target


def check_against_pipeline(target: Path, n_sources: int = 5) -> float:
    """Embed every phone-style patch of one real clip per source with yamnet.tflite and compare
    with src/preprocessing/features.extract_hop_embeddings (what training used)."""
    import csv

    from src.config import METADATA_DIR
    from src.preprocessing.audio_utils import load_audio
    from src.preprocessing.features import device_hops, extract_hop_embeddings

    interpreter = tf.lite.Interpreter(model_path=str(target))
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]["index"]
    out = next(d for d in interpreter.get_output_details() if d["shape"][-1] == 1024)["index"]
    picked: dict[str, str] = {}
    manifest = METADATA_DIR / "test_manifest.csv"
    if not manifest.exists():
        print("No test manifest; skipping the 5-clip pipeline check.")
        return float("nan")
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            picked.setdefault(row["source_dataset"], row["filepath"])
    worst = 0.0
    for source, path in list(picked.items())[:n_sources]:
        wave = load_audio(path, peak_normalize=False)
        if wave is None:
            continue
        ref = extract_hop_embeddings(wave)["embeddings"]
        starts, _ = device_hops(wave)
        padded = np.pad(wave, (max(0, 16_000 - len(wave)), 0))
        lite = []
        for s in starts:
            interpreter.set_tensor(inp, padded[s : s + INPUT_SAMPLES].astype(np.float32))
            interpreter.invoke()
            lite.append(interpreter.get_tensor(out)[0])
        diff = float(np.abs(np.asarray(lite) - ref).max())
        worst = max(worst, diff)
        print(f"  {source:15s} {Path(path).name[:40]:40s} {len(starts):3d} patches  max |tflite - pipeline| = {diff:.2e}")
    print(f"yamnet.tflite vs training pipeline, worst over {min(n_sources, len(picked))} clips: {worst:.2e}")
    return worst


if __name__ == "__main__":
    main()
