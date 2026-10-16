"""Pretrained speech embeddings for name spotting (Google speech_embedding, via openWakeWord).

Two small models from the `openwakeword==0.5.1` wheel, the same files the phone runs:

- melspectrogram: 16 kHz int16-scaled audio -> 32-bin log-mel frames every 10 ms
- embedding_model: 76 mel frames (0.775 s) -> one 96-d embedding, computed every 8 frames (80 ms)

The embedding model was pretrained on a large multi-speaker corpus for keyword spotting, so a
small head on top generalises to voices it never heard. Model weights are CC BY-NC-SA 4.0
(openWakeWord), which suits an academic project; see docs/name_call.md before any commercial use.
"""
from __future__ import annotations

from importlib import resources
from pathlib import Path

import numpy as np

MEL_FRAMES_PER_EMBEDDING = 76
EMBEDDING_STEP = 8
EMBEDDING_DIM = 96
MIN_SAMPLES = 16_000  # the mel model needs at least ~1 s


def default_models_dir() -> Path:
    try:
        return Path(str(resources.files("openwakeword") / "resources" / "models"))
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on the environment
        raise SystemExit("pip install openwakeword==0.5.1 onnxruntime (or pass --models-dir)") from exc


class SpeechEmbedder:
    def __init__(self, models_dir: Path | None = None):
        import onnxruntime as ort

        models_dir = Path(models_dir) if models_dir else default_models_dir()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        self._mel = ort.InferenceSession(str(models_dir / "melspectrogram.onnx"), opts)
        self._emb = ort.InferenceSession(str(models_dir / "embedding_model.onnx"), opts)
        self.models_dir = models_dir

    def mel(self, wave: np.ndarray) -> np.ndarray:
        x = (np.clip(wave, -1.0, 1.0) * 32767.0).astype(np.float32)[None]
        if x.shape[1] < MIN_SAMPLES:
            x = np.pad(x, ((0, 0), (0, MIN_SAMPLES - x.shape[1])))
        spec = np.squeeze(self._mel.run(None, {self._mel.get_inputs()[0].name: x})[0])
        return spec / 10.0 + 2.0  # openWakeWord's transform, matches Google's TF Hub front end

    def embed(self, wave: np.ndarray) -> np.ndarray:
        """[n, 96] embeddings, one per 80 ms of audio."""
        spec = self.mel(wave)
        windows = [spec[i : i + MEL_FRAMES_PER_EMBEDDING] for i in range(0, len(spec) - MEL_FRAMES_PER_EMBEDDING + 1, EMBEDDING_STEP)]
        batch = np.asarray(windows, dtype=np.float32)[..., None]
        out = self._emb.run(None, {self._emb.get_inputs()[0].name: batch})[0]
        return np.asarray(out, dtype=np.float32).reshape(len(windows), EMBEDDING_DIM)
