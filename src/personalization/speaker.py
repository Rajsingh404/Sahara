"""Caller voice vectors: a pretrained speaker encoder (GE2E, from Resemblyzer).

Resemblyzer's encoder (3-layer LSTM, 256 units, 40 mel bands, trained on thousands of speakers
for speaker verification, Apache-2.0) maps an utterance to a 256-d unit vector; the same voice
lands close by cosine similarity. The phone runs the same maths in pure Kotlin
(app/android/.../core/SpeakerEncoder.kt) from the weights this module exports, so there is no
extra runtime on the phone.

The phone path is simplified on purpose: the speech around the name (up to 1.6 s, found with
`voice_bounds`) is padded to one 1.6 s partial and encoded once. `embed` here follows that path, not Resemblyzer's multi-partial averaging.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
N_FFT = 400  # 25 ms
HOP = 160  # 10 ms
N_MELS = 40
PARTIAL_FRAMES = 160  # 1.6 s
TARGET_DBFS = -30.0
MAGIC = b"SPK1"


def voice_bounds(wave: np.ndarray) -> tuple[int, int]:
    """Wider than the name spotter's endpointing (30 dB below peak): more voice, better identity."""
    from src.personalization.name_head import speech_bounds

    return speech_bounds(wave, db_below_peak=30.0)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class SpeakerEncoder:
    def __init__(self, weights: dict[str, np.ndarray] | None = None):
        if weights is None:
            weights = self._load_resemblyzer()
        self.w = weights

    @staticmethod
    def _load_resemblyzer() -> dict[str, np.ndarray]:
        import librosa
        import torch
        from importlib import resources

        pt = Path(str(resources.files("resemblyzer") / "pretrained.pt"))
        state = torch.load(pt, map_location="cpu")["model_state"]
        w = {k: v.numpy().astype(np.float32) for k, v in state.items() if k.startswith(("lstm.", "linear."))}
        w["mel_fb"] = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS).astype(np.float32)
        return w

    # -- front end (mirrors SpeakerEncoder.kt) --------------------------------------------------

    @staticmethod
    def normalise_volume(wave: np.ndarray) -> np.ndarray:
        rms = np.sqrt(np.mean(wave.astype(np.float64) ** 2)) + 1e-10
        gain = 10 ** ((TARGET_DBFS - 20 * np.log10(rms)) / 20)
        return (wave * gain).astype(np.float32) if gain > 1 else wave.astype(np.float32)

    def mel(self, wave: np.ndarray) -> np.ndarray:
        """[frames, 40] power mel spectrogram, librosa-style centred STFT with zero padding."""
        padded = np.pad(wave, (N_FFT // 2, N_FFT // 2))
        n = 1 + (len(padded) - N_FFT) // HOP
        window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)).astype(np.float32)  # periodic Hann
        idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n)[:, None]
        power = np.abs(np.fft.rfft(padded[idx] * window, N_FFT)) ** 2
        return (power @ self.w["mel_fb"].T).astype(np.float32)

    def embed(self, wave: np.ndarray) -> np.ndarray:
        """256-d unit vector for the voice in [wave] (16 kHz mono float)."""
        a, b = voice_bounds(wave)
        wave = self.normalise_volume(wave[a:b])
        need = (PARTIAL_FRAMES - 1) * HOP
        if len(wave) < need:
            wave = np.pad(wave, (0, need - len(wave)))
        mel = self.mel(wave)
        # Centre one 160-frame partial on the utterance.
        if len(mel) > PARTIAL_FRAMES:
            start = (len(mel) - PARTIAL_FRAMES) // 2
            mel = mel[start : start + PARTIAL_FRAMES]
        return self.forward(mel)

    def forward(self, mel: np.ndarray) -> np.ndarray:
        x = mel
        for layer in range(3):
            w_ih, w_hh = self.w[f"lstm.weight_ih_l{layer}"], self.w[f"lstm.weight_hh_l{layer}"]
            b = self.w[f"lstm.bias_ih_l{layer}"] + self.w[f"lstm.bias_hh_l{layer}"]
            hdim = w_hh.shape[1]
            h = np.zeros(hdim, np.float32)
            c = np.zeros(hdim, np.float32)
            pre = x @ w_ih.T + b
            out = np.empty((len(x), hdim), np.float32)
            for t in range(len(x)):
                g = pre[t] + w_hh @ h
                i, f, gg, o = _sigmoid(g[:hdim]), _sigmoid(g[hdim : 2 * hdim]), np.tanh(g[2 * hdim : 3 * hdim]), _sigmoid(g[3 * hdim :])
                c = f * c + i * gg
                h = o * np.tanh(c)
                out[t] = h
            x = out
        e = np.maximum(self.w["linear.weight"] @ x[-1] + self.w["linear.bias"], 0)
        return e / (np.linalg.norm(e) + 1e-9)

    # -- export for the phone ------------------------------------------------------------------

    ORDER = (
        "mel_fb",
        *(f"lstm.{k}_l{layer}" for layer in range(3) for k in ("weight_ih", "weight_hh", "bias_ih", "bias_hh")),
        "linear.weight",
        "linear.bias",
    )

    def export(self, path: Path) -> None:
        """Little-endian: magic, tensor count, then per tensor (rank, dims..., float32 data)."""
        with Path(path).open("wb") as f:
            f.write(MAGIC)
            f.write(struct.pack("<i", len(self.ORDER)))
            for key in self.ORDER:
                arr = np.ascontiguousarray(self.w[key], dtype="<f4")
                f.write(struct.pack("<i", arr.ndim))
                f.write(struct.pack(f"<{arr.ndim}i", *arr.shape))
                f.write(arr.tobytes())

    @classmethod
    def load(cls, path: Path) -> "SpeakerEncoder":
        data = Path(path).read_bytes()
        assert data[:4] == MAGIC, "not a speaker_encoder.bin"
        off = 4
        (count,) = struct.unpack_from("<i", data, off)
        off += 4
        w = {}
        for key in cls.ORDER[:count]:
            (ndim,) = struct.unpack_from("<i", data, off)
            off += 4
            shape = struct.unpack_from(f"<{ndim}i", data, off)
            off += 4 * ndim
            n = int(np.prod(shape))
            w[key] = np.frombuffer(data, dtype="<f4", count=n, offset=off).reshape(shape).copy()
            off += 4 * n
        return cls(w)
