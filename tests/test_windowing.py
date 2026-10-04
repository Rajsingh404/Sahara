"""Window vectors must equal what the phone computes (AudioCaptureService.kt + InferenceEngine.kt)."""
from collections import deque

import numpy as np
import pytest
import soundfile as sf

from src.config import (
    EMBEDDING_POOL,
    HOP_SAMPLES,
    SAMPLE_RATE,
    VAD_ENERGY_THRESHOLD,
    WINDOW_SAMPLES,
    YAMNET_PATCH_SAMPLES,
)
from src.preprocessing.windowing import clip_examples, device_window_vectors, select_event_windows


def _fake_hops(vad, n_classes=8, dim=4):
    n = len(vad)
    emb = np.arange(n * dim, dtype=np.float32).reshape(n, dim)
    return {
        "embeddings": emb,
        "class_scores": np.zeros((n, n_classes), np.float32),
        "energy": np.where(vad, 1.0, 0.0).astype(np.float32),
        "vad": np.asarray(vad),
    }


def test_pool_skips_silent_hops_like_the_phone():
    # Hops 0,1 loud, 2,3 silent, 4 loud: the phone's pool after hop 4 holds hops 1 and 4.
    hops = _fake_hops([True, True, False, False, True])
    win = device_window_vectors(hops)
    assert win["hop_index"].tolist() == [0, 1, 4]
    np.testing.assert_allclose(win["vectors"][0], hops["embeddings"][0])  # pool of one after a reset
    np.testing.assert_allclose(win["vectors"][2], hops["embeddings"][[1, 4]].mean(axis=0))


def test_event_selection_keeps_top_windows_and_drops_the_rest():
    scores = np.array([0.0, 0.01, 0.9, 0.8, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0])
    keep = select_event_windows(scores, energy=np.ones(10))
    assert keep.tolist() == [2, 3]  # top-3 by score, minus the one below 25 % of the best


def test_event_selection_falls_back_to_energy_when_yamnet_is_unsure():
    keep = select_event_windows(np.full(4, 0.01), energy=np.array([0.1, 0.9, 0.2, 0.1]))
    assert keep.tolist() == [1]


def test_background_clip_uses_every_window_with_zero_target():
    x, y, idx = clip_examples(_fake_hops([True] * 6), labels=set())
    assert len(x) == 6 and not y.any()


# --------------------------------------------------------------------------------------
# Full parity with real YAMNet (needs the TF-Hub model; skipped if it cannot be loaded).


def _device_reference(pcm16: np.ndarray, yamnet) -> tuple[list[int], np.ndarray]:
    """Line-by-line port of the phone's capture loop, one YAMNet call per patch."""
    ring = np.zeros(WINDOW_SAMPLES, np.float32)
    filled, pool, out, hops_done = 0, deque(maxlen=EMBEDDING_POOL), [], []
    for h in range(len(pcm16) // HOP_SAMPLES):
        chunk = pcm16[h * HOP_SAMPLES : (h + 1) * HOP_SAMPLES].astype(np.float32) / 32768.0  # AudioWindow.push
        ring = np.concatenate([ring[HOP_SAMPLES:], chunk])
        filled = min(filled + HOP_SAMPLES, WINDOW_SAMPLES)
        if filled < WINDOW_SAMPLES:
            continue
        energy = float(np.mean(ring[-HOP_SAMPLES:].astype(np.float64) ** 2))  # AudioWindow.meanSquare
        if energy < VAD_ENERGY_THRESHOLD:
            continue
        frames = yamnet(ring[-YAMNET_PATCH_SAMPLES:])[1].numpy()  # InferenceEngine.embed (tail)
        pool.append(frames.mean(axis=0))  # mean over frames (1 frame), then EmbeddingPool
        out.append(np.mean(pool, axis=0))
        hops_done.append(h - 1)  # index of the full-buffer hop sequence (first full buffer = 0)
    return hops_done, np.asarray(out, np.float32)


@pytest.fixture(scope="module")
def yamnet():
    try:
        from src.preprocessing.features import _get_yamnet

        return _get_yamnet()
    except Exception as exc:  # no network / hub cache
        pytest.skip(f"YAMNet unavailable: {exc!r}")


def test_python_window_vectors_match_device_loop(tmp_path, yamnet):
    from src.preprocessing.audio_utils import load_audio
    from src.preprocessing.features import extract_hop_embeddings

    rng = np.random.default_rng(7)
    t = np.arange(int(4.5 * SAMPLE_RATE)) / SAMPLE_RATE
    wave = 0.3 * np.sin(2 * np.pi * 1000 * t) * (np.sin(2 * np.pi * 2 * t) > 0) + 0.02 * rng.standard_normal(len(t))
    wave[SAMPLE_RATE * 2 : SAMPLE_RATE * 3] = 0.0  # one silent second exercises VAD and the pool
    pcm16 = np.clip(np.round(wave * 32767), -32768, 32767).astype(np.int16)
    path = tmp_path / "parity.wav"
    sf.write(path, pcm16, SAMPLE_RATE, subtype="PCM_16")

    ref_hops, ref_vectors = _device_reference(pcm16, yamnet)

    audio = load_audio(path, peak_normalize=False)
    np.testing.assert_array_equal(audio, pcm16.astype(np.float32) / 32768.0)  # same samples as the phone
    win = device_window_vectors(extract_hop_embeddings(audio))

    assert win["hop_index"].tolist() == ref_hops
    assert len(ref_hops) < len(pcm16) // HOP_SAMPLES - 1  # VAD really skipped something
    assert np.abs(win["vectors"] - ref_vectors).max() < 1e-4
