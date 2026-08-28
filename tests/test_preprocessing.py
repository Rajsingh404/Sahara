import numpy as np
import soundfile as sf

from src.config import SAMPLE_RATE
from src.preprocessing.audio_utils import load_audio


def test_load_audio_resamples_mono_and_peak_normalizes(tmp_path):
    source_sr = 8_000
    samples = np.column_stack([
        np.linspace(-0.25, 0.5, source_sr, dtype=np.float32),
        np.linspace(0.25, -0.5, source_sr, dtype=np.float32),
    ])
    path = tmp_path / "stereo.wav"
    sf.write(path, samples, source_sr)
    waveform = load_audio(path)
    assert waveform is not None
    assert waveform.ndim == 1
    assert len(waveform) == SAMPLE_RATE
    assert np.isclose(np.max(np.abs(waveform)), 1.0, atol=1e-5)


def test_load_audio_returns_none_for_missing_file(tmp_path):
    assert load_audio(tmp_path / "missing.wav") is None
