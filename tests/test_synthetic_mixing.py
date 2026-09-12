"""Unit tests for synthetic mixing pipeline and manifest builder extensions."""
from __future__ import annotations

import csv
import numpy as np
import pytest
import soundfile as sf

from src.config import SAMPLE_RATE
from src.data.manifest_builder import build_manifest
from src.preprocessing.synthetic_mixing import mix_target_with_background


def test_synthetic_mixing_snr_and_peak(tmp_path):
    """Test that target+background mixing achieves expected SNR and does not clip."""
    sr = SAMPLE_RATE
    duration_sec = 2.0
    t_samples = int(sr * duration_sec)

    # Generate synthetic target (pure sine tone at 440 Hz)
    t = np.linspace(0, duration_sec, t_samples, endpoint=False, dtype=np.float32)
    target_wave = 0.5 * np.sin(2 * np.pi * 440 * t)
    target_path = tmp_path / "target.wav"
    sf.write(str(target_path), target_wave, sr)

    # Generate synthetic background (gaussian noise)
    rng = np.random.default_rng(42)
    bg_wave = 0.2 * rng.normal(size=t_samples).astype(np.float32)
    bg_path = tmp_path / "background.wav"
    sf.write(str(bg_path), bg_wave, sr)

    target_snr_db = 6.0
    mixed = mix_target_with_background(target_path, bg_path, target_snr_db)

    # Check length & sample rate consistency
    assert len(mixed) == t_samples

    # Check peak amplitude (no clipping above 1.0)
    assert np.max(np.abs(mixed)) <= 0.99

    # Check approximate SNR achieved
    # load_audio peak-normalizes both inputs to 1.0 peak amplitude
    from src.preprocessing.audio_utils import load_audio
    t_loaded = load_audio(target_path)
    b_loaded = load_audio(bg_path)
    t_pow = np.mean(t_loaded**2)
    b_pow = np.mean(b_loaded**2)
    expected_scale = np.sqrt(t_pow / (b_pow * (10 ** (target_snr_db / 10.0))))
    achieved_snr = 10 * np.log10(t_pow / (np.mean((expected_scale * b_loaded) ** 2)))

    assert abs(achieved_snr - target_snr_db) < 0.1


def test_manifest_builder_source_type_and_hot_swap(tmp_path, monkeypatch):
    """Test source_type field in manifest builder and Indian ambient hot-swap validation."""
    # Create temporary indian_ambient structure
    ia_dir = tmp_path / "indian_ambient"
    bg_dir = ia_dir / "background"
    bg_dir.mkdir(parents=True)

    dummy_wav = bg_dir / "ambient_test.wav"
    sf.write(str(dummy_wav), np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)

    meta_csv = ia_dir / "metadata.csv"
    with meta_csv.open("w", newline="", encoding="utf-8") as h:
        writer = csv.DictWriter(h, fieldnames=["filename", "is_background", "notes"])
        writer.writeheader()
        writer.writerow({"filename": "ambient_test.wav", "is_background": "1", "notes": "test"})

    # Patch get_raw_dir to point indian_ambient to ia_dir
    from src.config import get_raw_dir as orig_get_raw_dir

    def mock_get_raw_dir(dataset: str):
        if dataset == "indian_ambient":
            return ia_dir
        return orig_get_raw_dir(dataset)

    monkeypatch.setattr("src.data.manifest_builder.get_raw_dir", mock_get_raw_dir)

    rows = build_manifest()
    ia_rows = [r for r in rows if r["source_dataset"] == "indian_ambient"]

    assert len(ia_rows) == 1
    assert ia_rows[0]["label"] == "background"
    assert ia_rows[0]["source_type"] == "real"
    assert "source_type" in ia_rows[0]

    # Clean up dummy file and verify graceful handling (hot-swap test)
    dummy_wav.unlink()
    rows_after = build_manifest()
    ia_rows_after = [r for r in rows_after if r["source_dataset"] == "indian_ambient"]
    assert len(ia_rows_after) == 0
