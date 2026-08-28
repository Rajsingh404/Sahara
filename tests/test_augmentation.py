import numpy as np

from src.config import SAMPLE_RATE, YAMNET_EMBEDDING_DIM
from src.preprocessing.augmentation import augment_embedding_batch, augment_waveform


def test_augmentation_handles_silence():
    result = augment_waveform(np.zeros(SAMPLE_RATE, dtype=np.float32))
    assert result.dtype == np.float32
    assert result.ndim == 1
    assert result.size > 0


def test_embedding_jitter_preserves_shape_and_is_reproducible():
    embeddings = np.zeros((2, YAMNET_EMBEDDING_DIM), dtype=np.float32)
    first = augment_embedding_batch(embeddings, rng=np.random.default_rng(7))
    second = augment_embedding_batch(embeddings, rng=np.random.default_rng(7))
    assert first.shape == embeddings.shape
    assert np.array_equal(first, second)
    assert not np.array_equal(first, embeddings)
