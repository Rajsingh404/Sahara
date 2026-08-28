import numpy as np

from src.config import SAMPLE_RATE, YAMNET_EMBEDDING_DIM
from src.preprocessing import features


def test_logmel_shape_has_64_bands():
    waveform = np.zeros(SAMPLE_RATE, dtype=np.float32)
    logmel = features.extract_logmel(waveform)
    assert logmel.dtype == np.float32
    assert logmel.shape[0] == 64
    assert logmel.shape[1] > 0


def test_yamnet_embedding_is_mean_pooled(monkeypatch):
    class FakeTensor:
        def __init__(self, array): self.array = array
        def numpy(self): return self.array

    def fake_model(_waveform):
        embeddings = np.ones((3, YAMNET_EMBEDDING_DIM), dtype=np.float32)
        embeddings[1] *= 2
        return FakeTensor(np.empty((3, 1))), FakeTensor(embeddings), None

    monkeypatch.setattr(features, "_get_yamnet", lambda: fake_model)
    embedding = features.extract_yamnet_embedding(np.zeros(SAMPLE_RATE, dtype=np.float32))
    assert embedding.shape == (YAMNET_EMBEDDING_DIM,)
    assert np.allclose(embedding, 4 / 3)
