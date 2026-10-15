import numpy as np

from src.personalization import prototypes as pt


def _hops(vectors, energy=None):
    vectors = np.asarray(vectors, np.float32)
    n = len(vectors)
    return {"embeddings": vectors, "class_scores": np.zeros((n, 8), np.float32),
            "energy": np.asarray(energy if energy is not None else np.ones(n), np.float32), "vad": np.ones(n, bool)}


def test_prototype_is_unit_and_matches_its_own_sound():
    rng = np.random.default_rng(0)
    sound = rng.normal(size=1024)
    samples = [_hops([sound + 0.05 * rng.normal(size=1024) for _ in range(3)]) for _ in range(5)]
    proto = pt.build_prototype(samples)
    assert np.isclose(np.linalg.norm(proto), 1.0, atol=1e-5)
    same, _ = pt.window_similarities(_hops([sound] * 3), proto)
    other, _ = pt.window_similarities(_hops([rng.normal(size=1024)] * 3), proto)
    assert same.min() > 0.95 and other.max() < 0.3


def test_enrolment_keeps_the_loudest_windows():
    quiet, loud = np.zeros(1024), np.ones(1024)
    vecs = pt.sample_vectors(_hops([quiet, quiet, loud, loud, quiet], energy=[0.1, 0.1, 5, 5, 0.1]), top_k=1)
    # Pooled window at the loudest hop mixes loud with its predecessor; it must lean towards "loud".
    assert vecs.shape == (1, 1024) and vecs[0].sum() > 0


def test_centering_removes_the_shared_component():
    rng = np.random.default_rng(1)
    common = 10 * np.ones(1024)
    a, b = common + rng.normal(size=1024), common + rng.normal(size=1024)
    try:
        pt.set_center(None)
        raw, _ = pt.window_similarities(_hops([b]), pt.build_prototype([_hops([a])]))
        pt.set_center(common)
        centred, _ = pt.window_similarities(_hops([b]), pt.build_prototype([_hops([a])]))
    finally:
        pt.set_center(None)
    assert raw[0] > 0.9 and abs(centred[0]) < 0.2
