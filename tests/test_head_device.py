"""head_predict must equal a NumPy forward pass of the head's weights.

Guards the tensorflow-metal bug found on 2026-10-05: Model.predict() on the Metal GPU returned
wrong scores for 99.9 % of rows, while CPU predict and direct calls were correct.
"""
import numpy as np

from src.models.classifier_head import build_classifier_head, head_predict


def _numpy_forward(model, x):
    dense = [layer for layer in model.layers if layer.__class__.__name__ == "Dense"]
    for layer in dense:
        x = x @ layer.kernel.numpy() + layer.bias.numpy()
        x = np.maximum(x, 0) if layer.activation.__name__ == "relu" else 1 / (1 + np.exp(-x))
    return x


def test_head_predict_matches_numpy():
    model = build_classifier_head()
    x = np.abs(np.random.default_rng(0).normal(0, 1, (512, 1024))).astype(np.float32)
    np.testing.assert_allclose(head_predict(model, x), _numpy_forward(model, x.astype(np.float64)), atol=1e-5)
