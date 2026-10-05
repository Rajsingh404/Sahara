"""Trainable classifier head on frozen YAMNet embeddings."""
from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers

from src.config import HEAD_INPUT_DIM, SOUND_CLASSES


def build_classifier_head(input_dim: int = HEAD_INPUT_DIM, num_classes: int | None = None) -> keras.Model:
    num_classes = num_classes or len(SOUND_CLASSES)
    inputs = keras.Input(shape=(input_dim,), name="embedding")
    x = layers.Dense(256, activation="relu")(inputs)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="sigmoid", name="predictions")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="sahara_classifier_head")


# The head always runs on the CPU. On Apple Silicon with tensorflow-metal 1.2.0 (TF 2.16,
# Keras 3.15), Model.predict() on the Metal GPU returns wrong scores (max |diff| 0.80 vs a NumPy
# forward pass, 99.9 % of rows; found 2026-10-05), while direct calls and CPU predict agree to
# < 1e-6. The head is tiny, so CPU costs nothing. tests/test_head_device.py guards this.
HEAD_DEVICE = "/CPU:0"


def head_predict(model: keras.Model, X, batch_size: int = 4096):
    """Scores for window vectors ``X`` [n, 1024] -> [n, 8], computed on the CPU."""
    import tensorflow as tf

    with tf.device(HEAD_DEVICE):
        return model.predict(X, batch_size=batch_size, verbose=0)
