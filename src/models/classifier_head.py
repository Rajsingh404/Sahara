"""Trainable classifier head on frozen YAMNet embeddings."""
from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.config import SOUND_CLASSES, YAMNET_EMBEDDING_DIM


def build_classifier_head(input_dim: int = YAMNET_EMBEDDING_DIM, num_classes: int | None = None) -> keras.Model:
    num_classes = num_classes or len(SOUND_CLASSES)
    inputs = keras.Input(shape=(input_dim,), name="embedding")
    x = layers.Dense(256, activation="relu")(inputs)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="sigmoid", name="predictions")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="sahara_classifier_head")
