"""Cost-sensitive objectives for safety-critical SAHARA outputs."""
from __future__ import annotations

import tensorflow as tf

from src.config import POSITIVE_CLASS_WEIGHTS, SOUND_CLASSES


def weighted_binary_crossentropy(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """BCE with configurable extra cost for positive-class false negatives."""
    weights = tf.constant([POSITIVE_CLASS_WEIGHTS[name] for name in SOUND_CLASSES], dtype=tf.float32)
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    loss = -(weights * y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
    return tf.reduce_mean(loss, axis=-1)
