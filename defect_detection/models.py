"""Learned baselines: a random forest on OpenCV features and a small Keras CNN on raw pixels."""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402

from defect_detection.config import (CNN_BATCH, CNN_EPOCHS, CNN_FILTERS, CNN_LEARNING_RATE,  # noqa: E402
                                     DECISION_THRESHOLD, IMAGE_SIZE, RF_TREES, SEED)

GRAY_MAX = 255.0
COLOR_CHANNELS = 3


def train_random_forest(features, labels):
    model = RandomForestClassifier(n_estimators=RF_TREES, random_state=SEED)
    return model.fit(features, labels)


def build_cnn():
    """Three conv-BN-ReLU-pool blocks -> global max pooling -> one logit.

    Global *max* pooling matters here: a defect covers a few pixels, and
    averaging the feature map over the whole image dilutes that response
    (an average-pooling version caught only 7% of defects).
    """
    layers = [tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, COLOR_CHANNELS)),
              tf.keras.layers.Rescaling(1.0 / GRAY_MAX)]
    for filters in CNN_FILTERS:
        layers.append(tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False))
        layers.append(tf.keras.layers.BatchNormalization())
        layers.append(tf.keras.layers.ReLU())
        layers.append(tf.keras.layers.MaxPooling2D())
    layers.append(tf.keras.layers.GlobalMaxPooling2D())
    layers.append(tf.keras.layers.Dense(1))
    model = tf.keras.Sequential(layers)
    model.compile(optimizer=tf.keras.optimizers.Adam(CNN_LEARNING_RATE),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.0)])
    return model


def train_cnn(images, labels):
    tf.keras.utils.set_random_seed(SEED)
    model = build_cnn()
    history = model.fit(images, labels, epochs=CNN_EPOCHS, batch_size=CNN_BATCH, validation_split=0.1, verbose=0)
    return model, history.history


def predict_cnn(model, images):
    probabilities = tf.sigmoid(model.predict(images, verbose=0)).numpy().ravel()
    return (probabilities > DECISION_THRESHOLD).astype(int)
