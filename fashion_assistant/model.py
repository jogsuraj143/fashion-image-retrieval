"""Frozen ResNet50 feature extractor + trainable 256-D embedding head."""
import tensorflow as tf
from tensorflow.keras import Input, Model, layers
from tensorflow.keras.applications import ResNet50


def build_embedding_model(embedding_dim=256):
    backbone = ResNet50(weights='imagenet', include_top=False,
                        pooling=None, input_shape=(224, 224, 3))
    backbone.trainable = False
    inputs = Input(shape=(224, 224, 3), name='image')
    features = backbone(inputs, training=False)
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(features)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(embedding_dim)(x)
    outputs = layers.UnitNormalization(axis=-1)(x)
    return Model(inputs, outputs, name='fashion_embedding_model')


def triplet_loss(y_true, embeddings):
    a, p, n = tf.split(embeddings, 3, axis=1)
    positive_distance = tf.reduce_sum(tf.square(a - p), axis=1)
    negative_distance = tf.reduce_sum(tf.square(a - n), axis=1)
    return tf.reduce_mean(tf.maximum(0.0, positive_distance - negative_distance + 0.2))


def build_triplet_model(embedding_model, learning_rate=1e-4):
    a_input = Input(shape=(224, 224, 3), name='anchor')
    p_input = Input(shape=(224, 224, 3), name='positive')
    n_input = Input(shape=(224, 224, 3), name='negative')
    merged = layers.Concatenate(axis=1)([
        embedding_model(a_input), embedding_model(p_input), embedding_model(n_input)
    ])
    model = Model([a_input, p_input, n_input], merged, name='fashion_triplet_model')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss=triplet_loss)
    return model
