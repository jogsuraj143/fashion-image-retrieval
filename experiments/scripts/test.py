
"""Local training entry point. Run from the project root: python train.py"""
import argparse
from pathlib import Path
import tensorflow as tf
from fashion_assistant.config import Config

defaults = Config()
embedding_model = tf.keras.models.load_model(
    str(defaults.output_dir / "fashion_triplet_model_semi_hard_negatives.keras")
    , compile=False)

for layer in embedding_model.layers:
    print(layer.name)

inner_model=embedding_model.get_layer("fashion_embedding_model")
for layer in inner_model.layers:
    print(layer.name)


base_model=inner_model.get_layer("resnet50")
for layer in base_model.layers:
    print(layer.name)