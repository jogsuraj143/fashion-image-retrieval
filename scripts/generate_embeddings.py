"""Generate aligned embeddings for a DeepFashion partition from a chosen checkpoint."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from fashion_assistant.config import Config

def main():
    config = Config()
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True, help='Single-image .keras model or 3-input triplet checkpoint')
    p.add_argument('--partition', choices=['train', 'gallery', 'query'], default='gallery')
    p.add_argument('--data-dir', type=Path, default=config.dataset_root)
    p.add_argument('--out-dir', type=Path, default=config.output_dir)
    p.add_argument('--batch-size', type=int, default=16)
    args = p.parse_args()
    partition_file = args.data_dir / 'Eval' / 'list_eval_partition.txt'
    if not partition_file.exists(): partition_file = args.data_dir / 'eval' / 'list_eval_partition.txt'
    df = pd.read_csv(partition_file, sep=r'\s+', skiprows=1)
    paths = df.loc[df.evaluation_status == args.partition, 'image_name'].astype(str).to_numpy()
    model = tf.keras.models.load_model(args.model, compile=False)
    if len(model.inputs) == 3: model = model.get_layer('fashion_embedding_model')
    root = tf.constant(str(args.data_dir.resolve()) + '/')
    def load(path):
        image = tf.io.decode_jpeg(tf.io.read_file(tf.strings.join([root, path])), channels=3)
        image = tf.image.resize(image, (224, 224))
        return tf.keras.applications.resnet50.preprocess_input(image)
    ds = tf.data.Dataset.from_tensor_slices(paths).map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    embeddings = model.predict(ds, verbose=1)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / f'{args.partition}_embeddings.npy', embeddings)
    np.save(args.out_dir / f'{args.partition}_paths.npy', paths)
    print('Saved', embeddings.shape, 'embeddings for', len(paths), args.partition, 'images')
if __name__ == '__main__': main()
