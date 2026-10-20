"""Load DeepFashion In-Shop partitions and build aligned triplets."""
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input


def load_train_dataframe(dataset_root: Path) -> pd.DataFrame:
    partition_file = dataset_root / 'Eval' / 'list_eval_partition.txt'
    if not partition_file.is_file():
        partition_file = dataset_root / 'eval' / 'list_eval_partition.txt'
    if not partition_file.is_file():
        raise FileNotFoundError(f'Missing partition file: {partition_file}')
    df = pd.read_csv(partition_file, sep=r'\s+', skiprows=1)
    needed = {'image_name', 'item_id', 'evaluation_status'}
    if not needed.issubset(df.columns):
        raise ValueError(f'Partition file missing columns: {needed - set(df.columns)}')
    df = df.loc[df['evaluation_status'] == 'train', ['image_name', 'item_id']].copy()
    return df.reset_index(drop=True)


def prepare_triplets(df: pd.DataFrame, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same-item next image as positive; random different-item image as negative.

    Uses item-level sampling rather than repeatedly filtering the whole dataframe.
    """
    rng = np.random.default_rng(seed)
    groups = {key: group['image_name'].to_numpy() for key, group in df.groupby('item_id', sort=False)}
    eligible = {key: images for key, images in groups.items() if len(images) >= 2}
    if len(groups) < 2 or not eligible:
        raise ValueError('Need at least two item IDs and one item with two images.')
    all_ids = np.array(list(groups.keys()), dtype=object)
    anchors, positives, negatives = [], [], []
    for item_id, images in eligible.items():
        other_ids = all_ids[all_ids != item_id]
        for index, path in enumerate(images):
            negative_id = rng.choice(other_ids)
            negative_images = groups[negative_id]
            anchors.append(path)
            positives.append(images[(index + 1) % len(images)])
            negatives.append(rng.choice(negative_images))
    return (np.asarray(anchors, dtype=str),
            np.asarray(positives, dtype=str),
            np.asarray(negatives, dtype=str))


def make_dataset(paths, dataset_root: Path, batch_size=32, seed=42,
                 shuffle_buffer=2048, training=True):
    root = tf.constant(str(dataset_root.resolve()) + '/', dtype=tf.string)

    def read_image(relative_path):
        filename = tf.strings.join([root, relative_path])
        raw = tf.io.read_file(filename)
        img = tf.io.decode_jpeg(raw, channels=3)
        img = tf.image.resize(img, (224, 224))
        return preprocess_input(img)

    def load_triplet(anchor, positive, negative):
        return ((read_image(anchor), read_image(positive), read_image(negative)),
                tf.constant(0., dtype=tf.float32))

    ds = tf.data.Dataset.from_tensor_slices(paths)
    if training:
        ds = ds.shuffle(min(len(paths[0]), shuffle_buffer), seed=seed,
                        reshuffle_each_iteration=True)
    ds = ds.map(load_triplet, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
