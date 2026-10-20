"""Load DeepFashion In-Shop partitions and build aligned triplets."""
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from fashion_assistant.config import Config
config=Config()


def load_train_dataframe(dataset_root: Path) -> pd.DataFrame:
    partition_file = dataset_root / 'Eval' / 'list_eval_partition.txt'
    if not partition_file.is_file():
        raise FileNotFoundError(f'Missing partition file: {partition_file}')
    df = pd.read_csv(partition_file, sep=r'\s+', skiprows=1)
    needed = {'image_name', 'item_id', 'evaluation_status'}
    if not needed.issubset(df.columns):
        raise ValueError(f'Partition file missing columns: {needed - set(df.columns)}')
    df = df.loc[df['evaluation_status'] == 'train', ['image_name', 'item_id']].copy()
    return df.reset_index(drop=True)

def read_image(relative_path):
    filename = tf.strings.join([
        str(config.dataset_root) + "/",
        relative_path
    ])
    raw = tf.io.read_file(filename)
    img = tf.io.decode_jpeg(raw, channels=3)
    img = tf.image.resize(img, (224, 224))
    return preprocess_input(img)

def prepare_hard_negative_triplets(
    df: pd.DataFrame,
    seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    embeddings = np.load(
        config.output_dir / "training_embeddings.npy"
    ).astype(np.float32)

    image_paths = np.load(
        config.output_dir / "traing_image_paths.npy",
        allow_pickle=True
    ).astype(str)

    # Match DataFrame rows to saved embedding order
    train_df = (
        df.set_index("image_name")
          .loc[image_paths]
          .reset_index()
    )

    assert len(train_df) == len(embeddings)
    assert df["image_name"].is_unique

    # Convert DataFrame columns to NumPy arrays
    item_ids = train_df["item_id"].to_numpy()
    image_names = train_df["image_name"].astype(str).to_numpy()

    # Find item groups only once
    groups = train_df.groupby("item_id").indices

    anchors, positives, negatives = [], [], []

    # Prepare anchor-positive pairs
    for item_id, group_indices in groups.items():

        if len(group_indices) < 2:
            continue

        # Next image of same item, wrapping to first
        positive_indices = np.roll(group_indices, -1)

        anchors.extend(image_names[group_indices])
        positives.extend(image_names[positive_indices])

    anchors = np.asarray(anchors, dtype=str)
    positives = np.asarray(positives, dtype=str)

    # Map anchor paths to their embedding positions
    path_to_index = {
        path: i for i, path in enumerate(image_names)
    }

    anchor_indices = np.array([
        path_to_index[path] for path in anchors
    ])

    # Calculate similarities in batches to limit RAM usage
    batch_size = 128

    for start in range(0, len(anchor_indices), batch_size):

        end = min(start + batch_size, len(anchor_indices))

        batch_indices = anchor_indices[start:end]

        batch_embeddings = embeddings[batch_indices]

        # Shape: (batch_size, total_training_images)
        similarities = batch_embeddings @ embeddings.T

        # Exclude images belonging to the same item ID
        same_item_mask = (
            item_ids[batch_indices, None] == item_ids[None, :]
        )

        similarities[same_item_mask] = -np.inf

        # Closest different-item image for each anchor
        closest_indices = np.argmax(similarities, axis=1)

        negatives.extend(image_names[closest_indices])

        if start % 2560 == 0:
            print(
                f"Processed {end}/{len(anchor_indices)} anchors",
                flush=True
            )

    negatives = np.asarray(negatives, dtype=str)

    assert len(anchors) == len(positives) == len(negatives)

    return anchors, positives, negatives


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
