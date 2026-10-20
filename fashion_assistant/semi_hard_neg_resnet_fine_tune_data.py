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
def prepare_semi_hard_negative_triplets(
    df: pd.DataFrame,
    seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    embeddings = np.load(
        config.output_dir / "training_embeddings_resnet_fine_tune.npy"
    ).astype(np.float32)

    image_paths = np.load(
        config.output_dir / "traing_image_paths_resnet_fine_tune.npy",
        allow_pickle=True
    ).astype(str)

    train_df = (
        df.set_index("image_name")
          .loc[image_paths]
          .reset_index()
    )

    assert len(train_df) == len(embeddings)
    assert df["image_name"].is_unique

    item_ids = train_df["item_id"].to_numpy()
    image_names = train_df["image_name"].astype(str).to_numpy()

    groups = train_df.groupby("item_id").indices

    anchors, positives, negatives, positive_similarities = [], [], [], []

    # Prepare anchor-positive pairs
    for item_id, group_indices in groups.items():

        if len(group_indices) < 2:
            continue

        positive_indices = np.roll(group_indices, -1)

        # One cosine similarity per anchor-positive pair
        similarity = np.sum(
            embeddings[group_indices] * embeddings[positive_indices],
            axis=1
        )

        positive_similarities.extend(similarity)

        anchors.extend(image_names[group_indices])
        positives.extend(image_names[positive_indices])

    anchors = np.asarray(anchors, dtype=str)
    positives = np.asarray(positives, dtype=str)
    positive_similarities = np.asarray(
        positive_similarities, dtype=np.float32
    )

    path_to_index = {
        path: i for i, path in enumerate(image_names)
    }

    anchor_indices = np.array([
        path_to_index[path] for path in anchors
    ])

    batch_size = 128
    margin = 0.2
    rng = np.random.default_rng(seed)

    for start in range(0, len(anchor_indices), batch_size):

        end = min(start + batch_size, len(anchor_indices))

        batch_indices = anchor_indices[start:end]
        pos_simi = positive_similarities[start:end]

        batch_embeddings = embeddings[batch_indices]

        similarities = batch_embeddings @ embeddings.T

        # Standard semi-hard condition for cosine distance
        semi_hard_mask = (
            (similarities < pos_simi[:, None]) &
            (similarities > pos_simi[:, None] - margin)
        )

        same_item_mask = (
            item_ids[batch_indices, None] == item_ids[None, :]
        )

        semi_hard_mask &= ~same_item_mask

        # Select the hardest eligible semi-hard negative
        closest_indices = np.argmax(
            np.where(semi_hard_mask, similarities, -np.inf),
            axis=1
        )

        # Handle anchors with no semi-hard candidates
        has_semi_hard = semi_hard_mask.any(axis=1)

        for row in np.where(~has_semi_hard)[0]:

            valid_indices = np.flatnonzero(
                ~same_item_mask[row]
            )

            closest_indices[row] = rng.choice(valid_indices)

        negatives.extend(image_names[closest_indices])

        if start % 2560 == 0:
            print(
                f"Processed {end}/{len(anchor_indices)} anchors",
                flush=True
            )

    negatives = np.asarray(negatives, dtype=str)

    assert len(anchors) == len(positives) == len(negatives)

    return anchors, positives, negatives
def prepare_semi_hard_negative_triplets_new1(
    df: pd.DataFrame,
    seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    embeddings = np.load(
        config.output_dir / "training_embeddings_resnet_fine_tune_new1.npy"
    ).astype(np.float32)

    image_paths = np.load(
        config.output_dir / "traing_image_paths_resnet_fine_tune_new1.npy",
        allow_pickle=True
    ).astype(str)

    train_df = (
        df.set_index("image_name")
          .loc[image_paths]
          .reset_index()
    )

    assert len(train_df) == len(embeddings)
    assert df["image_name"].is_unique

    item_ids = train_df["item_id"].to_numpy()
    image_names = train_df["image_name"].astype(str).to_numpy()

    groups = train_df.groupby("item_id").indices

    anchors, positives, negatives, positive_similarities = [], [], [], []

    # Prepare anchor-positive pairs
    for item_id, group_indices in groups.items():

        if len(group_indices) < 2:
            continue

        positive_indices = np.roll(group_indices, -1)

        # One cosine similarity per anchor-positive pair
        similarity = np.sum(
            embeddings[group_indices] * embeddings[positive_indices],
            axis=1
        )

        positive_similarities.extend(similarity)

        anchors.extend(image_names[group_indices])
        positives.extend(image_names[positive_indices])

    anchors = np.asarray(anchors, dtype=str)
    positives = np.asarray(positives, dtype=str)
    positive_similarities = np.asarray(
        positive_similarities, dtype=np.float32
    )

    path_to_index = {
        path: i for i, path in enumerate(image_names)
    }

    anchor_indices = np.array([
        path_to_index[path] for path in anchors
    ])

    batch_size = 128
    margin = 0.2
    rng = np.random.default_rng(seed)

    for start in range(0, len(anchor_indices), batch_size):

        end = min(start + batch_size, len(anchor_indices))

        batch_indices = anchor_indices[start:end]
        pos_simi = positive_similarities[start:end]

        batch_embeddings = embeddings[batch_indices]

        similarities = batch_embeddings @ embeddings.T

        # Standard semi-hard condition for cosine distance
        semi_hard_mask = (
            (similarities < pos_simi[:, None]) &
            (similarities > pos_simi[:, None] - margin)
        )

        same_item_mask = (
            item_ids[batch_indices, None] == item_ids[None, :]
        )

        semi_hard_mask &= ~same_item_mask

        # Select the hardest eligible semi-hard negative
        closest_indices = np.argmax(
            np.where(semi_hard_mask, similarities, -np.inf),
            axis=1
        )

        # Handle anchors with no semi-hard candidates
        has_semi_hard = semi_hard_mask.any(axis=1)

        for row in np.where(~has_semi_hard)[0]:

            valid_indices = np.flatnonzero(
                ~same_item_mask[row]
            )

            closest_indices[row] = rng.choice(valid_indices)

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
