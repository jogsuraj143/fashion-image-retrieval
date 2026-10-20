
import numpy as np
from fashion_assistant.config import Config
import os
import pandas as pd
import tensorflow as tf
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import preprocess_input


config = Config()


def read_image(relative_path):
    filename = tf.strings.join([
        str(config.dataset_root) + "/",
        relative_path
    ])
    raw = tf.io.read_file(filename)
    img = tf.io.decode_jpeg(raw, channels=3)
    img = tf.image.resize(img, (224, 224))
    return preprocess_input(img)

EVAL_FILE = os.path.join(
    config.dataset_root,
    "eval",
    "list_eval_partition.txt"
)

df = pd.read_csv(
    EVAL_FILE,
    sep=r"\s+",
    skiprows=1
)

embeddings = np.load(
    config.output_dir / "gallery_embeddings_resnet_fine_tune_new.npy"
)

image_paths = np.load(
    config.output_dir / "gallery_image_paths_resnet_fine_tune_new.npy",
    allow_pickle=True
)



print("Embeddings:", embeddings.shape, embeddings.dtype)
print("Image paths:", image_paths.shape, image_paths.dtype)

assert len(embeddings) == len(image_paths)


print(config.dataset_root)

EVAL_FILE = os.path.join(
    config.dataset_root,
    "eval",
    "list_eval_partition.txt"
)

df = pd.read_csv(
    EVAL_FILE,
    sep=r"\s+",
    skiprows=1
)

# Use the full df to look up item IDs
path_to_item = dict(zip(
    df["image_name"].astype(str),
    df["item_id"]
))

model_path = config.output_dir / "fashion_embedding_model_finetuned_new.keras"

model = tf.keras.models.load_model(
    model_path,
    compile=False
)


# Query partition

query_df = df[df["evaluation_status"] == "query"]

# Use batches to avoid loading all images into RAM
query_paths = query_df["image_name"].astype(str).to_numpy()

query_dataset = (
    tf.data.Dataset.from_tensor_slices(query_paths)
    .map(read_image, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(16)
    .prefetch(tf.data.AUTOTUNE)
)

queryEmbedding = model.predict(query_dataset, verbose=1)
print(f"queryEmbedding: {queryEmbedding.shape}")


# Build image path -> item ID mapping
path_to_item = dict(zip(
    df["image_name"].astype(str),
    df["item_id"].astype(str)
))

# Keep only official gallery images
official_gallery_paths = set(
    df[df["evaluation_status"] == "gallery"]["image_name"].astype(str)
)

gallery_paths = np.asarray(image_paths).astype(str)

gallery_mask = np.array([
    path in official_gallery_paths
    for path in gallery_paths
])

gallery_embeddings = embeddings[gallery_mask]
gallery_paths = gallery_paths[gallery_mask]

gallery_item_ids = np.array([
    path_to_item.get(path)
    for path in gallery_paths
])

assert len(gallery_embeddings) >= 5
assert all(item is not None for item in gallery_item_ids)


# Count only images available for retrieval
counts_map = pd.Series(gallery_item_ids).value_counts()

precisions = []
recalls = []
hits_at_one = []
hits_at_five = []

# Cosine similarity matrix
cosine_similarities = queryEmbedding @ gallery_embeddings.T

for i in range(len(queryEmbedding)):

    top_5 = np.argsort(cosine_similarities[i])[-5:][::-1]

    match_count = 0
    first_match = 0

    query_item_id = str(query_df.iloc[i]["item_id"])
    items_count = counts_map.get(query_item_id, 0)

    if items_count == 0:
        continue

    for rank, index in enumerate(top_5):

        result_item_id = gallery_item_ids[index]

        if query_item_id == result_item_id:
            match_count += 1

            if rank == 0:
                first_match = 1

    precisions.append(match_count / 5)
    recalls.append(match_count / items_count)
    hits_at_one.append(first_match)
    hits_at_five.append(int(match_count > 0))


print("final model stats....")
print("Evaluated queries:", len(precisions))
print(f"precision@5: {np.mean(precisions):.4f}")
print(f"recall@5: {np.mean(recalls):.4f}")
print(f"hits_at_one: {np.mean(hits_at_one):.4f}")
print(f"hits_at_five: {np.mean(hits_at_five):.4f}")
# queryEmbedding: (14218, 256)
# final model stats....
# Evaluated queries: 14218
# precision@5: 0.1960
# recall@5: 0.2841
# hits_at_one: 0.3948
# hits_at_five: 0.6293