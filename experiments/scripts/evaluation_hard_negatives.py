
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


embeddings = np.load(
    config.output_dir / "gallery_embeddings_hard_negatives.npy"
)

image_paths = np.load(
    config.output_dir / "gallery_image_paths_hard_negatives.npy",
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

model_path = config.output_dir / "fashion_embedding_model_hard_negatives.keras"

model = tf.keras.models.load_model(
    model_path,
    compile=False
)


# Query partition
gallery_df = df[df["evaluation_status"] == "query"]

first_query = gallery_df.iloc[0]
image_path = str(first_query["image_name"])

pixes = read_image(image_path)
print(pixes.shape)

ext_dim = np.expand_dims(pixes, axis=0)

queryEmbedding = model.predict(ext_dim, verbose=0)
print(f"queryEmbedding: {queryEmbedding.shape}")


# Cosine similarity (embeddings are unit normalized)
cosine_similarity = embeddings @ queryEmbedding[0]

# Exclude the query image itself if present
gallery_paths = np.asarray(image_paths).astype(str)
cosine_similarity[gallery_paths == image_path] = -np.inf

top5_search = cosine_similarity.argsort()[-5:][::-1]


paths = [image_path] + [
    str(image_paths[i]) for i in top5_search
]

fig, axes = plt.subplots(1, 6, figsize=(18, 4))

query_item_id = first_query["item_id"]

for i, path in enumerate(paths):
    image = Image.open(Path(config.dataset_root) / path)
    axes[i].imshow(image)
    axes[i].axis("off")

    if i == 0:
        axes[i].set_title(
            f"Query\nItem ID: {query_item_id}"
        )
    else:
        idx = top5_search[i - 1]

        score = cosine_similarity[idx]

        # FIX: Lookup in full metadata, not query-only gallery_df
        result_item_id = path_to_item.get(
            str(image_paths[idx])
        )

        if result_item_id is None:
            status = "UNKNOWN"
        elif result_item_id == query_item_id:
            status = "CORRECT"
        else:
            status = "WRONG"

        axes[i].set_title(
            f"Match {i}\n"
            f"Score: {score:.3f}\n"
            f"Item ID: {result_item_id}\n"
            f"{status}"
        )

plt.tight_layout()
plt.show()


df.reset_index(drop=True,inplace=True)
counts_map=df["item_id"].value_counts()




