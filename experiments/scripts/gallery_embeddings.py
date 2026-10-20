import numpy as np
from fashion_assistant.config import Config
import os
import pandas as pd
import fashion_assistant.data as preprocessor
from tensorflow.keras.applications.resnet50 import preprocess_input
from pathlib import Path
import tensorflow as tf


def read_image(relative_path):
    filename = tf.strings.join([str(config.dataset_root)+ "/", relative_path])
    raw = tf.io.read_file(filename)
    img = tf.io.decode_jpeg(raw, channels=3)
    img = tf.image.resize(img, (224, 224))
    return preprocess_input(img)

config=Config()

print(config.dataset_root)
EVAL_FILE=os.path.join(config.dataset_root,"eval","list_eval_partition.txt")

df=pd.read_csv(EVAL_FILE,sep=r"\s+",skiprows=1)

model_path = Path("models/fashion_embedding_model.keras")

model = tf.keras.models.load_model(
    model_path,
    compile=False
)


gallery_df=df[df["evaluation_status"]=="gallery"]

#few_images=gallery_df.head(10)

# inputs = tf.stack(
#     few_images["image_name"].transform(lambda x: read_image(x)).tolist()
# )

gallery_data=gallery_df["image_name"].astype(str).to_numpy()
dataset=tf.data.Dataset.from_tensor_slices(gallery_data).map(lambda image_path: read_image(image_path),num_parallel_calls=tf.data.AUTOTUNE).batch(16).prefetch(tf.data.AUTOTUNE)


embeddings=model.predict(dataset,verbose=1)
np.save(config.output_dir / "gallery_embeddings.npy", embeddings)
# Save matching image paths in the same order
np.save(config.output_dir / "gallery_image_paths.npy", gallery_data)

print(embeddings[0])


# print(f"data:{df.shape}")
# print(f"gallery:{gallery_df.shape}")
# print(df.info)


