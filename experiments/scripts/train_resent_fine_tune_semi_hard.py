"""Fine-tune ResNet50 using the trained semi-hard-negative triplet model."""

import argparse
from pathlib import Path
import tensorflow as tf

from fashion_assistant.config import Config
from fashion_assistant import data
from fashion_assistant.model import triplet_loss
from fashion_assistant.data import (
    load_train_dataframe,
    make_dataset
)
from fashion_assistant.semi_hard_neg_resnet_fine_tune_data import (
    prepare_semi_hard_negative_triplets
)


def main():
    defaults = Config()

    parser = argparse.ArgumentParser(
        description="Fine-tune ResNet50 for DeepFashion retrieval"
    )
    parser.add_argument('--data-dir', type=Path, default=defaults.dataset_root)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--steps-per-epoch', type=int, default=None)

    args = parser.parse_args()

    print('TensorFlow:', tf.__version__)
    print('Detected GPUs:', tf.config.list_physical_devices('GPU'))
    print('Dataset root:', args.data_dir.resolve())

    # Load previously trained triplet model
    embedding_model = tf.keras.models.load_model(
        str(
            defaults.output_dir /
            "fashion_triplet_model_finetuned.keras"
        ),
        compile=False
    )

    # Extract the shared embedding model
    fasion_model = embedding_model.get_layer("fashion_embedding_model")

    # Extract pretrained ResNet50
    base_model = fasion_model.get_layer("resnet50")

    # Enable fine-tuning
    base_model.trainable = True

    # Freeze all except last 30 ResNet50 layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    # Freeze BatchNormalization
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    # Verify trainable layers
    print("\nResNet50 last 30 layers:")
    for layer in base_model.layers[-30:]:
        print(layer.name, layer.trainable)

    print(
        "Trainable parameters:",
        sum(int(tf.size(w)) for w in embedding_model.trainable_weights)
    )

    # Compile the existing triplet model
    embedding_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=triplet_loss
    )

    model = embedding_model

    # Prepare training data
    df = load_train_dataframe(args.data_dir)

    print(
        'Training images:', len(df),
        '| unique items:', df.item_id.nunique()
    )

    # IMPORTANT:
    # Regenerate training_embeddings.npy using the best
    # semi-hard embedding model before mining these triplets.
    paths = prepare_semi_hard_negative_triplets(
        df,
        seed=defaults.seed
    )

    print('Triplets:', len(paths[0]))

    ds = make_dataset(
        paths,
        args.data_dir,
        batch_size=args.batch_size,
        seed=defaults.seed,
        shuffle_buffer=defaults.shuffle_buffer
    )

    defaults.output_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(
                defaults.output_dir /
                'best_triplet_model_finetuned.keras'
            ),
            monitor='loss',
            save_best_only=True,
            verbose=1
        )
    ]

    model.fit(
        ds,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        callbacks=callbacks,
        verbose=1
    )

    # Save the single-input embedding model
    fasion_model.save(
        str(defaults.output_dir / 'fashion_embedding_model_finetuned_new.keras')
    )

    # Save the three-input triplet model
    model.save(
        str(defaults.output_dir / 'fashion_triplet_model_finetuned_new.keras')
    )

    print('Saved fine-tuned models to:', defaults.output_dir)


if __name__ == '__main__':
    main()