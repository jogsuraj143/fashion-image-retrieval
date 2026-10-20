"""Local training entry point. Run from the project root: python train.py"""
import argparse
from pathlib import Path
import tensorflow as tf
from fashion_assistant.config import Config
from fashion_assistant.data import load_train_dataframe, prepare_triplets, make_dataset
from fashion_assistant.model import build_embedding_model, build_triplet_model


def main():
    defaults = Config()
    parser = argparse.ArgumentParser(description='Train DeepFashion triplet embeddings locally')
    parser.add_argument('--data-dir', type=Path, default=defaults.dataset_root)
    parser.add_argument('--epochs', type=int, default=defaults.epochs)
    parser.add_argument('--batch-size', type=int, default=defaults.batch_size)
    parser.add_argument('--steps-per-epoch', type=int, default=None,
                        help='Optional small test run, e.g. 10')
    args = parser.parse_args()

    print('TensorFlow:', tf.__version__)
    print('Detected GPUs:', tf.config.list_physical_devices('GPU'))
    print('Dataset root:', args.data_dir.resolve())
    df = load_train_dataframe(args.data_dir)
    print('Training images:', len(df), '| unique items:', df.item_id.nunique())
    paths = prepare_triplets(df, seed=defaults.seed)
    print('Triplets:', len(paths[0]))
    ds = make_dataset(paths, args.data_dir, batch_size=args.batch_size,
                      seed=defaults.seed, shuffle_buffer=defaults.shuffle_buffer)

    embedding_model = build_embedding_model(defaults.embedding_dim)
    model = build_triplet_model(embedding_model, defaults.learning_rate)
    defaults.output_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [tf.keras.callbacks.ModelCheckpoint(
        filepath=str(defaults.output_dir / 'best_triplet_model.keras'),
        monitor='loss', save_best_only=True, verbose=1)]
    model.fit(ds, epochs=args.epochs, steps_per_epoch=args.steps_per_epoch,
              callbacks=callbacks, verbose=1)
    embedding_model.save(str(defaults.output_dir / 'fashion_embedding_model.keras'))
    model.save(str(defaults.output_dir / 'fashion_triplet_model.keras'))
    print('Saved models to:', defaults.output_dir)


if __name__ == '__main__':
    main()
