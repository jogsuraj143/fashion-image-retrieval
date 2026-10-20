# Fashion Image Retrieval with Deep Metric Learning

**DeepFashion In-Shop | ResNet50 | TensorFlow/Keras | Triplet Loss | Semi-hard Negative Mining**

An end-to-end computer vision capstone that retrieves matching clothing product images using a learned 256-dimensional embedding space. Built and iteratively improved through negative-mining experiments, backbone fine-tuning, and retrieval evaluation.

![Experiment progression](assets/experiment_progression.png)

## Key results

| Metric | Best reported score |
|---|---:|
| **Hit@1** | **71.78%** |
| **Hit@5** | **87.92%** |
| Precision@5 | 35.28% |
| Recall@5 | 51.34% |

Measured on **14,218 official query images** against the official DeepFashion In-Shop gallery partition. These are historical results from development, not scores re-evaluated from the downloadable repository. The official query/gallery split was used for repeated experiment selection, so it should not be described as an untouched test set.

## What problem does it solve?

Given an image of a clothing product, find images of the **same catalog item** in a gallery, even when images differ in viewpoint or presentation. This is **identity-based visual retrieval**, not an outfit recommendation system and not yet validated for arbitrary smartphone photos.

## How it works

1. Resize input to 224 × 224 and apply ResNet50 preprocessing.
2. Extract visual features with ImageNet-pretrained ResNet50.
3. Apply Conv2D(256) → global average pooling → Dense(256) → L2 normalization.
4. Train the shared encoder with margin-based triplet loss (anchor, same-item positive, different-item negative).
5. Generate gallery embeddings; rank query/gallery pairs by cosine similarity.
6. Measure Precision@5, Recall@5, Hit@1 and Hit@5 using official item IDs.

**[Architecture and training diagrams](docs/architecture.md)** · **[Full experiment log](docs/experiments.md)**

## Experiment journey

| Strategy | Hit@1 | Hit@5 |
|---|---:|---:|
| Random negatives | 39.48% | 62.93% |
| Hard negatives | 28.64% | 46.10% |
| Semi-hard negatives | 56.57% | 78.20% |
| ResNet50 fine-tuning | 66.13% | 84.53% |
| **Re-mined semi-hard (best)** | **71.78%** | **87.92%** |
| Extra fine-tuning | 71.20% | 87.66% |

The key learning: optimizing training triplet loss does **not** guarantee higher retrieval Hit@K. We stopped further fine-tuning when additional optimization failed to improve retrieval.

## Repository map

```text
fashion_assistant/        Shared encoder, loss, data pipeline, mining code
train.py                  Baseline random-negative training CLI
scripts/                  Consistent embedding generation and retrieval evaluation
experiments/scripts/      Original experiment entry points (historical provenance)
docs/                     Architecture diagrams, experiment results and caveats
assets/                   Experiment progression chart
tests/                    Lightweight source-level tests
```

The original experiment scripts are retained for transparency. They reference historical artifact names and may require editing checkpoint paths and regenerating matching embeddings. They are **not** a one-command reproducible experiment suite.

## Local setup

Recommended: Python 3.11, a TensorFlow-supported environment, and sufficient disk space for DeepFashion and model artifacts. The project was developed on an Apple Silicon Mac.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Apple Silicon with compatible versions: pip install -r requirements-macos-metal.txt
```

Download the **DeepFashion In-Shop Clothes Retrieval** dataset separately, subject to the dataset authors' terms, and place it at:

```text
data/raw/In-Shop Clothes Retrieval/
  Eval/list_eval_partition.txt    # some distributions use eval/
  img/...
```

Alternatively, set `DEEPFASHION_DATA_DIR` to the extracted dataset folder. This repository does not redistribute the images, metadata, model weights or embeddings.

## Run baseline training

```bash
python train.py --epochs 1 --steps-per-epoch 10  # smoke test
python train.py --epochs 10                        # baseline training
```

The baseline produces `models/fashion_embedding_model.keras` and `models/fashion_triplet_model.keras`. It is **not** the best fine-tuned model reported above. Training from scratch requires a pretrained ImageNet backbone download if not cached.

## Generate embeddings and evaluate a checkpoint

Use the **same model checkpoint** for query and gallery embedding generation:

```bash
python -m scripts.generate_embeddings --model models/fashion_embedding_model.keras --partition gallery
python -m scripts.generate_embeddings --model models/fashion_embedding_model.keras --partition query
python -m scripts.evaluate_retrieval
```

To evaluate your best checkpoint, replace the `--model` value with the actual **best** saved `.keras` file. The generator accepts either a single-input embedding model or a three-input triplet model and extracts the shared encoder from the latter.

Embedding files must correspond to the same model weights, preprocessing, and image-path order.

## What I'd improve next

- Add a separate model-selection validation split and freeze the final test protocol.
- Add model/version manifests and automated end-to-end checks.
- Benchmark approximate nearest-neighbor indexing and inference latency.
- Validate retrieval on out-of-distribution phone photos.
- Add a minimal interactive UI after selecting and packaging a compatible best checkpoint.

## Provenance and limitations

This portfolio version was reorganized from the author's supplied local source archive. The original training/evaluation scripts are retained in `experiments/scripts/`; a few convenience scripts and documentation were added. No model was retrained or evaluated while preparing this repository. Historical scores are author-reported and are documented with limitations in [experiments.md](docs/experiments.md).

**Dataset:** DeepFashion In-Shop Clothes Retrieval Benchmark. Credit and follow the dataset authors' distribution/licensing terms.
