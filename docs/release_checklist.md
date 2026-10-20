# Pre-publication checklist

- [ ] Confirm your **71.78% Hit@1 checkpoint** is preserved locally, with a matching encoder and gallery embeddings.
- [ ] Run `python train.py --epochs 1 --steps-per-epoch 10` against your dataset.
- [ ] Run `python -m scripts.generate_embeddings` for gallery and query using the **same checkpoint**.
- [ ] Run `python -m scripts.evaluate_retrieval` and compare with historical metrics.
- [ ] Confirm there are no data, model, private credentials or local paths in `git status`.
- [ ] Add a few licensed example retrieval screenshots if permitted by the dataset terms.
- [ ] Add GitHub repository URL to resume / LinkedIn.
- [ ] Do not describe the reported metrics as independent held-out test performance.
