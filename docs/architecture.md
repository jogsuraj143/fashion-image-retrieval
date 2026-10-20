# Architecture and training workflow

## Shared encoder and triplet training

```mermaid
flowchart TD
    A[Anchor image] --> EA[Shared embedding encoder]
    P[Positive image: same item ID] --> EP[Shared embedding encoder]
    N[Negative image: different item ID] --> EN[Shared embedding encoder]
    EA --> ZA[256-D L2-normalized vector A]
    EP --> ZP[256-D L2-normalized vector P]
    EN --> ZN[256-D L2-normalized vector N]
    ZA --> L[Margin-based triplet loss]
    ZP --> L
    ZN --> L
```

The three encoder boxes represent **the same model with shared weights**, not three independently trained networks. The model source is `fashion_assistant/model.py`.

## Encoder internals

```mermaid
flowchart LR
    I[224 x 224 RGB] --> PP[ResNet50 preprocessing]
    PP --> R[ResNet50 ImageNet backbone]
    R --> C[Conv2D 256, 3x3, ReLU]
    C --> G[Global average pooling]
    G --> D[Dense 256]
    D --> U[L2 normalization]
    U --> V[256-D embedding]
```

Training loss uses squared Euclidean distances between unit-normalized embeddings:

`max(0, ||A-P||² - ||A-N||² + 0.2)`

## Inference and retrieval

```mermaid
flowchart LR
    Q[Query clothing image] --> E[Embedding encoder]
    E --> QV[Query embedding]
    G[Official gallery images] --> GE[Same encoder]
    GE --> IDX[Gallery embedding matrix]
    QV --> S[Cosine similarity / dot product]
    IDX --> S
    S --> TOP[Ranked top-5 gallery results]
```

The current project uses an in-memory gallery embedding matrix, not a vector database. Because vectors are L2-normalized, their dot product equals cosine similarity.

## Semi-hard negative mining

```mermaid
flowchart TD
    T[Generate train embeddings with current checkpoint] --> M[Compute anchor-to-train cosine similarities]
    M --> X[Exclude same item ID]
    X --> F[Filter semi-hard candidates]
    F --> H{Candidate exists?}
    H -->|Yes| B[Pick highest-similarity eligible negative]
    H -->|No| R[Fallback random different-item negative]
    B --> TR[Train triplet model]
    R --> TR
    TR --> E[Regenerate embeddings for next mining round]
```

In the supplied miner, candidates satisfy `positive_similarity - 0.2 < negative_similarity < positive_similarity`, with a different-item constraint. **Important:** The cosine-similarity window is not mathematically identical to the triplet loss margin `0.2` in squared Euclidean distance; it is a mining heuristic.
