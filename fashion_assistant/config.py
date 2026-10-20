from dataclasses import dataclass
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]

@dataclass
class Config:
    dataset_root: Path = Path(os.environ.get('DEEPFASHION_DATA_DIR', str(PROJECT_ROOT / 'data' / 'raw' / 'In-Shop Clothes Retrieval')))
    output_dir: Path = Path(os.environ.get('FASHION_MODELS_DIR', str(PROJECT_ROOT / 'models')))
    image_size: tuple[int, int] = (224, 224)
    embedding_dim: int = 256
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 1e-4
    margin: float = 0.2
    seed: int = 42
    shuffle_buffer: int = 2048
