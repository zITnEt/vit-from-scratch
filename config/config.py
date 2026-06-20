from dataclasses import dataclass
from pathlib import Path

import torch

# Absolute paths anchored to the repo root, so scripts work from any CWD.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# CIFAR-10 channel statistics. Shared by training and inference: the model only
# understands inputs normalized with these exact values.
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

# CIFAR-10 class names, indexed by label id.
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


@dataclass(frozen=True)
class Config:
    """Hyperparameters for the Vision Transformer (CIFAR-10 defaults)."""

    # --- Data / image geometry ---
    img_size: int = 32
    patch_size: int = 4
    in_channels: int = 3
    num_classes: int = 10

    # --- Model dimensions ---
    d_model: int = 128
    num_heads: int = 8         # must divide d_model
    d_ff: int = 512
    num_encoder_layers: int = 6
    dropout: float = 0.1

    # --- Training ---
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    num_epochs: int = 50
    warmup_epochs: int = 5     # linear LR warmup before cosine decay

    @property
    def num_patches(self) -> int:
        # Raw patch count, excluding the CLS token added inside the model.
        return (self.img_size // self.patch_size) ** 2

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


config = Config()
