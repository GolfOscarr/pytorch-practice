import random

import numpy as np
import torch


def get_device(prefer: str | None = None) -> torch.device:
    """Return the best available device: cuda > mps (Apple Silicon) > cpu."""
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int = 0) -> None:
    """Seed python, numpy and torch RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
