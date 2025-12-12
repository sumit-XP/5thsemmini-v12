import os
import math
import numpy as np
import torch


def check_nan(x: torch.Tensor) -> bool:
    if not isinstance(x, torch.Tensor):
        return False
    return torch.isnan(x).any().item() or torch.isinf(x).any().item()


def gpu_mem() -> dict:
    if not torch.cuda.is_available():
        return {}
    return {
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
        "max_allocated": torch.cuda.max_memory_allocated(),
        "max_reserved": torch.cuda.max_memory_reserved(),
    }


def set_deterministic(seed: int = 0) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
