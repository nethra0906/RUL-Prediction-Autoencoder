"""Deterministic seeding utility. See AI_CONTEXT.md Section 20/21 (reproducibility)."""

import random
import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
