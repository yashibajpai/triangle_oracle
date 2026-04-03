import os
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    set random seeds across Python, NumPy, and PyTorch.

    helps to make experiments reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # if CUDA is available, seed CUDA too.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ask PyTorch for more deterministic behavior where possible.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # optional environment variable for extra reproducibility.
    os.environ["PYTHONHASHSEED"] = str(seed)