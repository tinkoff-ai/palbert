import os
import random

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_deterministic_mode(seed):
    set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.deterministic = True
    torch.backends.benchmark = False

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
