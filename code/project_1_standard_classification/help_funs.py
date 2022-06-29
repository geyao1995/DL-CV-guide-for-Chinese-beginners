import random
import warnings

import numpy as np
import torch


def set_seed(seed: int = 0):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
    else:
        device = torch.device('cpu')
        warnings.warn("You are using CPU!")

    return device
