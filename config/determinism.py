"""Global reproducibility configuration for the MLGenX Track B pipeline.

Forces deterministic execution across Python, PyTorch, XGBoost, and numerical backends
to guarantee identical tool-call sequences and LLM outputs across repeated sweeps.
"""

import os
import random
import numpy as np


def configure_determinism(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    try:
        import xgboost as xgb
        xgb.set_config(verbosity=0)
    except ImportError:
        pass
