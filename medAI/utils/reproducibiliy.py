import logging
import random
import numpy as np
import torch

    
def set_global_seed(seed):

    logging.info(f"Setting global seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # set deterministic cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_all_rng_states():
    return {
        "random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def set_all_rng_states(rng_states):
    logging.info("Setting all RNG states")
    random.setstate(rng_states["random"])
    np.random.set_state(rng_states["numpy"])
    torch.set_rng_state(rng_states["torch_cpu"])
    torch.cuda.set_rng_state_all(rng_states["torch_cuda"])