import logging
import sys
import random
import numpy as np
import torch
import os

from lib.activity import Activity

class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[94m',    # Blue
        'INFO': '\033[92m',     # Green
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',    # Red
        'CRITICAL': '\033[95m'  # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        formatted = super().format(record)
        color = self.COLORS.get(record.levelname.strip(self.RESET), self.RESET)
        return f"{color}{formatted}{self.RESET}"

def get_logger(rank, world_size):
    logger_name = f"Rank {rank}/{world_size}"
    logger = logging.getLogger(logger_name)

    # Prevent adding multiple handlers
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)

        formatter = ColoredFormatter(
            '[%(asctime)s][%(name)s][%(levelname)s] %(message)s'
        )
        handler.setFormatter(formatter)

        logger.addHandler(handler)
        logger.propagate = False  # Avoid duplication through propagation

    return logger

def set_seed(seed):
    # Set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_distribution_context():
    """
    Load the distributed context from the environment variables.
    """

    if "LOCAL_RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ.get("LOCAL_RANK"))
        world_size = int(os.environ.get("WORLD_SIZE"))
    elif "SLURM_PROCID" in os.environ and "SLURM_NTASKS" in os.environ:
        rank = int(os.environ.get("SLURM_PROCID"))
        world_size = int(os.environ.get("SLURM_NTASKS"))
    else:
        raise ValueError("No distributed context found.")
    
    devices_count = torch.cuda.device_count()
    assert devices_count > 0, "No CUDA devices found."
    device = torch.device('cuda', index=rank % devices_count)

    return rank, world_size, device

def build_activities(cfg):
    models = cfg.models
    data = cfg.data
    activities = []
    for model in models:
        for weights in model.weights:
            for ic_i in data.initial_conditions_idx:
                activities.append(
                    Activity(
                        id=len(activities),
                        inference_rollout_steps=cfg.inference.rollout_steps,

                        model_name=model.name,
                        model_type=model.type,
                        model_code_path=model.code_path,
                        model_config_path=model.config_path,

                        weights_file_path=weights,

                        dataset_metadata=data.metadata,
                        dataset_file_path=data.file_path,
                        dataset_initial_condition_i=ic_i,
                    )
                )
    return activities