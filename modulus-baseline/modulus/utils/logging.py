# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2025 schups@gmail.com (Modifications)
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from typing import Literal, Dict
from pathlib import Path
import logging
import os
from termcolor import colored

from omegaconf import OmegaConf

import wandb

from modulus.utils.distributed_manager import DistributedManager as DM

def initialize_wandb(
    mode: Literal["offline", "online", "disabled"],
    entity: str,
    project: str,
    name: str,
    experiment_label: str,
    config: Dict = None,
):
    """Function to initialize wandb client with the weights and biases server.

    Parameters
    ----------
    entity : str,
        Name of the wandb entity (account)
    project : str
        Name of the project to sync data with
    name : str, optional
        Name of the task running, by default "train"
    experiment_label : str
        Description of the experiment
    mode: str, optional
        Can be "offline", "online" or "disabled", by default "offline"
    config : optional
        a dictionary-like object for saving inputs , like hyperparameters.
        If dict, argparse or absl.flags, it will load the key value pairs into the
        wandb.config object. If str, it will look for a yaml file by that name,
        by default None.
    """

    time_string = datetime.now().astimezone().strftime("%y/%m/%d_%H:%M:%S")
    slurm_job_id = int(os.getenv("SLURM_JOB_ID", 0))
    name = f"{name}_{slurm_job_id}_{time_string}_{experiment_label}"

    wandb_dir = Path("./wandb").absolute()
    wandb_dir.mkdir(exist_ok=True)

    wandb.init(
        entity=entity,
        project=project,
        name=name,
        config=config,
        mode=mode,
        dir=wandb_dir
    )

class Logger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)
        self.rank = DM.rank()

    def log(self, message: str, all=False):
        if self.rank == 0 or all:
            self.logger.info(message)

    def info(self, message: str, all=False):
        if self.rank == 0 or all:
            self.logger.info(colored(message, "light_blue"))

    def success(self, message: str, all=False):
        if self.rank == 0 or all:
            self.logger.info(colored(message, "light_green"))

    def warning(self, message: str, all=False):
        if self.rank == 0 or all:
            self.logger.warning(colored(message, "light_yellow"))

    def error(self, message: str, all=False):
        if self.rank == 0 or all:
            self.logger.error(colored(message, "light_red"))

def flatten_omegaconf(cfg, prefix=''):
    if OmegaConf.is_config(cfg):
        cfg = OmegaConf.to_container(cfg)
    for k, v in cfg.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict) or OmegaConf.is_config(v):
            yield from flatten_omegaconf(v, full_key)
        else:
            yield full_key, v