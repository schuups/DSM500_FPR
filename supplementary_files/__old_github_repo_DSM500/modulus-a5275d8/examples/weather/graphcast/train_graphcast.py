# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2025 schups@gmail.com (Modifications)
# SPDX-License-Identifier: Apache-2.0

import sys
sys.path.append('/iopsstor/scratch/cscs/stefschu/DSM500/github/modulus-a5275d8')

import os
import torch
import time

from modulus.launch.logging import (
    PythonLogger,
    initialize_wandb,
    RankZeroLoggingWrapper,
)

from modulus.distributed import DistributedManager
from modulus.utils_new.caching import Cache

import hydra
from omegaconf import DictConfig
from trainer import GraphCastTrainer

from modulus.launch.utils import save_checkpoint

# import torch._dynamo
# torch._dynamo.config.suppress_errors = True  # TODO check if this can be removed

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # initialize DistributedManager
    DistributedManager.initialize()
    dist = DistributedManager()

    # initialize Cache
    Cache.initialize(dir=cfg.cache_dir)

    # initialize loggers
    if dist.rank == 0:
        initialize_wandb(
            project="GraphCast",
            entity=cfg.wb_entity,
            name=hydra.core.hydra_config.HydraConfig.get().job.name,
            group="group",
            mode=cfg.wb_mode,
        )
    logger = PythonLogger("main")
    logger.info(f"Rank: {dist.rank}, Device: {dist.device}")
    zlogger = RankZeroLoggingWrapper(logger, dist)
    zlogger.file_logging()

    # initialize trainer
    trainer = GraphCastTrainer(cfg, dist, zlogger)

    # training state
    zlogger.info("Training started...")
    update_datapipe = False
    while trainer.do_terminate is False:
        for sample in trainer.training_datapipe:
            trainer.iter_start()

            # training step
            time_training_start = time.time()
            
            data = sample[0]["data"].to(dtype=trainer.dtype)
            global_sample_id = sample[0]["global_sample_id"].item()

            reanalysis, generated = torch.split(data, [trainer.training_datapipe.channels_count, trainer.training_datapipe.generated_channels_count], dim=2)
            loss = trainer.train(reanalysis, generated)
            trainer.log_train_step(
                loss=loss,
                time_taken=time.time()-time_training_start,
                global_sample_id=global_sample_id,
            )

            if trainer.do_testing or trainer.do_produce_artifacts:
                time_testing_start = time.time()
                test_mse = trainer.test()
                trainer.log_test_step(
                    test_mse=test_mse,
                    time_taken=time.time()-time_testing_start
                )

                if trainer.do_produce_artifacts:
                    raise NotImplementedError("Artifacts generation not yet implemented")

            if trainer.is_phase3 and trainer.do_rollout_increase:
                trainer.increase_rollout()
                zlogger.info(f"Switched to {trainer.current_rollout_steps}-long rollouts")
                update_datapipe = True # provove a break to force dataloader reinitialization

            if trainer.do_checkpoint:
                trainer.save_checkpoint(
                    training_datapipe_epoch_idx=data[0]["epoch_idx"].item(),
                    training_datapipe_sample_in_epoch=data[0]["idx_in_epoch"].item(),
                )

            if trainer.do_terminate or update_datapipe:
                break
    
    zlogger.info("Training ended")

if __name__ == "__main__":
    main()
