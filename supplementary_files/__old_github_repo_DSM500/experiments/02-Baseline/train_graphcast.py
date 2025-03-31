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

@hydra.main(version_base="1.3", config_path=".", config_name="config")
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
            experiment_label=cfg.wb_experiment_label
        )
    logger = PythonLogger("main")
    logger.info(f"Rank: {dist.rank}, Device: {dist.device}")
    zlogger = RankZeroLoggingWrapper(logger, dist)

    # initialize trainer
    trainer = GraphCastTrainer(cfg, dist, zlogger)

    # initialize dataloaders
    zlogger.info("Initializing dataloaders...")
    training_iterator = iter(trainer.training_datapipe)
    testing_iterator = iter(trainer.testing_datapipe)

    # training state
    zlogger.info("Training started...")
    while not trainer.do_terminate:
        trainer.iter_start()

        # training step
        _time_training_start = time.perf_counter()

        try:
            _time_dataloader_start = time.perf_counter()
            train_sample = next(training_iterator)
            _time_dataloader_end = time.perf_counter()
        except StopIteration:
            zlogger.info("Re-initializing training dataloader...")
            training_iterator = iter(trainer.training_datapipe)
            _time_dataloader_start = time.perf_counter()
            train_sample = next(training_iterator)
            _time_dataloader_end = time.perf_counter()
        
        _time_model_start = time.perf_counter()
        loss, global_sample_id = trainer.train(train_sample)
        _time_model_end = time.perf_counter()

        trainer.log_train_step(
            loss=loss,
            time_start=_time_training_start,
            time_dataloader=_time_dataloader_end - _time_dataloader_start,
            time_model=_time_model_end - _time_model_start,
            global_sample_id=global_sample_id,
        )

        # testing step
        if trainer.do_testing:
            _time_testing_start = time.perf_counter()

            try:
                _time_dataloader_start = time.perf_counter()
                test_sample = next(testing_iterator)
                _time_dataloader_end = time.perf_counter()
            except StopIteration:
                zlogger.info("Re-initializing testing dataloader...")
                testing_iterator = iter(trainer.testing_datapipe)
                _time_dataloader_start = time.perf_counter()
                test_sample = next(testing_iterator)
                _time_dataloader_end = time.perf_counter()

            _time_model_start = time.perf_counter()
            mse, global_sample_id = trainer.test(test_sample)
            _time_model_end = time.perf_counter()

            trainer.log_test_step(
                mse=mse,
                time_start=_time_testing_start,
                time_dataloader=_time_dataloader_end - _time_dataloader_start,
                time_model=_time_model_end - _time_model_start,
                global_sample_id=global_sample_id,
            )  

        # rollout increase
        if trainer.is_phase3 and trainer.do_rollout_increase:
            del training_iterator
            trainer.increase_rollout()
            zlogger.info(f"Switched to {trainer.current_rollout_steps}-long rollouts.")
            zlogger.info("Re-initializing training dataloader...")
            training_iterator = iter(trainer.training_datapipe)
        
        # checkpoint
        if trainer.do_checkpoint:
            trainer.save_checkpoint(
                training_datapipe_epoch_idx=train_sample[0]["epoch_idx"].item(),
                training_datapipe_sample_in_epoch=train_sample[0]["idx_in_epoch"].item(),
                testing_datapipe_epoch_idx=test_sample[0]["epoch_idx"].item(),
                testing_datapipe_sample_in_epoch=test_sample[0]["idx_in_epoch"].item(),
            )
    
    zlogger.info("Training ended")

if __name__ == "__main__":
    main()
