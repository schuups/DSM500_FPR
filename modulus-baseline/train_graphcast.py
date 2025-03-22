# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2025 schups@gmail.com (Modifications)
# SPDX-License-Identifier: Apache-2.0

import os
import torch
import time
import hydra
from omegaconf import DictConfig

from modulus.utils.logging import Logger, initialize_wandb 
from modulus.utils.distributed_manager import DistributedManager as DM
from modulus.utils.caching import Cache
from modulus.utils.timer import Timer
from modulus.utils.trainer import GraphCastTrainer

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # initialize DistributedManager and Cache
    DM.initialize()
    Cache.initialize(dir=cfg.cache.dir, verbose=cfg.cache.verbose)

    # initialize logging
    logger = Logger("main")
    logger.info(f"Rank: {DM.rank()}, Device: {DM.device()}", all=True)
    if DM.is_rank_zero():
        initialize_wandb(
            mode=cfg.wb.mode,
            entity=cfg.wb.entity,
            project="DSM500_FPR",
            name=hydra.core.hydra_config.HydraConfig.get().job.name,
            experiment_label=cfg.wb.experiment_label
        )

    # initialize trainer
    trainer = GraphCastTrainer(cfg)

    # initialize dataloaders
    logger.info("Initializing dataloaders...")
    iterator_training = iter(trainer.datapipe_training)
    iterator_testing = iter(trainer.datapipe_testing)

    # training state
    logger.info("Training started...")
    while not trainer.do_terminate:
        trainer.start_iteration()

        # training step
        with Timer() as timer_training:
            try:
                with Timer() as timer_dataloader:
                    train_sample = next(iterator_training)
            except StopIteration:
                logger.info("Re-initializing training dataloader...")
                iterator_training = iter(trainer.datapipe_training)
                with Timer() as timer_dataloader:
                    train_sample = next(iterator_training)
            with Timer() as timer_model:
                loss, global_sample_id = trainer.train(train_sample)

        trainer.log_step(
            type="train",
            metric=loss,
            global_sample_id=global_sample_id,
            timers={
                "training": timer_training,
                "dataloader": timer_dataloader,
                "model": timer_model
            }
        )

        # testing step
        if trainer.do_testing:
            with Timer() as timer_testing:
                try:
                    with Timer() as timer_dataloader:
                        test_sample = next(iterator_testing)
                except StopIteration:
                    logger.info("Re-initializing testing dataloader...")
                    iterator_testing = iter(trainer.datapipe_testing)
                    with Timer() as timer_dataloader:
                        test_sample = next(iterator_testing)
                with Timer() as timer_model:
                    mse, global_sample_id = trainer.test(test_sample)

            trainer.log_step(
                type="test",
                metric=mse,
                global_sample_id=global_sample_id,
                timers={
                    "testing": timer_testing,
                    "dataloader": timer_dataloader,
                    "model": timer_model
                }
            )

        # rollout increase
        if trainer.is_phase3 and trainer.do_rollout_increase:
            del iterator_training
            trainer.increase_rollout()
            logger.info("Re-initializing training dataloader...")
            iterator_training = iter(trainer.datapipe_training)
        
        # checkpoint
        if trainer.do_checkpoint:
            trainer.save_checkpoint(
                iterators={
                    "train": {
                        "initial_epoch_idx": train_sample["epoch_idx"],
                        "initial_sample_idx": train_sample["idx_in_epoch"] + 1
                    },
                    "test": {
                        "initial_epoch_idx": test_sample["epoch_idx"],
                        "initial_sample_idx": test_sample["idx_in_epoch"] + 1
                    },
                }
            )

    logger.info("Training ended")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        raise e
    finally:
        DM.destroy()
