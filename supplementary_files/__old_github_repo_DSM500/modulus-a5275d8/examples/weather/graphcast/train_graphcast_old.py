# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
sys.path.append('/iopsstor/scratch/cscs/stefschu/DSM500/github/modulus-a5275d8')

import torch
import apex
from contextlib import nullcontext
from torch.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel
import numpy as np
import time
import wandb
import torch.cuda.profiler as profiler
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, LambdaLR

import torch._dynamo

torch._dynamo.config.suppress_errors = True  # TODO check if this can be removed

# import modules
import os

from modulus.models.graphcast.graph_cast_net import GraphCastNet
from modulus.utils.graphcast.loss import (
    CellAreaWeightedLossFunction,
    GraphCastLossFunction,
)
from modulus.launch.logging import (
    PythonLogger,
    initialize_wandb,
    RankZeroLoggingWrapper,
)
from modulus.launch.utils import load_checkpoint, save_checkpoint

from train_utils import count_trainable_params, prepare_input
from loss.utils import normalized_grid_cell_area
from train_base import BaseTrainer
from validation_base import Validation
from modulus.datapipes.climate.era5_hdf5_new import ERA5HDF5Datapipe
from modulus.distributed import DistributedManager
from modulus.utils.graphcast.data_utils import StaticData

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig


class GraphCastTrainer(BaseTrainer):
    """GraphCast Trainer"""

    def __init__(self, cfg: DictConfig, dist, rank_zero_logger):
        super().__init__()
        self.dist = dist
        self.dtype = torch.bfloat16 if cfg.full_bf16 else torch.float32
        self.enable_scaler = False
        self.amp = cfg.amp
        self.amp_dtype = None
        self.pyt_profiler = cfg.pyt_profiler
        self.grad_clip_norm = cfg.grad_clip_norm

        if cfg.full_bf16:
            assert torch.cuda.is_bf16_supported()
            rank_zero_logger.info(f"Using {str(self.dtype)} dtype")
            if cfg.amp:
                raise ValueError(
                    "Full bfloat16 training is enabled, switch off amp in config"
                )

        if cfg.amp:
            rank_zero_logger.info(f"Using config amp with dtype {cfg.amp_dtype}")
            if cfg.amp_dtype == "float16" or cfg.amp_dtype == "fp16":
                self.amp_dtype = torch.float16
                self.enable_scaler = True
            elif self.amp_dtype == "bfloat16" or self.amp_dtype == "bf16":
                self.amp_dtype = torch.bfloat16
            else:
                raise ValueError("Invalid dtype for config amp")

        # instantiate the model
        self.model = GraphCastNet(
            mesh_level=cfg.mesh_level,
            multimesh=cfg.multimesh,
            input_res=tuple(cfg.latlon_res),
            input_dim_grid_nodes=31, # TODO: make this a parameter
            input_dim_mesh_nodes=3,
            input_dim_edges=4,
            output_dim_grid_nodes=cfg.num_channels_climate,
            processor_type=cfg.processor_type,
            khop_neighbors=cfg.khop_neighbors,
            num_attention_heads=cfg.num_attention_heads,
            processor_layers=cfg.processor_layers,
            hidden_dim=cfg.hidden_dim,
            norm_type=cfg.norm_type,
            do_concat_trick=cfg.concat_trick,
            use_cugraphops_encoder=cfg.cugraphops_encoder,
            use_cugraphops_processor=cfg.cugraphops_processor,
            use_cugraphops_decoder=cfg.cugraphops_decoder,
            recompute_activation=cfg.recompute_activation,
        )

        # set gradient checkpointing
        if cfg.force_single_checkpoint:
            self.model.set_checkpoint_model(True)
        if cfg.checkpoint_encoder:
            self.model.set_checkpoint_encoder(True)
        if cfg.checkpoint_processor:
            self.model.set_checkpoint_processor(cfg.segments)
        if cfg.checkpoint_decoder:
            self.model.set_checkpoint_decoder(True)

        # JIT compile the model, and specify the device and dtype
        if cfg.jit:
            torch.jit.script(self.model).to(dtype=self.dtype).to(device=dist.device)
            rank_zero_logger.success("JIT compiled the model")
        else:
            self.model = self.model.to(dtype=self.dtype).to(device=dist.device)
        if cfg.watch_model and not cfg.jit and dist.rank == 0:
            wandb.watch(self.model)

        # Get required model attributes
        if hasattr(self.model, "module"):
            self.latitudes = self.model.module.latitudes
            self.longitudes = self.model.module.longitudes
            self.lat_lon_grid = self.model.module.lat_lon_grid
            self.is_distributed = self.model.module.is_distributed
            self.expect_partitioned_input = self.model.module.expect_partitioned_input
        else:
            self.latitudes = self.model.latitudes
            self.longitudes = self.model.longitudes
            self.lat_lon_grid = self.model.lat_lon_grid
            self.is_distributed = self.model.is_distributed
            self.expect_partitioned_input = self.model.expect_partitioned_input

        # distributed data parallel for multi-node training
        if dist.world_size > 1:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[dist.local_rank],
                output_device=dist.device,
                broadcast_buffers=dist.broadcast_buffers,
                find_unused_parameters=dist.find_unused_parameters,
                gradient_as_bucket_view=True,
                static_graph=True,
            )
        rank_zero_logger.info(
            f"Model parameter count is {count_trainable_params(self.model)}"
        )

        # instantiate the training datapipe
        self.channels_list = [i for i in range(cfg.num_channels_climate)]
        self.datapipe = ERA5HDF5Datapipe(
            base_path=cfg.dataset_path,
            data_folder="train",
            num_steps=1,
            dist=dist,
        )
        rank_zero_logger.success(
            f"Loaded training datapipe of size {len(self.datapipe)}"
        )

        # enable train mode
        self.model.train()

        # get normalized area
        self.area = normalized_grid_cell_area(self.lat_lon_grid[:, :, 0], unit="deg")
        self.area = self.area.to(dtype=self.dtype).to(device=dist.device)

        # instantiate loss, optimizer, and scheduler
        self.criterion = GraphCastLossFunction(
            self.area,
            self.channels_list,
            cfg.dataset_metadata_path,
            cfg.time_diff_std_path,
            self.dtype,
            dist.device
        )
        self.optimizer = apex.optimizers.FusedAdam(
            self.model.parameters(),
            lr=cfg.lr,
            betas=(0.9, 0.95),
            adam_w_mode=True,
            weight_decay=0.1,
        )
        scheduler1 = LinearLR(
            self.optimizer,
            start_factor=1e-3,
            end_factor=1.0,
            total_iters=cfg.num_iters_step1,
        )
        scheduler2 = CosineAnnealingLR(
            self.optimizer, T_max=cfg.num_iters_step2, eta_min=0.0
        )
        scheduler3 = LambdaLR(
            self.optimizer, lr_lambda=lambda epoch: (cfg.lr_step3 / cfg.lr)
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[scheduler1, scheduler2, scheduler3],
            milestones=[cfg.num_iters_step1, cfg.num_iters_step1 + cfg.num_iters_step2],
        )
        self.scaler = GradScaler('cuda', enabled=self.enable_scaler)

        # load checkpoint
        if dist.world_size > 1:
            torch.distributed.barrier()
        self.iter_init = load_checkpoint(
            to_absolute_path(cfg.ckpt_path),
            models=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=dist.device,
        )

        # instantiate the validation
        self.validation = Validation(cfg, self.model, self.dtype, self.dist, rank_zero_logger)


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:

    # TODO: The following is added temporarily to study the model
    import random
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
    torch.backends.cudnn.benchmark = False  # Might slow down training but ensures reproducibility



    if cfg.cugraphops_encoder or cfg.cugraphops_processor or cfg.cugraphops_decoder:
        try:
            import pylibcugraphops
        except:
            raise ImportError(
                "pylibcugraphops is not installed. Refer the Dockerfile for instructions"
                + "on how to install this package."
            )

    # initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    # initialize loggers
    if dist.rank == 0:
        initialize_wandb(
            project="GraphCast",
            entity=cfg.wb_entity,
            name=HydraConfig.get().job.name,
            group="group",
            mode=cfg.wb_mode,
        )  # Wandb logger
    logger = PythonLogger("main")  # General python logger
    rank_zero_logger = RankZeroLoggingWrapper(logger, dist)  # Rank 0 logger
    rank_zero_logger.file_logging()

    # print ranks and devices
    logger.info(f"Rank: {dist.rank}, Device: {dist.device}")

    # initialize trainer
    trainer = GraphCastTrainer(cfg, dist, rank_zero_logger)
    start = time.time()
    rank_zero_logger.info("Training started...")
    loss_agg, iter, tagged_iter, num_rollout_steps = 0, trainer.iter_init + 1, 1, 1
    terminate_training, finetune, update_dataloader = False, False, False

    with torch.autograd.profiler.emit_nvtx() if cfg.profile else nullcontext():
        # training loop
        while True:
            assert (
                iter < cfg.num_iters_step1 + cfg.num_iters_step2 + cfg.num_iters_step3
            ), "Training is already finished!"
            for _, data in enumerate(trainer.datapipe):
                # profiling
                if cfg.profile and iter == cfg.profile_range[0]:
                    rank_zero_logger.info("Starting profile", "green")
                    profiler.start()
                if cfg.profile and iter == cfg.profile_range[1]:
                    rank_zero_logger.info("Ending profile", "green")
                    profiler.stop()
                torch.cuda.nvtx.range_push(f"Training iteration {iter}")

                if iter >= cfg.num_iters_step1 + cfg.num_iters_step2 and not finetune:
                    finetune = True
                    if cfg.force_single_checkpoint_finetune:
                        if hasattr(trainer.model, "module"):
                            trainer.model.module.set_checkpoint_model(True)
                        else:
                            trainer.model.set_checkpoint_model(True)
                    if cfg.checkpoint_encoder_finetune:
                        if hasattr(trainer.model, "module"):
                            trainer.model.module.set_checkpoint_encoder(True)
                        else:
                            trainer.model.set_checkpoint_encoder(True)
                    if cfg.checkpoint_processor_finetune:
                        if hasattr(trainer.model, "module"):
                            trainer.model.module.set_checkpoint_processor(cfg.segments)
                        else:
                            trainer.model.set_checkpoint_encoder(True)
                    if cfg.checkpoint_decoder_finetune:
                        if hasattr(trainer.model, "module"):
                            trainer.model.module.set_checkpoint_decoder(True)
                        else:
                            trainer.model.set_checkpoint_encoder(True)
                if (
                    finetune
                    and (iter - (cfg.num_iters_step1 + cfg.num_iters_step2))
                    % cfg.step_change_freq
                    == 0
                    and iter != tagged_iter
                ):
                    update_dataloader = True
                    tagged_iter = iter

                # update the dataloader for finetuning
                if update_dataloader:
                    num_rollout_steps = (
                        iter - (cfg.num_iters_step1 + cfg.num_iters_step2)
                    ) // cfg.step_change_freq + 2
                    trainer.datapipe = ERA5HDF5Datapipe(
                        base_path=cfg.dataset_path,
                        data_folder="train",
                        num_steps=num_rollout_steps,
                        dist=dist,
                    )
                    update_dataloader = False
                    rank_zero_logger.info(
                        f"Switching to {num_rollout_steps}-step rollout!"
                    )
                    break

                # Prepare the input & output
                input = data[0]["input"].to(dtype=trainer.dtype)
                output = data[0]["output"].to(dtype=trainer.dtype)

                # TODO: Control
                # from datetime import datetime
                # input_timestamp = datetime.fromtimestamp(int(data[0]["input_timestamps"].item()))
                # output_timestamp = datetime.fromtimestamp(int(data[0]["output_timestamps"][-1].item()))
                # print(f"Training > Rank {dist.rank} - Input timestamp: {input_timestamp} - Last output timestamp: {output_timestamp}")

                # training step
                loss = trainer.train(input, output)
                torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
                loss /= dist.world_size

                if dist.rank == 0:
                    loss_agg += loss.detach().cpu()

                # validation
                if iter % cfg.val_freq == 0:
                    # free up GPU memory
                    del input, output
                    torch.cuda.empty_cache()
                    # Generate plot at the first validation step and every val_images_freq steps
                    generate_plots = dist.rank == 0 and (iter == 5 or iter % cfg.val_images_freq == 0)
                    mse = trainer.validation.step(
                        channels=list(np.arange(cfg.num_channels_val)), 
                        iter=iter,
                        generate_plots=generate_plots 
                    )
                    if dist.rank == 0:
                        logger.log(f"iteration {iter}, Validation MSE: {mse:.04f}")
                        wandb.log({"Validation MSE": mse,}, step=iter)
                # distributed barrier
                if dist.world_size > 1:
                    torch.distributed.barrier()

                # print logs and save checkpoint
                if dist.rank == 0 and iter % cfg.save_freq == 0:
                    # save_checkpoint(
                    #     to_absolute_path(cfg.ckpt_path),
                    #     models=trainer.model,
                    #     optimizer=trainer.optimizer,
                    #     scheduler=trainer.scheduler,
                    #     scaler=trainer.scaler,
                    #     epoch=iter,
                    # )
                    # logger.info(f"Saved model on rank {dist.rank}")
                    loss_all = loss_agg / cfg.save_freq
                    iter_time = (time.time()-start)/cfg.save_freq
                    logger.log(
                        f"iteration: {iter}, loss: {loss_all:10.3e}, \
                            time per iter: {iter_time:10.3e}"
                    )
                    if dist.rank == 0:
                        wandb.log(
                            {
                                "loss": loss_all,
                                "learning_rate": trainer.scheduler.get_last_lr()[0],
                                "time_per_iter": iter_time,
                            },
                            step=iter,
                        )
                    loss_agg = 0
                    start = time.time()
                iter += 1

                torch.cuda.nvtx.range_pop()

                # wrap up & terminate if training is finished
                if (
                    iter
                    >= cfg.num_iters_step1 + cfg.num_iters_step2 + cfg.num_iters_step3
                ):
                    if dist.rank == 0:
                        del data_x, y
                        torch.cuda.empty_cache()
                        error = trainer.validation.step(
                            channels=list(np.arange(cfg.num_channels_val)), iter=iter
                        )
                        logger.log(f"iteration {iter}, Validation MSE: {error:.04f}")

                        save_checkpoint(
                            to_absolute_path(cfg.ckpt_path),
                            trainer.model,
                            trainer.optimizer,
                            trainer.scheduler,
                            trainer.scaler,
                            iter,
                        )
                        logger.info(f"Saved model on rank {dist.rank}")
                        logger.log(
                            f"iteration: {iter}, loss: {loss_agg/cfg.save_freq:10.3e}, \
                                time per iter: {(time.time()-start)/cfg.save_freq:10.3e}"
                        )
                    terminate_training = True
                    break
            if terminate_training:
                rank_zero_logger.info("Finished training!")
                break


if __name__ == "__main__":
    main()
