# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2025 schups@gmail.com (Modifications)
# SPDX-License-Identifier: Apache-2.0

import torch
import wandb
import apex
import glob
import re
import time
import random
import numpy as np

from hydra.utils import to_absolute_path
from pathlib import Path
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, LambdaLR

from modulus.models_baseline.graphcast.graph_cast_net import GraphCastNetBaseline
from modulus.launch.logging import RankZeroLoggingWrapper
from modulus.launch.utils import load_checkpoint
from modulus.distributed import DistributedManager
from modulus.datapipes.climate.era5_hdf5 import ERA5HDF5Datapipe

class GraphCastTrainer():
    def __init__(
        self, 
        cfg: DictConfig,
        dist: DistributedManager,
        zlogger: RankZeroLoggingWrapper
    ):
        super().__init__()

        self.cfg = cfg
        self.dist = dist
        self.zlogger = zlogger
        self.dtype = torch.bfloat16 if self.cfg.dtype == "bfloat16" else torch.float32

        if self.cfg.seed is not None:
            self.zlogger.info(f"Setting seed to {self.cfg.seed}")
            self.set_seed(self.cfg.seed)

        self.model = self.instantiate_model()
        self.optimizer = self.instantiate_optimizer()
        self.scheduler = self.instantiate_scheduler()

        (
            self._iter,
            self._current_rollout_steps,
            self.iterator_seed,
            self.training_iterator_offset_epoch_idx,
            self.training_iterator_offset_sample_idx,
            self.testing_iterator_offset_epoch_idx,
            self.testing_iterator_offset_sample_idx
        ) = self.load_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler
        )

        if isinstance(self.model, DistributedDataParallel):
            self.model.module.update_checkpoint_for_rollout_step(rollout_steps=self._current_rollout_steps)
        else:
            self.model.update_checkpoint_for_rollout_step(rollout_steps=self._current_rollout_steps)

        # Initialize datapipe
        self.training_datapipe = self.instantiate_training_datapipe(
            current_rollout_steps=self.current_rollout_steps,
            iterator_seed=self.iterator_seed,
            iterator_offset_epoch_idx=self.training_iterator_offset_epoch_idx,
            iterator_offset_sample_idx=self.training_iterator_offset_sample_idx
        )
        self.testing_datapipe = self.instantiate_testing_datapipe(
            iterator_seed=self.iterator_seed,
            iterator_offset_epoch_idx=self.testing_iterator_offset_epoch_idx,
            iterator_offset_sample_idx=self.testing_iterator_offset_sample_idx
        )

    def set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def instantiate_model(self):
        model = GraphCastNetBaseline(                    # Typical values
            sample_height=self.cfg.sample.height,        # 721
            sample_width=self.cfg.sample.width,          # 1440
            sample_channels=self.cfg.sample.channels,    # 21
            
            include_static_data=self.cfg.include.static_data,         # True
            include_spatial_info=self.cfg.include.spatial_info,       # True
            include_temporal_info=self.cfg.include.temporal_info,     # True
            include_solar_radiation=self.cfg.include.solar_radiation, # True

            batch_size=self.cfg.datapipe.batch_size,      # 1
            mesh_level=self.cfg.mesh_level,               # 6
            activation_fn=self.cfg.activation_fn,         # "silu",
            hidden_dim=self.cfg.hidden_dim,               # 512
            hidden_layers=self.cfg.hidden_layers,         # 1
            aggregation_op=self.cfg.aggregation_op,       # "sum"
            processor_layers=self.cfg.processor_layers,   # 16

            cache_enabled=self.cfg.cache_enabled,         # True
        )
        
        parameters_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.zlogger.info(f"Model parameter count is {parameters_count:,}".replace(",", "'"))
        model = model.to(dtype=self.dtype).to(device=self.dist.device)

        if self.dist.rank == 0 and self.cfg.wb_watch_model:
            wandb.watch(model)

        # distributed data parallel for multi-node training
        if self.dist.world_size > 1:
            model = DistributedDataParallel(
                model,
                device_ids=[self.dist.local_rank],
                output_device=self.dist.device,
                broadcast_buffers=self.dist.broadcast_buffers,
                find_unused_parameters=self.dist.find_unused_parameters,
                gradient_as_bucket_view=True,
                static_graph=True,
            )
        model.train()
        return model

    def instantiate_scheduler(self):
        return SequentialLR(
            self.optimizer,
            schedulers=[
                LinearLR(
                    self.optimizer,
                    start_factor=1e-3,
                    end_factor=1.0,
                    total_iters=self.cfg.phase1_iters,
                ),
                CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.cfg.phase2_iters,
                    eta_min=0,
                ),
                LambdaLR(
                    self.optimizer,
                    lr_lambda=lambda epoch: (self.cfg.phase3_lr / self.cfg.lr)
                ),
            ],
            milestones=[
                self.cfg.phase1_iters,
                self.cfg.phase1_iters + self.cfg.phase2_iters
            ],
        )

    def instantiate_optimizer(self):
        return apex.optimizers.FusedAdam(
            self.model.parameters(),
            lr=self.cfg.lr,
            betas=(0.9, 0.95),
            adam_w_mode=True,
            weight_decay=0.1,
        )

    def _instantiate_datapipe(
        self, 
        type: str, 
        num_output_steps: int,
        num_threads: int, 
        prefetch_queue_depth: int,
        iterator_seed: int,
        iterator_offset_epoch_idx: int,
        iterator_offset_sample_idx: int,
        parallel: bool
    ):
        assert type in ["train", "test", "out_of_sample"]
        datapipe = ERA5HDF5Datapipe(
            model=self.model.module if isinstance(self.model, DistributedDataParallel) else self.model,
            dataset_base_path=self.cfg.dataset.base_path,
            dataset_folder=type,
            dataset_samples_per_file=self.cfg.dataset.samples_per_file,

            num_output_steps=num_output_steps,

            iterator_seed=iterator_seed,
            iterator_offset_epoch_idx=iterator_offset_epoch_idx,
            iterator_offset_sample_idx=iterator_offset_sample_idx,

            num_threads=num_threads,
            prefetch_queue_depth=prefetch_queue_depth,
            parallel=parallel
        )
        self.zlogger.success(f"Loaded {type} datapipe of size {len(datapipe):,} samples".replace(",", "'"))
        return datapipe

    def instantiate_training_datapipe(
        self,
        current_rollout_steps: int,
        iterator_seed: int,
        iterator_offset_epoch_idx: int,
        iterator_offset_sample_idx: int
    ):
        return self._instantiate_datapipe(
            type="train",
            num_output_steps=current_rollout_steps,
            num_threads=self.cfg.datapipe.num_threads,
            prefetch_queue_depth=self.cfg.datapipe.prefetch_queue_depth,
            iterator_seed=iterator_seed + current_rollout_steps,
            iterator_offset_epoch_idx=iterator_offset_epoch_idx,
            iterator_offset_sample_idx=iterator_offset_sample_idx,
            parallel=True
        )

    def instantiate_testing_datapipe(
        self, 
        iterator_seed: int,
        iterator_offset_epoch_idx: int,
        iterator_offset_sample_idx: int
    ):
        return self._instantiate_datapipe(
            type="test",
            num_output_steps=self.cfg.num_testing_steps,
            num_threads=self.cfg.num_testing_samples_per_rank,
            prefetch_queue_depth=self.cfg.num_testing_samples_per_rank,
            iterator_seed=iterator_seed,
            iterator_offset_epoch_idx=iterator_offset_epoch_idx,
            iterator_offset_sample_idx=iterator_offset_sample_idx,
            parallel=True
        )

    def train(self, sample):
        data = sample[0]["data"].to(dtype=self.dtype)

        self.optimizer.zero_grad()

        reanalysis, generated = torch.split(data, [
            self.training_datapipe.channels_count,
            self.training_datapipe.generated_channels_count
        ], dim=2)

        forecasts = torch.empty_like(reanalysis[:, 1:])
        steps_without_grad = max(self.current_rollout_steps - 5, 0)
        for step_i in range(self.current_rollout_steps):
            model_input = torch.empty_like(data[0, 0])
            if step_i == 0:
                model_input.copy_(data[0, 0], non_blocking=True)
            else:
                model_input[:self.training_datapipe.channels_count].copy_(forecasts[0, step_i-1], non_blocking=True)
                model_input[self.training_datapipe.channels_count:].copy_(generated[0, step_i], non_blocking=True)
            
            # This is necessary to avoid going out of memory
            if step_i < steps_without_grad:
                with torch.no_grad():
                    # [21, 721, 1440] <- [31, 721, 1440]
                    output = self.model(model_input)
            else:
                output = self.model(model_input)

            forecasts[0, step_i] = output

        loss = torch.mean(torch.square(reanalysis[0, steps_without_grad+1:] - forecasts[0, steps_without_grad:]))
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss, sample[0]["global_sample_id"].item()

    def test(self, sample):
        data = sample[0]["data"].to(dtype=self.dtype)

        reanalysis, generated = torch.split(data, [
            self.testing_datapipe.channels_count,
            self.testing_datapipe.generated_channels_count
        ], dim=2)

        forecasts = torch.empty_like(reanalysis[:, 1:])
        model_input = torch.empty_like(data[0, 0])
        for step_i in range(self.cfg.num_testing_steps):
            if step_i == 0:
                model_input.copy_(data[0, 0], non_blocking=True)
            else:
                model_input[:self.testing_datapipe.channels_count].copy_(forecasts[0, step_i-1], non_blocking=True)
                model_input[self.testing_datapipe.channels_count:].copy_(generated[0, step_i], non_blocking=True)

            with torch.no_grad():
                output = self.model(model_input)
            forecasts[0, step_i] = output

        mse = torch.mean(torch.square(reanalysis[0, 1:] - forecasts[0]))
        return mse, sample[0]["global_sample_id"].item()

    def log_train_step(self, loss, time_start, time_dataloader, time_model, global_sample_id):
        # # Get average among all ranks
        # torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
        # loss /= self.dist.world_size
        # loss = loss.cpu().item()

        # Log info
        if self.dist.rank == 0:
            gb = torch.cuda.memory_reserved() / 1024**3
            time_total = time.perf_counter() - time_start
            self.zlogger.log(f"Iteration {self.iter:5d} | Train loss: {loss:2.3f} | Time taken: {time_total:6.3f} sec | GPU memory: {gb:4.1f} GB | Global sample ID: {global_sample_id}")

            wandb.log({
                "loss": loss,
                "learning_rate": self.scheduler.get_last_lr()[0],
                "time_training_total": time_total,
                "time_training_dataloader": time_dataloader,
                "time_training_model": time_model,
                "global_sample_id": global_sample_id
            }, step=self.iter)

    def log_test_step(self, mse, time_start, time_dataloader, time_model, global_sample_id):
        # Get average among all ranks
        # torch.distributed.all_reduce(mse, op=torch.distributed.ReduceOp.SUM)
        # mse = mse / self.dist.world_size
        # mse = mse.cpu().item()

        # Log info
        if self.dist.rank == 0:
            gb = torch.cuda.memory_reserved() / 1024**3
            time_total = time.perf_counter() - time_start
            self.zlogger.info(f"Iteration {self.iter:5d} | Test MSE: {mse:7.3f} | Time taken: {time_total:6.3f} sec | GPU memory: {gb:4.1f} GB | Global sample ID: {global_sample_id}")
        
            wandb.log({
                "test_mse": mse,
                "time_testing_total": time_total,
                "time_testing_dataloader": time_dataloader,
                "time_testing_model": time_model,
                "global_sample_id": global_sample_id
            }, step=self.iter)

    def iter_start(self):
        self._iter += 1

    def increase_rollout(self):
        self._current_rollout_steps += 1
        self.training_datapipe = self.instantiate_training_datapipe(
            current_rollout_steps=self._current_rollout_steps,
            iterator_seed=self.iterator_seed,
            iterator_offset_epoch_idx=0,
            iterator_offset_sample_idx=0
        )
        if isinstance(self.model, DistributedDataParallel):
            self.model.module.update_checkpoint_for_rollout_step(rollout_steps=self._current_rollout_steps)
        else:
            self.model.update_checkpoint_for_rollout_step(rollout_steps=self._current_rollout_steps)
 
    @property
    def iter(self):
        return self._iter

    @property
    def current_rollout_steps(self):
        return self._current_rollout_steps

    @property
    def do_testing(self):
        return self.iter % self.cfg.testing_frequency == 0

    @property
    def is_phase3(self):
        return self.cfg.phase3_iters > 0 and self.iter >= self.cfg.phase1_iters + self.cfg.phase2_iters

    @property
    def do_rollout_increase(self):
        if self.cfg.phase3_iters == 0:
            return False
        iter_in_phase3 = self.iter - (self.cfg.phase1_iters + self.cfg.phase2_iters)
        iters_per_increase = self.cfg.phase3_iters // self.cfg.phase3_increments
        return self.is_phase3 and iter_in_phase3 % iters_per_increase == 0

    @property
    def do_checkpoint(self):
        return self.dist.rank == 0 and \
            self.cfg.checkpoint_enabled and \
            self.iter % self.cfg.checkpoint_frequency == 0

    @property
    def do_terminate(self):
        return self.iter >= self.cfg.phase1_iters + self.cfg.phase2_iters + self.cfg.phase3_iters

    def _get_checkpoint_filename(
        self,
        folder: Path,
        saving: bool,
        iter: int = None,
    ):
        assert isinstance(folder, Path) 
        if iter is None:
            files = [f.name for f in folder.glob(f"{self.cfg.checkpoint_names}.iter*.pth")]

            if len(files) == 0:
                iter = 0
            else:
                max_iter = max([
                    int(match.group(1))
                    for file in files if (match := re.search(r"\.iter(\d+)\.", file))
                ])
                iter = max_iter + 1 if saving else max_iter
        return folder / f"{self.cfg.checkpoint_names}.iter{iter:06d}.pth"

    def save_checkpoint(
        self,
        training_datapipe_epoch_idx,
        training_datapipe_sample_in_epoch,
        testing_datapipe_epoch_idx,
        testing_datapipe_sample_in_epoch
    ):
        folder = Path(to_absolute_path(self.cfg.checkpoint_folder))
        folder.mkdir(parents=True, exist_ok=True)
        filepath = self._get_checkpoint_filename(
            folder=folder,
            saving=True,
            iter=self.iter
        )

        cp_dict = {
            "metadata": self.model.module.args,
            "iter": self.iter,
            "current_rollout_steps": self.current_rollout_steps,
            
            "iterator_seed": self.iterator_seed,
            "training_datapipe_epoch_idx": training_datapipe_epoch_idx,
            "training_datapipe_sample_in_epoch": training_datapipe_sample_in_epoch,
            "testing_datapipe_epoch_idx": testing_datapipe_epoch_idx,
            "testing_datapipe_sample_in_epoch": testing_datapipe_sample_in_epoch,

            "model": self.model.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }

        torch.save(cp_dict, filepath)
        self.zlogger.success(f"Checkpoint saved to: " + str(filepath))

    def load_checkpoint(
        self,
        model,
        optimizer,
        scheduler,
    ):
        folder = Path(to_absolute_path(self.cfg.checkpoint_folder))

        filepath = self._get_checkpoint_filename(
            folder=Path(to_absolute_path(self.cfg.checkpoint_folder)),
            saving=False
        )
        
        default_return = (
            0, # iter
            1, # current_rollout_steps
            self.cfg.seed if self.cfg.seed is not None else int(os.getenv("SLURM_JOB_ID", 0)), # iterator_seed,
            0, # training_iterator_offset_epoch_idx
            0, # training_iterator_offset_sample_idx
            0, # testing_iterator_offset_epoch_idx
            0, # testing_iterator_offset_sample_idx
        )

        if not self.cfg.checkpoint_enabled:
            return default_return
            
        if not Path(filepath).is_file():
            self.zlogger.info(f"No checkpoint found at: " + str(filepath))
            return default_return
        
        cp_dict = torch.load(
            filepath, 
            map_location=self.dist.device,
            weights_only=True
        )

        if isinstance(self.model, DistributedDataParallel):
            model.module.check_args(cp_dict["metadata"]) # raises ValueError if metadata is inconsistent
            model.module.load_state_dict(cp_dict["model"])
        else:
            model.check_args(cp_dict["metadata"])
            model.load_state_dict(cp_dict["model"])
        
        optimizer.load_state_dict(cp_dict["optimizer"])
        scheduler.load_state_dict(cp_dict["scheduler"])

        # Fix: optimizer state to be restored with float32 state, not bfloat16!
        for op_state, cp_state in zip(optimizer.state.values(), cp_dict["optimizer"]["state"].values()):
            for (op_k, op_v), (cp_k, cp_v) in zip(op_state.items(), cp_state.items()):
                op_state[op_k] = cp_v.to(torch.float32)

        self.zlogger.success(f"Loaded checkpoint from: {str(filepath)} (Current rollout steps: {cp_dict['current_rollout_steps']})")

        return (
            cp_dict["iter"],
            cp_dict["current_rollout_steps"],
            cp_dict["iterator_seed"],
            cp_dict["training_datapipe_epoch_idx"],
            cp_dict["training_datapipe_sample_in_epoch"] + 1,
            cp_dict["testing_datapipe_epoch_idx"],
            cp_dict["testing_datapipe_sample_in_epoch"] + 1
        )