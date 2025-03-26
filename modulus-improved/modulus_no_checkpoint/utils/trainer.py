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
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, LambdaLR

from modulus.models.graph_cast_net import GraphCastNet
from modulus.models.utils.loss import GraphCastLossFunction
from modulus.utils.logging import Logger
from modulus.utils.distributed_manager import DistributedManager as DM
from modulus.datapipes.era5_hdf5 import ERA5HDF5Datapipe

class GraphCastTrainer:
    def __init__(
        self, 
        cfg: DictConfig
    ):
        self.cfg = cfg
        self.logger = Logger("trainer")

        # Set seed
        self.set_seed(seed=self.cfg.datapipe.seed)

        self.model = self.instantiate_model()
        self.creterion = GraphCastLossFunction(
            cfg=self.cfg,
            area=self.model.module.area,
            channels_metadata=self.model.module.metadata
        )
        self.optimizer = self.instantiate_optimizer()
        self.scheduler = self.instantiate_scheduler()
        (
            self._current_iteration,
            self._current_rollout_steps,
            _iterators
        ) = self.load_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler
        )

        # Initialize datapipes
        self.datapipe_training = self.instantiate_datapipe(
            type="train",
            num_output_steps=self.current_rollout_steps,
            iterator=_iterators["train"]
        )
        self.datapipe_testing = self.instantiate_datapipe(
            type="test",
            num_output_steps=self.cfg.testing.rollout_steps,
            iterator=_iterators["test"]
        )

    def set_seed(self, seed):
        if seed is None:
            return
        
        self.logger.info(f"Setting seed to {seed}")

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def instantiate_model(self):
        model = GraphCastNet(cfg=self.cfg, device=DM.device())

        parameters_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Model created. Trainable parameters count is {parameters_count:,}".replace(",", "'"))

        #model = model.to(dtype=dtype).to(device=self.dist.device)

        if DM.is_rank_zero() and self.cfg.wb.watch_model:
            wandb.watch(model)

        # distributed data parallel for multi-node training
        model = DistributedDataParallel(
            model,
            device_ids=[DM.local_rank()],
            output_device=DM.device(),
            broadcast_buffers=self.cfg.model.ddp.broadcast_buffers,
            find_unused_parameters=self.cfg.model.ddp.find_unused_parameters,
            gradient_as_bucket_view=self.cfg.model.ddp.gradient_as_bucket_view,
            static_graph=self.cfg.model.ddp.static_graph
        )
        model.train()
        
        return model

    def instantiate_optimizer(self):
        return apex.optimizers.FusedAdam(
            self.model.parameters(),
            betas=(0.9, 0.95),
            adam_w_mode=True,
            weight_decay=0.1,
        )

    def instantiate_scheduler(self):
        return SequentialLR(
            self.optimizer,
            schedulers=[
                LinearLR(
                    self.optimizer,
                    start_factor=self.cfg.schedule.phase1.lr_start,
                    end_factor=self.cfg.schedule.phase1.lr_end,
                    total_iters=self.cfg.schedule.phase1.iterations,
                ),
                CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.cfg.schedule.phase2.iterations,
                    eta_min=0,
                ),
                LambdaLR(
                    self.optimizer,
                    lr_lambda=lambda epoch: self.cfg.schedule.phase3.lr
                ),
            ],
            milestones=[
                self.cfg.schedule.phase1.iterations,
                self.cfg.schedule.phase1.iterations + self.cfg.schedule.phase2.iterations,
            ],
        )

    def instantiate_datapipe(
        self,
        type: str,
        num_output_steps: int,
        iterator: dict
    ):
        if type == "train":
            prefetch_queue_depth = self.cfg.datapipe.prefetch_queue_depth
            num_threads = self.cfg.datapipe.num_threads
        else:
            prefetch_queue_depth = self.cfg.testing.samples_per_rank
            num_threads = 1

        datapipe = ERA5HDF5Datapipe(
            cfg=self.cfg,
            dataset_folder=type,
            num_output_steps=num_output_steps,
            model=self.model.module,
            seed=self.cfg.datapipe.seed + num_output_steps,
            iterator=iterator,
            prefetch_queue_depth=prefetch_queue_depth,
            num_threads=num_threads,
            
            shuffle=True,
            device=DM.device(),
            rank=DM.rank(),
            world_size=DM.world_size()
        )

        self.logger.success(f"Loaded {type} datapipe of size {len(datapipe):,} samples".replace(",", "'"))

        return datapipe

    def train(self, sample):
        self.optimizer.zero_grad()

        _channels_dataset = self.model.module.input_channels_count_dataset()
        _channels_generated = self.model.module.input_channels_count_generated()

        data = sample["data"]
        reanalysis, generated = torch.split(data, [_channels_dataset, _channels_generated], dim=1)
        forecasts = torch.empty_like(reanalysis[1:])

        steps_without_grad = max(self.current_rollout_steps - 5, 0)
        for step_i in range(self.current_rollout_steps):
            model_input = torch.empty_like(data[0])
            if step_i == 0:
                model_input.copy_(data[0], non_blocking=True)
            else:
                model_input[:_channels_dataset].copy_(forecasts[step_i-1], non_blocking=True)
                model_input[_channels_dataset:].copy_(generated[step_i], non_blocking=True)
            
            # This is necessary to avoid going out of memory
            if step_i < steps_without_grad:
                with torch.no_grad():
                    # [21, 721, 1440] <- [31, 721, 1440]
                    output = self.model(model_input)
            else:
                output = self.model(model_input)

            forecasts[step_i] = output
        
        loss = self.creterion(reanalysis[steps_without_grad+1:], forecasts[steps_without_grad:])
        loss.backward()

        self.optimizer.step()
        self.scheduler.step()

        return loss, sample["global_sample_id"]

    def test(self, sample):
        _channels_dataset = self.model.module.input_channels_count_dataset()
        _channels_generated = self.model.module.input_channels_count_generated()

        data = sample["data"]
        reanalysis, generated = torch.split(data, [_channels_dataset, _channels_generated], dim=1)
        forecasts = torch.empty_like(reanalysis[1:])
        model_input = torch.empty_like(data[0])
        for step_i in range(self.cfg.testing.rollout_steps):
            if step_i == 0:
                model_input.copy_(data[0], non_blocking=True)
            else:
                model_input[:_channels_dataset].copy_(forecasts[step_i-1], non_blocking=True)
                model_input[_channels_dataset:].copy_(generated[step_i], non_blocking=True)
            
            with torch.no_grad():
                output = self.model(model_input)
            
            forecasts[step_i] = output

        mse = torch.mean(torch.square(reanalysis[1:] - forecasts))
        return mse, sample["global_sample_id"]

    def log_step(self, type, metric, global_sample_id, timers):
        # # Get average among all ranks
        # torch.distributed.all_reduce(metric, op=torch.distributed.ReduceOp.SUM)
        # metric /= DM.world_size()
        # metric = metric.cpu().item()

        if not DM.is_rank_zero():
            return

        gb = torch.cuda.memory_reserved() / 1024**3
        if type == "train":
            payload = {
                "learning_rate": self.scheduler.get_last_lr()[0],

                "training_loss": metric,
                "training_time": timers["training"].elapsed,
                "training_time_dataloader": timers["dataloader"].elapsed,
                "training_time_model": timers["model"].elapsed,
            }
            self.logger.log(f"Iteration {self.current_iteration:5d} | Train loss: {metric:6.4f} | Time taken: {timers['dataloader'].elapsed:5.2f}/{timers['model'].elapsed:5.2f}/{timers['training'].elapsed:5.2f} sec | GPU memory: {gb:4.1f} GB | Global sample ID: {global_sample_id}")
        else:
            payload = {
                "testing_mse": metric,
                "testing_time": timers["testing"].elapsed,
                "testing_time_dataloader": timers["dataloader"].elapsed,
                "testing_time_model": timers["model"].elapsed,
            }
            self.logger.info(f"Iteration {self.current_iteration:5d} | Test MSE: {metric:8.4f} | Time taken: {timers['dataloader'].elapsed:5.2f}/{timers['model'].elapsed:5.2f}/{timers['testing'].elapsed:5.2f} sec | GPU memory: {gb:4.1f} GB | Global sample ID: {global_sample_id}")

        wandb.log(data=payload, step=self.current_iteration)           

    def start_iteration(self):
        self._current_iteration += 1

    def increase_rollout(self):
        self._current_rollout_steps += 1

        self.logger.info(f"Switched to {self.current_rollout_steps}-long rollouts.")

        self.datapipe_training = self.instantiate_datapipe(
            type="train",
            num_output_steps=self.current_rollout_steps,
            iterator={
                "initial_epoch_idx": 0,
                "initial_sample_idx": 0
            }
        )
    
    @property
    def current_iteration(self):
        if not hasattr(self, "_current_iteration"):
            return 1
        return self._current_iteration

    @property
    def current_rollout_steps(self):
        if not hasattr(self, "_current_rollout_steps"):
            return 1
        return self._current_rollout_steps

    @property
    def do_testing(self):
        return self.current_iteration % self.cfg.testing.frequency == 0

    @property
    def is_phase3(self):
        return self.cfg.schedule.phase3.iterations > 0 and \
            self.current_iteration >= (self.cfg.schedule.phase1.iterations + self.cfg.schedule.phase2.iterations)

    @property
    def do_rollout_increase(self):
        if self.cfg.schedule.phase3.iterations == 0:
            return False
        _iter_in_phase3 = self.current_iteration - (self.cfg.schedule.phase1.iterations + self.cfg.schedule.phase2.iterations)
        _iters_per_increase = self.cfg.schedule.phase3.iterations // self.cfg.schedule.phase3.rollout_steps_increments
        return self.is_phase3 and _iter_in_phase3 % _iters_per_increase == 0

    @property
    def do_checkpoint(self):
        return DM.is_rank_zero() and \
            self.cfg.checkpoint.enabled and \
            self.current_iteration % self.cfg.checkpoint.frequency == 0

    @property
    def do_terminate(self):
        _max_iter = self.cfg.schedule.phase1.iterations + self.cfg.schedule.phase2.iterations + self.cfg.schedule.phase3.iterations
        return self.current_iteration >= _max_iter

    def _get_checkpoint_filename(
        self,
        folder: Path,
        saving: bool,
        iter: int = None,
    ):
        assert isinstance(folder, Path) 
        if iter is None:
            files = [f.name for f in folder.glob(f"{self.cfg.checkpoint.names}.iter*.pth")]

            if len(files) == 0:
                iter = 0
            else:
                max_iter = max([
                    int(match.group(1))
                    for file in files if (match := re.search(r"\.iter(\d+)\.", file))
                ])
                iter = max_iter + 1 if saving else max_iter
        return folder / f"{self.cfg.checkpoint.names}.iter{iter:06d}.pth"

    def save_checkpoint(
        self,
        iterators
    ):
        _folder = Path(self.cfg.checkpoint.folder).absolute()
        _folder.mkdir(parents=True, exist_ok=True)
        _filepath = self._get_checkpoint_filename(folder=_folder, saving=True, iter=self.current_iteration)

        cp_dict = {
            "cfg": OmegaConf.to_container(self.cfg),
            "current_iteration": self.current_iteration,
            "current_rollout_steps": self.current_rollout_steps,

            "model": self.model.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),

            "iterators": iterators
        }

        torch.save(cp_dict, _filepath)
        self.logger.success(f"Checkpoint saved to: " + str(_filepath))

    def load_checkpoint(
        self,
        model,
        optimizer,
        scheduler,
    ):
        _folder = Path(self.cfg.checkpoint.folder).absolute()
        _filepath = self._get_checkpoint_filename(folder=_folder, saving=False)

        _default_return = (
            0, # current_iteration
            1, # current_rollout_steps
            {
                "train": {
                    "initial_epoch_idx": 0,
                    "initial_sample_idx": 0
                },
                "test": {
                    "initial_epoch_idx": 0,
                    "initial_sample_idx": 0
                }
            }
        )

        if not self.cfg.checkpoint.enabled:
            return _default_return
            
        if not _filepath.is_file():
            self.logger.info(f"No checkpoint found at: " + str(_filepath))
            return _default_return
        
        cp_dict = torch.load(
            _filepath, 
            map_location=DM.device(),
            weights_only=True
        )

        # Check configurations
        if OmegaConf.to_container(self.cfg) != cp_dict["cfg"]:
            print("Configuration mismatch between checkpoint and current run")
            print("- Current run configuration: ", self.cfg)
            print("- Checkpoint configuration: ", cp_dict["cfg"])
            raise ValueError("Configuration mismatch between checkpoint and current run")

        model.module.load_state_dict(cp_dict["model"])        
        optimizer.load_state_dict(cp_dict["optimizer"])
        scheduler.load_state_dict(cp_dict["scheduler"])

        # Fix: optimizer state to be restored with float32 state, not bfloat16!
        for op_state, cp_state in zip(optimizer.state.values(), cp_dict["optimizer"]["state"].values()):
            for (op_k, op_v), (cp_k, cp_v) in zip(op_state.items(), cp_state.items()):
                op_state[op_k] = cp_v.to(torch.float32)

        self.logger.success(f"Loaded checkpoint from: {str(_filepath)} (Current rollout steps: {cp_dict['current_rollout_steps']})")

        return(
            cp_dict["current_iteration"],
            cp_dict["current_rollout_steps"],
            cp_dict["iterators"]
        )
