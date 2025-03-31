# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2025 schups@gmail.com (Modifications)
# SPDX-License-Identifier: Apache-2.0

import torch
import wandb
import apex
import glob
import re
import random
import numpy as np

from hydra.utils import to_absolute_path
from pathlib import Path
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, LambdaLR

from modulus.models_baseline.graphcast.graph_cast_net import GraphCastNetBaseline
from modulus.utils.graphcast.loss import GraphCastLossFunction
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
        self.criterion = GraphCastLossFunction()
        self.optimizer = self.instantiate_optimizer()
        self.scheduler = self.instantiate_scheduler()

        (
            self._iter,
            self._current_rollout_steps,
            self.iterator_seed,
            self.iterator_offset_epoch_idx,
            self.iterator_offset_sample_idx
        ) = self.load_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler
        )

        # FIXME: Remove
        self._current_rollout_steps = 20

        # Initialize datapipe
        self.training_datapipe = self.instantiate_training_datapipe(
            current_rollout_steps=self.current_rollout_steps,
            iterator_seed=self.iterator_seed,
            iterator_offset_epoch_idx=self.iterator_offset_epoch_idx,
            iterator_offset_sample_idx=self.iterator_offset_sample_idx
        )
        self.testing_datapipe = self.instantiate_testing_datapipe(
            iterator_seed=self.iterator_seed
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
        model = GraphCastNetBaseline(                         # Typical values
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
        )
        
        self.zlogger.info(f"Model parameter count is {self.count_trainable_params()}")
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
        import math

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
        iterator_offset_sample_idx: int
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
            prefetch_queue_depth=prefetch_queue_depth
        )
        self.zlogger.success(f"Loaded {type} datapipe of size {len(datapipe)}")
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
            iterator_seed=iterator_seed,
            iterator_offset_epoch_idx=iterator_offset_epoch_idx,
            iterator_offset_sample_idx=iterator_offset_sample_idx
        )

    def instantiate_testing_datapipe(
        self, 
        iterator_seed: int
    ):
        return self._instantiate_datapipe(
            type="test",
            num_output_steps=self.cfg.num_testing_steps,
            num_threads=self.cfg.num_testing_samples_per_rank,
            prefetch_queue_depth=self.cfg.num_testing_samples_per_rank,
            iterator_seed=iterator_seed,
            iterator_offset_epoch_idx=0,
            iterator_offset_sample_idx=0
        )

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def train(self, reanalysis, generated):
        """
        reanalysis shape: [1, 6, 21, 721, 1440] i.e. [batch_size, steps, num_channels, height, width]
        generated shape:  [1, 6, 10, 721, 1440] i.e. [batch_size, steps, num_channels, height, width]

        Step 0 is the input!

        forecasts shape:  [1, 6, 21, 721, 1440] i.e. [batch_size, steps, num_channels, height, width]

        The first input to the model is formed by:
            - reanalysis shape: [1, 0, 21, 721, 1440]
            - generated shape:  [1, 0, 10, 721, 1440]

        The output of the model is stored in the forecasts tensor, at:
            - forecasts shape: [1, 1, 21, 721, 1440]

        The next input to the model is formed by:
            - forecasts shape: [1, 1, 21, 721, 1440]
            - generated shape:  [1, 1, 10, 721, 1440]
        """
        self.optimizer.zero_grad()
        
        # Build forecasts container
        forecasts = torch.zeros_like(reanalysis[:, 1:])
        for step_i in range(forecasts.shape[1]):
            if step_i == 0:
                model_input = torch.cat([reanalysis[0, 0], generated[0, 0]], dim=0)
            else:
                model_input = torch.cat([forecasts[0, step_i-1], generated[0, step_i]], dim=0)
            # [21, 721, 1440] <- [21, 721, 1440]
            output = self.model(model_input)
            forecasts[0, step_i] = output
            print(step_i, torch.cuda.max_memory_allocated() / 1024**3, "GB")

        print("reanalysis[0, 1:]", reanalysis[0, 1:].shape)
        print("forecasts[0]", forecasts[0].shape)

        loss = torch.mean(torch.square(reanalysis[0, 1:] - forecasts[0]))
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss

        exit()




        #         model_input = torch.cat([forecasts[step_i-1], generated[step_i]], dim=0)
        #         print("model_input shape: ", model_input.shape)
        #         print("model_input dtype: ", model_input.dtype)
        #         print("model_input device: ", model_input.device)

        #         forecasts[step_i] = self.model(model_input)
        #         print("forecasts shape: ", forecasts[step_i].shape)
        #         print("forecasts dtype: ", forecasts[step_i].dtype)
        #         print("forecasts device: ", forecasts[step_i].device)

        # loss = torch.mean(torch.square(reanalysis - forecasts))
        # loss.backward()
        # self.optimizer.step()
        # self.scheduler.step()
        # return loss

        # loss = None
        # rollout_steps = truth.shape[1]
        # for rollout_step in range(rollout_steps):
        #     # [1, 21, 721, 1440] <- [1, 1, 31, 721, 1440]
        #     prediction = self.model(input if loss is None else next_input)

        #     if loss is None:
        #         loss = torch.mean(torch.pow(prediction - truth[:, rollout_step, :self.cfg.output_channels], 2))
        #     else:
        #         loss = loss.detach() + torch.mean(torch.square(prediction - truth[:, rollout_step, :self.cfg.output_channels]))

        #     # Next input needed?
        #     if rollout_step + 1 < rollout_steps:
        #         next_input = torch.cat([
        #             prediction.detach().unsqueeze(1), # [1, 21, 721, 1440] ----------------------> [1, 1, 21, 721, 1440]
        #             truth[:, [rollout_step], self.cfg.output_channels:] # [1, 5, 31, 721, 1440] -> [1, 1, 10, 721, 1440]
        #         ], dim=2)  # -> [1, 1, 31, 721, 1440]

        # loss.backward()
        # self.optimizer.step()
        # self.scheduler.step()
        # return loss

    @torch.inference_mode()
    def test(self):
        total_mse = 0.0
        samples_tested = 0
        for data in self.testing_datapipe:
            input = data[0]["input"].to(dtype=self.dtype)
            expectations = data[0]["output"].to(dtype=self.dtype)

            predictions = expectations.clone()
            next_input = input
            for step in range(self.cfg.num_testing_steps):
                prediction = self.model(next_input)
                predictions[0, step, :21] = prediction
                next_input = predictions[:, [step]]

            total_mse += torch.mean(torch.square(predictions[:, :, :self.cfg.output_channels] - expectations[:, :, :self.cfg.output_channels]))

            # Tested enough?
            samples_tested += 1
            if samples_tested >= self.cfg.num_testing_samples_per_rank:
                break

        mse = total_mse / samples_tested
        return mse

    def log_train_step(self, loss, time_taken, global_sample_id):
        # Get average among all ranks
        torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
        loss /= self.dist.world_size
        loss = loss.detach().cpu().item()

        # Log info
        self.zlogger.log(f"Iteration {self.iter:4d}:  train loss: {loss:10.3e}, time taken: {time_taken:10.3e}, global_sample_id: {global_sample_id}")
        if self.dist.rank == 0:
            wandb.log({
                "loss": loss,
                "learning_rate": self.scheduler.get_last_lr()[0],
                "time_taken_training": time_taken,
            }, step=self.iter)

    def log_test_step(self, test_mse, time_taken):
        # Get average among all ranks
        torch.distributed.all_reduce(test_mse, op=torch.distributed.ReduceOp.SUM)
        test_mse = test_mse / self.dist.world_size
        test_mse = test_mse.detach().cpu().item()

        # Log info
        self.zlogger.info(f"Iteration {self.iter:4d}:  test MSE:   {test_mse:10.3e}, time taken: {time_taken:10.3e}")
        if self.dist.rank == 0:
            wandb.log({
                "test_mse": test_mse,
                "time_taken_testing": time_taken,
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
        return self.iter >= self.cfg.phase1_iters + self.cfg.phase2_iters

    @property
    def do_rollout_increase(self):
        iter_in_phase3 = self.iter - (self.cfg.phase1_iters + self.cfg.phase2_iters)
        iters_per_increase = self.cfg.phase3_iters // self.cfg.phase3_increments
        return self.is_phase3 and iter_in_phase3 % iters_per_increase == 0

    @property
    def do_produce_artifacts(self):
        return self.iter % self.cfg.testing_artifacts_frequency == 0

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
        training_datapipe_sample_in_epoch
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
            0, # iterator_offset_epoch_idx
            0, # iterator_offset_sample_idx
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

        model.module.check_args(cp_dict["metadata"]) # raises ValueError if metadata is inconsistent

        model.module.load_state_dict(cp_dict["model"])
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
            cp_dict["training_datapipe_sample_in_epoch"] + 1
        )