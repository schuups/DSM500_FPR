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

import os
import torch
import matplotlib.pyplot as plt

from modulus.datapipes.climate.era5_hdf5_new import ERA5HDF5Datapipe

from train_utils import prepare_input


import wandb
from hydra.utils import to_absolute_path
from omegaconf import DictConfig


class Validation:
    """Run validation on GraphCast model"""

    def __init__(self, cfg: DictConfig, model, dtype, dist, rank_zero_logger):
        self.val_dir = to_absolute_path(cfg.val_dir)
        self.model = model
        self.dtype = dtype
        self.dist = dist
        self.num_val_samples_per_rank = cfg.num_val_samples_per_rank
        self.val_datapipe = ERA5HDF5Datapipe(
            base_path=cfg.dataset_path,
            data_folder="test",
            num_steps=cfg.num_val_steps,
            dist=dist
        )

        rank_zero_logger.success(
            f"Loaded validation datapipe of size {len(self.val_datapipe)}"
        )

    @torch.inference_mode()
    def step(self, channels=[0, 1, 2], iter=0, time_idx=None, generate_plots=True):
        torch.cuda.nvtx.range_push("Validation")
        os.makedirs(self.val_dir, exist_ok=True)
        total_mse = 0.0
        for i, data in enumerate(self.val_datapipe):
            if i >= self.num_val_samples_per_rank:
                break

            # Prepare the input & output
            input = data[0]["input"].to(dtype=self.dtype)
            output = data[0]["output"].to(dtype=self.dtype)

            predictions = (
                torch.empty(output.shape)
                .to(dtype=self.dtype)
                .to(device=self.dist.device)
            )
            for t in range(output.size(dim=1)):
                _pred = self.model(input)

                _pred = _pred.reshape(1, 1, 21, 721, 1440)
                _pred = torch.cat([_pred, output[:, [t], 21:]], axis=2)

                predictions[0, t] = _pred
                input = _pred

            total_mse += torch.mean(torch.pow(predictions - output, 2))
            torch.cuda.nvtx.range_pop()

            del input, _pred

            if generate_plots:
                predictions = predictions.to(torch.float32).cpu().numpy()
                output = output.to(torch.float32).cpu().numpy()

                if i == 0:
                    for chan in channels:
                        plt.close("all")
                        fig, ax = plt.subplots(3, predictions.shape[1], figsize=(15, 5))
                        fig.subplots_adjust(hspace=0.5, wspace=0.3)

                        for t in range(output.shape[1]):
                            im_pred = ax[0, t].imshow(predictions[0, t, chan], vmin=-1.5, vmax=1.5)
                            ax[0, t].set_title(f"Prediction (t={t+1})", fontsize=10)
                            fig.colorbar(
                                im_pred, ax=ax[0, t], orientation="horizontal", pad=0.4
                            )

                            im_outvar = ax[1, t].imshow(
                                output[0, t, chan], vmin=-1.5, vmax=1.5
                            )
                            ax[1, t].set_title(f"Ground Truth (t={t+1})", fontsize=10)
                            fig.colorbar(
                                im_outvar, ax=ax[1, t], orientation="horizontal", pad=0.4
                            )

                            im_diff = ax[2, t].imshow(
                                abs(predictions[0, t, chan] - output[0, t, chan])#, vmin=0.0, vmax=0.5
                            )
                            ax[2, t].set_title(f"Abs. Diff. (t={t+1})", fontsize=10)
                            fig.colorbar(
                                im_diff, ax=ax[2, t], orientation="horizontal", pad=0.4
                            )

                        fig.savefig(
                            os.path.join(self.val_dir, f"era5_validation_channel{chan}_iter{iter}.png")
                        )
                        wandb.log({f"val_chan{chan}_iter{iter}": fig}, step=iter)

        mse = torch.tensor([total_mse / self.num_val_samples_per_rank], dtype=self.dtype, device=self.dist.device)
        torch.distributed.all_reduce(mse, op=torch.distributed.ReduceOp.SUM)
        mse /= self.dist.world_size

        return mse.item()
