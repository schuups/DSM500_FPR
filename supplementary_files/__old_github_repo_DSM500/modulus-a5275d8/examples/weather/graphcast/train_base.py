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

import torch
from torch.profiler import profile, record_function, ProfilerActivity
from torch.amp import autocast

import sys


class BaseTrainer:
    """Trainer class"""

    def __init__(self):
        pass

    def rollout(self, input, output):
        with autocast('cuda', enabled=self.amp, dtype=self.amp_dtype):
            total_loss = 0.0
            previous_prediction = input
            for t in range(output.size(dim=1)):
                # Shape of y is [N, M, C, H, W]. M is the number of steps
                prediction = self.model(previous_prediction)
                loss = self.criterion(prediction, output[:, t, :21])
                total_loss += loss
                # Add static and time data to the prediction
                prediction = prediction.reshape(1, 1, 21, 721, 1440)
                prediction = torch.cat([prediction, output[:, [t], 21:]], axis=2)
                previous_prediction = prediction
            return total_loss

    def forward(self, input, output):
        # forward pass
        torch.cuda.nvtx.range_push("Loss computation")
        if self.pyt_profiler:
            with profile(
                activities=[ProfilerActivity.CUDA], record_shapes=True
            ) as prof:
                with record_function("training_step"):
                    loss = self.rollout(input, output)

            print(
                prof.key_averages(group_by_input_shape=True).table(
                    sort_by="cuda_time_total", row_limit=10
                )
            )
            exit(0)
        else:
            loss = self.rollout(input, output)
        torch.cuda.nvtx.range_pop()
        return loss

    def backward(self, loss):
        # backward pass
        torch.cuda.nvtx.range_push("Weight gradients")
        if self.amp:
            self.scaler.scale(loss).backward()
            torch.cuda.nvtx.range_pop()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            torch.cuda.nvtx.range_pop()
            self.optimizer.step()

    def train(self, input, output):
        self.optimizer.zero_grad()
        loss = self.forward(input, output)
        self.backward(loss)
        self.scheduler.step()

        return loss / output.size(dim=1)
