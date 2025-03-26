# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2025 schups@gmail.com (Modifications)
# SPDX-License-Identifier: Apache-2.0

import math
from collections import defaultdict, Counter
from pathlib import Path

import json
import numpy as np
import torch
import torch.nn as nn


class GraphCastLossFunction(nn.Module):
    """Loss function as specified in GraphCast.
    Parameters
    ----------
    area : torch.Tensor
        Cell area with shape [H, W].
    """

    def __init__(
        self,
        cfg,
        area,
        channels_metadata
    ):
        super().__init__()

        self.cfg = cfg
        self.area = area

        _channels_list = list(range(0, 21 if self.cfg.toggles.data.include_sst_channel else 20))

        # Determine time step std paths
        time_diff_std_path_new = Path(self.cfg.dataset.base_path) / "stats/time_diff_std_with_sst_fix.npy"
        time_diff_std_path_old = Path(self.cfg.dataset.base_path) / "stats/time_diff_std.npy"
        
        # Load time std for inverse variance
        self.inverse_variance_weights_old = self.get_time_diff_std(time_diff_std_path_old, _channels_list)
        self.inverse_variance_weights_new = self._get_inverse_variance_weights(time_diff_std_path_new)

        # Build variable groups weights
        old_channels_metadata_path = Path(self.cfg.dataset.base_path) / "metadata.json"
        self.channel_dict = self.get_channel_dict(old_channels_metadata_path, _channels_list)
        self.variable_weights_old = self.assign_variable_weights()
        self.variable_weights_new = self._get_variable_weights(channels_metadata)

    def _get_inverse_variance_weights(self, time_diff_std_path):
        time_diff_std = torch.from_numpy(np.load(time_diff_std_path)).to(
            dtype=self.area.dtype, 
            device=self.area.device
        )

        if not self.cfg.toggles.data.include_sst_channel:
            time_diff_std = time_diff_std[:, :20]

        return 1.0 / torch.square(time_diff_std.view(-1, 1, 1))

    def _get_variable_weights(self, channels_metadata):
        """
        Assigns weights to variables, so that each group is equally represented in the loss
        """
        group_weights = list(map(lambda x: x["group"], channels_metadata))  
        group_weights_map = {k: 1 / v for k, v in Counter(group_weights).items()}

        group_weights = torch.tensor(
            [group_weights_map[k] for k in group_weights], 
            dtype=self.area.dtype,
            device=self.area.device
        )

        return group_weights.view(-1, 1, 1)

    def forward(self, invar, outvar):
        """
        Implicit forward function which computes the loss given
        a prediction and the corresponding targets.
        Parameters
        ----------
        invar : torch.Tensor
            prediction of shape [T, C, H, W].
        outvar : torch.Tensor
            target values of shape [T, C, H, W].
        """

        # outvar normalization
        loss = torch.square(invar - outvar).mean(dim=0)

        # weighted by inverse variance
        # The fixed invariance only makes sense when sst is present and filled
        if self.cfg.toggles.data.include_sst_channel and self.cfg.toggles.data.fix_sst_data and self.cfg.toggles.loss.fix_inverse_variance_data:
            loss *= self.inverse_variance_weights_new
        else:
            loss = (
                loss
                * 1.0
                / torch.square(self.inverse_variance_weights_old.view(-1, 1, 1).to(loss.device))
            )

        # weighted by variables
        if self.cfg.toggles.loss.use_original_variable_weights:
            variable_weights = self.variable_weights_old.view(-1, 1, 1).to(loss.device)
            loss = loss * variable_weights  # [T,C,H,W]
        else:
            loss *= self.variable_weights_new
        
        # weighted by area
        loss = torch.mul(loss, self.area)
        loss = loss.mean()
        return loss

    def get_time_diff_std(self, time_diff_std_path, channels_list):
        """Gets the time difference standard deviation"""
        if time_diff_std_path is not None:
            time_diff_np = np.load(time_diff_std_path)
            time_diff_np = time_diff_np[:, channels_list, ...]
            return torch.FloatTensor(time_diff_np)
        else:
            return torch.tensor([1.0], dtype=torch.float)

    def get_channel_dict(self, dataset_metadata_path, channels_list):
        """Gets lists of surface and atmospheric channels"""
        with open(dataset_metadata_path, "r") as f:
            data_json = json.load(f)
            channel_list = [data_json["coords"]["channel"][c] for c in channels_list]

            # separate atmosphere and surface variables
            channel_dict = {"surface": [], "atmosphere": []}
            for each_channel in channel_list:
                if each_channel[-1].isdigit():
                    channel_dict["atmosphere"].append(each_channel)
                else:
                    channel_dict["surface"].append(each_channel)
            return channel_dict

    def parse_variable(self, variable_list):
        """Parse variable into its letter and numeric parts."""
        for i, char in enumerate(variable_list):
            if char.isdigit():
                return variable_list[:i], int(variable_list[i:])

    def calculate_linear_weights(self, variables):
        """Calculate weights for each variable group."""
        groups = defaultdict(list)
        # Group variables by their first letter
        for variable in variables:
            letter, number = self.parse_variable(variable)
            groups[letter].append((variable, number))
        # Calculate weights for each group
        weights = {}
        for values in groups.values():
            total = sum(number for _, number in values)
            for variable, number in values:
                weights[variable] = number / total

        return weights

    def assign_surface_weights(self):
        """Assigns weights to surface variables"""
        surface_weights = {i: 0.1 for i in self.channel_dict["surface"]}
        if "t2m" in surface_weights:
            surface_weights["t2m"] = 1
        return surface_weights

    def assign_atmosphere_weights(self):
        """Assigns weights to atmospheric variables"""
        return self.calculate_linear_weights(self.channel_dict["atmosphere"])

    def assign_variable_weights(self):
        """assigns per-variable per-pressure level weights"""
        surface_weights_dict = self.assign_surface_weights()
        atmosphere_weights_dict = self.assign_atmosphere_weights()
        surface_weights = list(surface_weights_dict.values())
        atmosphere_weights = list(atmosphere_weights_dict.values())
        variable_weights = torch.cat(
            (torch.FloatTensor(surface_weights), torch.FloatTensor(atmosphere_weights))
        )  # [num_channel]
        return variable_weights
