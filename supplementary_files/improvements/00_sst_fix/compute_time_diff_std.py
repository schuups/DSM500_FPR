# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2025 schups@gmail.com (Modifications)
# SPDX-License-Identifier: Apache-2.0

import sys
sys.path.insert(0, '/iopsstor/scratch/cscs/stefschu/DSM500_FPR/modulus-baseline')

from modulus.utils.distributed_manager import DistributedManager as DM
from modulus.utils.caching import Cache
from modulus.datapipes.era5_hdf5 import ERA5HDF5Datapipe
from modulus.models.graph_cast_net import GraphCastNet

import torch
from omegaconf import OmegaConf
import numpy as np

def main():
    cfg = OmegaConf.load('/iopsstor/scratch/cscs/stefschu/DSM500_FPR/modulus-baseline/conf/config.yaml')

    # force
    cfg.toggles.data.include_sst_channel = True
    cfg.toggles.data.fix_sst_data = False

    # ligthen up loading
    cfg.toggles.model.include_static_data = False
    cfg.toggles.model.include_spatial_info = False
    cfg.toggles.model.include_solar_radiation = False
    cfg.toggles.model.include_temporal_info = False

    DM.initialize()
    Cache.initialize(dir=cfg.cache.dir)

    # init the model to have the field data structures
    
    model = GraphCastNet(cfg)

    datapipe = ERA5HDF5Datapipe(
        cfg=cfg,
        dataset_folder='lap',
        num_output_steps=5,

        latitudes=model.latitudes,
        longitudes=model.longitudes,
        map_grid_to_latlon=model.map_grid_to_latlon,
        dtype=model.dtype(),

        iterator={
            "shuffle": True,
            "shuffle_seed": 0,
            "initial_epoch_idx": 0,
            "initial_sample_idx": 0
        }
    )

    mean, mean_sqr = 0, 0
    samples_per_rank = 10
    for i, sample in enumerate(datapipe):
        if i >= samples_per_rank:
            break

        data = sample["data"]
        invar = data[0, :21]
        outvar = data[1, :21]

        diff = outvar - invar

        weighted_diff = model.area * diff
        weighted_diff_sqr = torch.square(weighted_diff)

        mean += torch.mean(weighted_diff, dim=(1, 2))
        mean_sqr += torch.mean(weighted_diff_sqr, dim=(1, 2))

        if i % 10 == 0 and i != 0 and DM.rank() == 0:
            print(f"Progress: {100 * i / samples:.2f}%")

    mean /= samples_per_rank
    mean_sqr /= samples_per_rank
    
    if DM.world_size() > 1:
        torch.distributed.all_reduce(mean, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(mean_sqr, op=torch.distributed.ReduceOp.SUM)
        mean /= DM.world_size()
        mean_sqr /= DM.world_size()

    if DM.rank() == 0:
        variance = mean_sqr - mean**2  # [1,num_channel, 1,1]
        std = torch.sqrt(variance)
        np.save("time_diff_std_with_sst_fix.npy", std.to(dtype=torch.float32).cpu().unsqueeze(0).numpy())
        np.save("time_diff_mean_with_sst_fix.npy", mean.to(dtype=torch.float32).to(torch.device("cpu")).unsqueeze(0).numpy())

    print("ended!")


if __name__ == "__main__":
    main()
