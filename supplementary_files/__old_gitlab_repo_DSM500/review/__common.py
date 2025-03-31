# The findings of the various notebook files are consolidated here, to avoid reapeating the same code in each notebook.

import os
import sys
import json
import h5py
import hydra
from hydra.utils import to_absolute_path
import torch
import torch.nn as nn
import torch.nn.functional as F
import modulus
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import netCDF4 as nc

# if issues opening .nc files, run "!pip install netCDF4 h5py --upgrade" and restart kernel

# example scripts
sys.path.append('modulus/examples/weather/graphcast')

from loss.utils import normalized_grid_cell_area
from transformer_engine import pytorch as te
from modulus.models.layers import get_activation

# config
with hydra.initialize(config_path="../modulus/examples/weather/graphcast/conf", version_base="1.3"):
    cfg = hydra.compose(config_name="config")
# config overwrites
cfg.num_channels_climate = 21 # <--- My files are those from NERSC, thus with 21 channels (default for modulus's graphcast is 73)
cfg.dataset_path = "/mnt/data-slow/data/FCN_ERA5_data_v0"
cfg.time_diff_std_path = "/mnt/data-slow/data/FCN_ERA5_data_v0/stats/time_diff_std.npy"
cfg.dataset_metadata_path = "/mnt/data-slow/data/FCN_ERA5_data_v0/metadata.json"
cfg.static_dataset_path = '/mnt/data-slow/data/FCN_ERA5_data_v0/static'
cfg.start_year = 2014 # <--- This is the first year of data available for this development

# area
latitudes = torch.linspace(-90, 90, steps=cfg.latlon_res[0])
longitudes = torch.linspace(-180, 180, steps=cfg.latlon_res[1] + 1)[1:]
lat_lon_grid = torch.stack(torch.meshgrid(latitudes, longitudes, indexing="ij"), dim=-1)
area = normalized_grid_cell_area(lat_lon_grid[:, :, 0], unit="deg")

# channel list
channels_list = [i for i in range(cfg.num_channels_climate)]

# distributed manager initialization (by mocking a simple slurm job)
from modulus.distributed import DistributedManager
import random
os.environ["MODULUS_DISTRIBUTED_INITIALIZATION_METHOD"] = "ENV"
os.environ["MASTER_PORT"] = str(random.randint(10000, 20000)) # <--- Several notebooks can be running at the same time, none of which is running anything distributed
os.environ["MASTER_ADDR"] = "localhost"
os.environ["WORLD_SIZE"] = "1"
os.environ["RANK"] = "0"

DistributedManager.initialize()
dist = DistributedManager()
