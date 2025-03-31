# The following aggregates all that can be put in common between all notebooks

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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import defaultdict
import netCDF4 as nc
import random
import xarray as xr
import time
import timeit

# Add example scripts
sys.path.append('/iopsstor/scratch/cscs/stefschu/DSM500/github/modulus-a5275d8/examples/weather/graphcast')

from train_utils import prepare_input

# config
with hydra.initialize(config_path=".", version_base="1.3"):
    cfg = hydra.compose(config_name="config")

# area
from loss.utils import normalized_grid_cell_area
latitudes = torch.linspace(-90, 90, steps=cfg.latlon_res[0])
longitudes = torch.linspace(-180, 180, steps=cfg.latlon_res[1] + 1)[1:]
lat_lon_grid = torch.stack(torch.meshgrid(latitudes, longitudes, indexing="ij"), dim=-1)
area = normalized_grid_cell_area(lat_lon_grid[:, :, 0], unit="deg")

# channel list
channels_list = [i for i in range(cfg.num_channels_climate)]

# distributed manager initialization (by mocking a simple slurm job)
from modulus.distributed import DistributedManager
os.environ["MODULUS_DISTRIBUTED_INITIALIZATION_METHOD"] = "ENV"
os.environ["MASTER_PORT"] = str(random.randint(10000, 20000)) # <--- Several notebooks can be running at the same time, none of which is running anything distributed
os.environ["MASTER_ADDR"] = "localhost"
os.environ["WORLD_SIZE"] = "1"
os.environ["RANK"] = "0"

DistributedManager.initialize()
dist = DistributedManager()

def plot(data, title, axs=None, last=False):
    if axs is None:
        fig, axs = plt.subplots(1, 1)
        last = True
    im = axs.imshow(data)
    axs.set_title(title, fontsize=10)
    divider = make_axes_locatable(axs)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax)
    plt.tight_layout()
    if last:
        plt.show()

def get_dataset_metadata():
    with open("/iopsstor/scratch/cscs/stefschu/DSM500/data/FCN_ERA5_data_v0/metadata.json") as f:
        return json.load(f)
metadata = get_dataset_metadata()
channel_names = metadata["coords"]["channel"]

# For studying the train script
from modulus.datapipes.climate import ERA5HDF5Datapipe
from modulus.launch.logging import (
    PythonLogger,
    RankZeroLoggingWrapper
)
from omegaconf import DictConfig
logger = PythonLogger("main")  # General python logger
rank_zero_logger = RankZeroLoggingWrapper(logger, dist)  # Rank 0 logger
rank_zero_logger.file_logging()

# Instantiate dummy trainer
from train_graphcast import GraphCastTrainer
trainer = GraphCastTrainer(cfg, dist, rank_zero_logger)

cos_zenith_args = {
    "dt": cfg.dt,
    "start_year": cfg.start_year,
}