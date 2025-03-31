import os
import torch
import json

import numpy as np
from hydra.utils import to_absolute_path
from pathlib import Path
from hydra.utils import to_absolute_path
from modulus.distributed import DistributedManager

import netCDF4 as nc
from scipy.ndimage import sobel

from modulus.utils_new.caching import Cache
from modulus.models_baseline.graphcast.graph_cast_net import GraphCastNetBaseline
from modulus.datapipes.climate.era5_hdf5 import ERA5HDF5Datapipe, fix_latitude_alignment

class Inference:
    def __init__(self, cfg, logger):
        self.cfg = cfg
        DistributedManager.initialize()
        self.dist = DistributedManager()
        self.rank = self.dist.rank
        self.world_size = self.dist.world_size
        self.logger = logger

        # Setup artifacts folders
        self.output_dir = Path(to_absolute_path(self.cfg.inference.artifacts_folder))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)

        # Load channels metadata
        with open(Path(self.cfg.dataset.base_path) / "metadata.json", "r") as f:
            metadata = json.loads(f.read())
            self.channel_titles = metadata["coords"]["channel_titles"]
            self.channel_tick_format = metadata["coords"]["channel_tick_format"]

    def load_model_and_datapipe(self):
        assert self.world_size == 1, "Inference is not supported in distributed mode"

        Cache.initialize(dir=self.cfg.cache_dir)

        model = GraphCastNetBaseline(
            sample_height=self.cfg.sample.height,
            sample_width=self.cfg.sample.width,
            sample_channels=self.cfg.sample.channels,

            include_static_data=self.cfg.include.static_data,
            include_spatial_info=self.cfg.include.spatial_info,
            include_temporal_info=self.cfg.include.temporal_info,
            include_solar_radiation=self.cfg.include.solar_radiation,

            batch_size=self.cfg.datapipe.batch_size,
            mesh_level=self.cfg.mesh_level,
            activation_fn=self.cfg.activation_fn,
            hidden_dim=self.cfg.hidden_dim,
            hidden_layers=self.cfg.hidden_layers,
            aggregation_op=self.cfg.aggregation_op,
            processor_layers=self.cfg.processor_layers,
        )
        device = torch.device('cuda', index=0)
        model = model.to(dtype=self.dtype).to(device)

        # Load checkpoint
        checkpoint = torch.load(
            self.cfg.inference.checkpoint_file_path,
            map_location=device,
            weights_only=True
        )
        model.check_args(checkpoint["metadata"])
        model.load_state_dict(checkpoint["model"])

        # Load datapipe
        datapipe = ERA5HDF5Datapipe(
            model=model,
            dataset_base_path=self.cfg.dataset.base_path,
            dataset_folder="out_of_sample",
            dataset_samples_per_file=self.cfg.dataset.samples_per_file,

            num_output_steps=self.cfg.inference.rollout_steps,

            iterator_seed=self.cfg.seed,
            iterator_offset_epoch_idx=0,
            iterator_offset_sample_idx=0,

            num_threads=1,
            prefetch_queue_depth=1
        )

        return model, datapipe

    def load_means_and_stds(self):
        means = np.load(Path(to_absolute_path(self.cfg.dataset.base_path)) / "stats/global_means.npy")
        stds = np.load(Path(to_absolute_path(self.cfg.dataset.base_path)) / "stats/global_stds.npy")
        return means, stds

    def load_edges(self, threshold=0.5):
        with nc.Dataset(Path(to_absolute_path(self.cfg.dataset.base_path)) / "static/land_sea_mask.nc", "r") as f:
            lsm = np.array(f["lsm"][0])
            edge_x = sobel(lsm, axis=0)
            edge_y = sobel(lsm, axis=1)
            edges = np.hypot(edge_x, edge_y)
            edges = edges / edges.max()
            
            edges[edges > threshold] = 1.
            edges[edges <= threshold] = 0.

            # Fix longitutes alignment, to match that coming out of the model itself
            edges = fix_latitude_alignment(edges)
            return edges

    def get_channel_title(self, channel_id):
        return self.channel_titles[channel_id]

    def get_channel_tick_format(self, channel_id):
        return self.channel_tick_format[channel_id]

    @property
    def container_file_path(self):
        return self.output_dir / "container.h5"

    @property
    def video_file_path(self):
        return self.output_dir / "video.mp4"

    @property
    def plan_file_path(self):
        return self.output_dir / "plan.json"

    @property
    def dtype(self):
        return torch.bfloat16 if self.cfg.dtype else torch.float32
    
    @property
    def device(self):
        return self.dist.device
