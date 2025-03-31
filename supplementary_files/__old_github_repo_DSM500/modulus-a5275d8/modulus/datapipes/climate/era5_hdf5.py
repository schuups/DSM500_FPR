# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2025 schups@gmail.com (Modifications)
# SPDX-License-Identifier: Apache-2.0

import os
import h5py
import torch
import warnings
import numpy as np
import netCDF4 as nc
from pathlib import Path
from datetime import datetime, timedelta

import nvidia.dali as dali
import nvidia.dali.plugin.pytorch as dali_pth

from modulus.distributed import DistributedManager
from modulus.datapipes.climate.utils.zenith_angle import cos_zenith_angle

def fix_latitude_alignment(data):
    if isinstance(data, torch.Tensor):
        return torch.cat((data[..., 720:], data[..., :720]), axis=-1)
    elif isinstance(data, np.ndarray):
        return np.concatenate((data[..., 720:], data[..., :720]), axis=-1)
    raise ValueError("Data must be either a torch.Tensor or a numpy.ndarray")


class ERA5HDF5Datapipe:
    def __init__(
        self,
        model,
        
        dataset_base_path: str,
        dataset_folder: str,
        dataset_samples_per_file: int,

        num_output_steps: int = 1,

        # Needed for restarting from checkpoint
        iterator_shuffle: bool = True,
        iterator_seed: int = 0,
        iterator_offset_epoch_idx: int = 0,
        iterator_offset_sample_idx: int = 0,

        num_threads: int = 2,
        prefetch_queue_depth: int = 2,
        parallel: bool = False
    ):
        self.model = model

        self.dataset_base_path = Path(dataset_base_path)
        assert self.dataset_base_path.is_dir(), f"Dataset base path {self.dataset_base_path} does not exist"
        self.dataset_folder = self.dataset_base_path / dataset_folder
        self.dataset_samples_per_file = dataset_samples_per_file

        self.num_steps = 1 + num_output_steps # i.e. the input datapoint + num_output_steps output datapoints

        self.num_threads = num_threads
        self.prefetch_queue_depth = prefetch_queue_depth
        self.parallel = parallel

        if DistributedManager.is_initialized():
            _dist = DistributedManager()
            self.rank = _dist.rank
            self.world_size = _dist.world_size
            self.device_index = _dist.device.index
        else:
            self.rank, self.world_size, self.device_index = 0, 1, dali.types.CPU_ONLY_DEVICE_ID

        self._load_file_paths(dataset_folder=self.dataset_folder)
        self._load_statistics(base_path=self.dataset_base_path / "stats")
        if self.model.includes_static_data:
            self._load_static_data(base_path=self.dataset_base_path / "static")
        if self.model.includes_spatial_info:
            self._generate_spatial_info()
        if self.model.includes_solar_radiation:
            self.map_grid_to_latlon = dali.types.Constant(
                # (721, 1440, 2) -> (2, 721, 1440)
                self.model.map_grid_to_latlon.permute(2, 0, 1)
            )

        self._create_pipeline(
            iterator_shuffle=iterator_shuffle,
            iterator_seed=iterator_seed,
            iterator_offset_epoch_idx=iterator_offset_epoch_idx,
            iterator_offset_sample_idx=iterator_offset_sample_idx
        )

        # Checks
        assert 0 < num_output_steps < len(self), f"'num_output_steps' must be > 0, gotten {num_output_steps}"       

    def _load_file_paths(self, dataset_folder: Path):
        assert dataset_folder.is_dir(), f"Dataset folder {dataset_folder} does not exist"
        # Results in a dictionary with year as key and path as value
        self.file_paths = {int(path.stem): str(path) for path in sorted(dataset_folder.glob("????.h5"))}        

    def _load_statistics(self, base_path: Path):
        path_means = base_path / "global_means.npy"
        path_stds = base_path / "global_stds.npy"

        if path_means.exists() or path_stds.exists():    
            self.means = torch.from_numpy(np.load(path_means)) # shape: (1, 21, 1, 1)
            self.stds = torch.from_numpy(np.load(path_stds)) # shape: (1, 21, 1, 1)
            assert self.means.shape[1] == 21 and self.stds.shape[1] == 21
        else:
            warnings.warn(f"Statistics files not found in {base_path}, setting mean=0.0 and std=1.0")
            self.means = torch.zeros((1, 21, 1, 1), dtype=torch.float32)
            self.stds = torch.ones((1, 21, 1, 1), dtype=torch.float32)

    def _load_static_data(self, base_path: Path):
        # land-sea mask
        path_lsm = base_path / "land_sea_mask.nc"
        assert path_lsm.exists(), f"Land-sea mask file not found in {path_lsm}"
        lsm = torch.as_tensor(nc.Dataset(path_lsm)["lsm"][:])
        lsm = (lsm - lsm.mean()) / lsm.std()
        # (1, 721, 1440) -> (1, 1, 721, 1440) -> (num_steps, 1, 721, 1440)
        lsm = lsm.unsqueeze(0).expand(self.num_steps, -1, -1, -1)
        lsm = fix_latitude_alignment(lsm)
        
        # geo-potential
        path_z = base_path / "geopotential.nc"
        assert path_z.exists(), f"Geopotential file not found in {path_z}"
        z = torch.as_tensor(nc.Dataset(path_z)["z"][:])
        z = (z - z.mean()) / z.std()
        # (1, 721, 1440) -> (1, 1, 721, 1440) -> (num_steps, 1, 721, 1440)
        z = z.unsqueeze(0).expand(self.num_steps, -1, -1, -1)
        z = fix_latitude_alignment(z)

        self.static_data = dali.types.Constant(
            torch.cat([lsm, z], dim=1)
        )
    
    def _generate_spatial_info(self):
        # cos latitudes
        cos_lat = torch.cos(torch.deg2rad(self.model.latitudes)) # shape: (721,)
        cos_lat = (cos_lat - cos_lat.mean()) / cos_lat.std()
        # (721,) -> (1, 1, 721, 1) -> (num_steps, 1, 721, 1440)
        cos_lat = cos_lat.view(1, 1, -1, 1).expand(self.num_steps, 1, -1, self.model.sample_width)
        
        # sin longitudes
        sin_lon = torch.sin(torch.deg2rad(self.model.longitudes)) # shape: (1440,)
        sin_lon = (sin_lon - sin_lon.mean()) / sin_lon.std()
        # (721,) -> (1, 1, 1, 1440) -> (num_steps, 1, 721, 1440)
        sin_lon = sin_lon.view(1, 1, 1, -1).expand(self.num_steps, 1, self.model.sample_height, -1)
        
        # cos longitudes
        cos_lon = torch.cos(torch.deg2rad(self.model.longitudes)) # shape: (1440,)
        cos_lon = (cos_lon - cos_lon.mean()) / cos_lon.std()
        # (721,) -> (1, 1, 1, 1440) -> (num_steps, 1, 721, 1440)
        cos_lon = cos_lon.view(1, 1, 1, -1).expand(self.num_steps, 1, self.model.sample_height, -1)

        self.spatial_info = dali.types.Constant(
            torch.cat([cos_lat, sin_lon, cos_lon], dim=1),
            dtype=dali.types.FLOAT
        )

    def is_on_gpu(self):
        return self.device_index >= 0

    def __iter__(self):
        self.pipe.reset()
        return dali_pth.DALIGenericIterator([self.pipe], [
            "epoch_idx", "idx_in_epoch", "global_sample_id", "data", "timestamps"
        ])

    def __len__(self):
        return len(self.file_paths) * self.dataset_samples_per_file - self.num_steps

    @property
    def channels_count(self):
        return 21 # Okeyish being hardcoded because it is tight to the data on disk

    @property
    def generated_channels_count(self):
        return int(self.model.includes_static_data) * 2 \
            + int(self.model.includes_spatial_info) * 3 \
            + int(self.model.includes_solar_radiation) * 1 \
            + int(self.model.includes_temporal_info) * 4

    def _create_pipeline(
        self,
        iterator_shuffle: bool,
        iterator_seed: int,
        iterator_offset_epoch_idx: int,
        iterator_offset_sample_idx: int
    ):
        self.pipe = dali.Pipeline(
            batch_size=self.model.batch_size,
            py_start_method="spawn",
            prefetch_queue_depth=self.prefetch_queue_depth,
            num_threads=self.num_threads,
            device_id=self.device_index,
            set_affinity=self.is_on_gpu(),
        )

        with self.pipe:
            source = ERA5DaliExternalSource(
                num_samples=len(self),
                file_paths=self.file_paths,
                num_steps=self.num_steps,
                samples_per_file=self.dataset_samples_per_file,

                iterator_shuffle=iterator_shuffle,
                iterator_seed=iterator_seed,
                iterator_offset_epoch_idx=iterator_offset_epoch_idx,
                iterator_offset_sample_idx=iterator_offset_sample_idx,

                rank=self.rank,
                world_size=self.world_size
            )

            epoch_idx, idx_in_epoch, global_sample_id, timestamps, temporal_features, data = dali.fn.external_source(
                source,
                num_outputs=6,
                parallel=self.parallel,
                batch=False
            )

            # normalize data
            data = dali.fn.normalize(data, mean=self.means, stddev=self.stds)

            # other channels
            other_channels = list()
            if self.model.includes_static_data:
                other_channels.append(self.static_data)
            if self.model.includes_spatial_info:
                other_channels.append(self.spatial_info)
            if self.model.includes_solar_radiation:
                other_channels.append(
                    cos_zenith_angle(timestamps, latlon=self.map_grid_to_latlon) / .55  # standardization 
                )
            if self.model.includes_temporal_info:
                temporal_features_template = dali.fn.ones(
                    shape=(1, 1, self.model.sample_height, self.model.sample_width),
                    dtype=dali.types.FLOAT
                )
                other_channels.append(
                    dali.fn.reshape(temporal_features, shape=(self.num_steps, 4, 1, 1)) * temporal_features_template
                )
            
            # concatenate, move, split and return
            data = dali.fn.cat(data, *other_channels, axis=1)
            data = data.gpu()
            
            self.pipe.set_outputs(epoch_idx, idx_in_epoch, global_sample_id, data, timestamps)

class ERA5DaliExternalSource:
    def __init__(
        self,
        num_samples: int,
        file_paths: list,
        num_steps: int,
        samples_per_file: int,

        iterator_shuffle: bool,
        iterator_seed: int,
        iterator_offset_epoch_idx: int,
        iterator_offset_sample_idx: int,

        rank: int,
        world_size: int
    ):
        self.num_samples = num_samples
        self.file_paths = file_paths
        self.files = None
        self.first_year = min(self.file_paths.keys())
        self.num_steps = num_steps
        self.samples_per_file = samples_per_file

        self.iterator_shuffle = iterator_shuffle
        self.iterator_seed = iterator_seed
        self.iterator_offset_epoch_idx = iterator_offset_epoch_idx
        self.iterator_offset_sample_idx = iterator_offset_sample_idx
        self.rank = rank
        self.world_size = world_size

        self.build_idx_map(epoch_idx=self.iterator_offset_epoch_idx)

    def build_idx_map(self, epoch_idx=0):
        idx_map = np.arange(self.num_samples, dtype=np.uint16)
        if self.iterator_shuffle:
            np.random.default_rng(seed=self.iterator_seed + epoch_idx).shuffle(idx_map)
        self.idx_map = np.array_split(idx_map, self.world_size)[self.rank]
        self.idx_map_length = len(self.idx_map)

    def open_files(self):
        self.files = {year: h5py.File(path, "r", libver="latest", swmr=True) for year, path in self.file_paths.items()}

    def parse_sample_info(self, sample_info: dali.types.SampleInfo):
        epoch_idx = sample_info.epoch_idx + self.iterator_offset_epoch_idx
        idx_in_epoch = sample_info.idx_in_epoch + self.iterator_offset_sample_idx

        if idx_in_epoch >= self.idx_map_length:
            self.iterator_offset_sample_idx = 0 # in the next epoch, start from the beginning!
            self.build_idx_map(epoch_idx=epoch_idx + 1) # prepare map for the next epoch
            raise StopIteration()

        # Effective reading index
        global_sample_idx = self.idx_map[idx_in_epoch]
        year = self.first_year + global_sample_idx // self.samples_per_file
        idx_start = global_sample_idx % self.samples_per_file
        idx_end = idx_start + self.num_steps

        return epoch_idx, idx_in_epoch, year, idx_start, idx_end

    def extract_data(self, idx_start, idx_end, year):
        if idx_end < self.samples_per_file:
            data = torch.from_numpy(self.files[year]["fields"][idx_start:idx_end])
        else:
            data_left = self.files[year]["fields"][idx_start:]
            data_right = self.files[year+1]["fields"][:idx_end - self.samples_per_file]
            data = torch.from_numpy(np.concatenate([data_left, data_right], axis=0))
        # Fix sst by filling missing data with t2m data
        data[:, 20] = torch.where(data[:, 20] == -32767.0, data[:, 2], data[:, 20])
        # Fix latitude alignment: dataset is from 0 to 360 (Europe on the left), but should be -180 to 180 (Europe at the centre)
        return fix_latitude_alignment(data)

    def extract_timestamps(self, idx_start, idx_end, year):
        timestamps = list()
        temporal_features = list()
        for idx in range(idx_start, idx_end):
            _date = datetime(year, 1, 1) + timedelta(hours=6) * idx
            timestamps.append(_date.timestamp())
            # progress in the year
            _year_start = datetime(_date.year, 1, 1).timestamp()
            _year_end = datetime(_date.year + 1, 1, 1).timestamp()
            _progress_in_year = (_date.timestamp() - _year_start) / (_year_end - _year_start)
            # progress in the day
            _progress_in_day = _date.hour / 24
            temporal_features.append([
                np.sin(_progress_in_year * np.pi * 2),
                np.cos(_progress_in_year * np.pi * 2),
                np.sin(_progress_in_day * np.pi * 2),
                np.cos(_progress_in_day * np.pi * 2)
            ])
        return timestamps, temporal_features

    def __call__(self, sample_info: dali.types.SampleInfo):
        if self.files is None:
            self.open_files()

        epoch_idx, idx_in_epoch, year, idx_start, idx_end = self.parse_sample_info(sample_info)

        # Extract data
        data = self.extract_data(idx_start, idx_end, year)
        
        # Generate timestamps for this data, and also precompute temporal features
        timestamps, temporal_features = self.extract_timestamps(idx_start, idx_end, year)

        # Dali does not support uint16, so int32 is needed
        return (
            # epoch id
            torch.tensor(epoch_idx, dtype=torch.int32),
            # sample id in epoch
            torch.tensor(idx_in_epoch, dtype=torch.int32),
            # global sample ids
            torch.tensor(self.idx_map[idx_in_epoch], dtype=torch.int32),
            # timestsamp of data
            np.array(timestamps, dtype=np.int32),
            # precomputed temporal features
            np.array(temporal_features, dtype=np.float32),
            # data (it is already a torch tensor)
            data
        )
