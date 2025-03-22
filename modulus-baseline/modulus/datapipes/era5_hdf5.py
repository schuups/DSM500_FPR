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

from modulus.utils.distributed_manager import DistributedManager as DM
from modulus.datapipes.utils.zenith_angle import cos_zenith_angle

def fix_latitude_alignment(data):
    if isinstance(data, torch.Tensor):
        return torch.cat((data[..., 720:], data[..., :720]), axis=-1)
    elif isinstance(data, np.ndarray):
        return np.concatenate((data[..., 720:], data[..., :720]), axis=-1)
    raise ValueError("Data must be either a torch.Tensor or a numpy.ndarray")

class ERA5HDF5Datapipe:
    def __init__(
        self,
        cfg,
        dataset_folder: str,
        num_output_steps: int,
        model,

        seed: int,
        iterator: dict,

        prefetch_queue_depth: int,
        num_threads: int,
    ):
        self.cfg = cfg
        self.num_steps = 1 + num_output_steps # i.e. the input datapoint + num_output_steps output datapoints
        self.model = model

        # Load file paths
        dataset_base_path = Path(self.cfg.dataset.base_path).absolute()
        self.file_paths = self._load_h5_file_paths(path=dataset_base_path / dataset_folder)
        self.means, self.stds = self._load_statistics(path=dataset_base_path / "stats")

        # Load static data
        if self.cfg.toggles.model.include_static_data:
            self.static_data = self._load_static_data(path=dataset_base_path / "static")

        # Load spatial data
        if self.cfg.toggles.model.include_spatial_info:
            self.spatial_info = self._generate_spatial_info()

        # Prepare solar radiation data
        if self.cfg.toggles.model.include_solar_radiation:
            self.map_latlon_to_grid = dali.types.Constant(
                self.model.map_grid_to_latlon.permute(2, 0, 1) # i.e. (721, 1440, 2) -> (2, 721, 1440)
            )

        # Setup pipeline
        self.pipe = self._create_pipeline(
            seed=seed,
            iterator=iterator, 
            prefetch_queue_depth=prefetch_queue_depth,
            num_threads=num_threads
        )

    def _load_h5_file_paths(self, path: Path):
        """
        Given a path, it loads the abs filepath for all .h5 files in the folder.
        It retuns a dict mapping year to filepath
        """
        assert isinstance(path, Path), f"'{path}' is not a Path object"
        assert path.is_dir(), f"'{path}' does not exist"
        # Results in a dictionary with year as key and path as value
        return {int(path.stem): str(path) for path in sorted(path.glob("????.h5"))}        

    def _load_statistics(self, path: Path):
        if self.cfg.toggles.data.fix_sst_data:
            path_means = path / "global_means_with_sst_fix.npy"
            path_stds = path / "global_stds_with_sst_fix.npy"
        else:
            path_means = path / "global_means.npy"
            path_stds = path / "global_stds.npy"

        if path_means.exists() or path_stds.exists():    
            means = torch.from_numpy(np.load(path_means)).to(dtype=torch.float32) # shape: (1, 21, 1, 1)
            stds = torch.from_numpy(np.load(path_stds)).to(dtype=torch.float32) # shape: (1, 21, 1, 1)
        else:
            warnings.warn(f"Statistics files not found in {path}, setting mean=0.0 and std=1.0")
            means = torch.zeros((1, 21, 1, 1), dtype=torch.float32)
            stds = torch.ones((1, 21, 1, 1), dtype=torch.float32)

        if not self.cfg.toggles.data.include_sst_channel:
            means = means[:, :-1]
            stds = stds[:, :-1]

        return means, stds

    def _load_static_data(self, path: Path):
        # land-sea mask
        path_lsm = path / "land_sea_mask.nc"
        assert path_lsm.exists(), f"Land-sea mask file not found in {path_lsm}"
        lsm = torch.as_tensor(nc.Dataset(path_lsm)["lsm"][:])
        lsm = (lsm - lsm.mean()) / lsm.std()
        # (1, 721, 1440) -> (1, 1, 721, 1440) -> (num_steps, 1, 721, 1440)
        lsm = lsm.unsqueeze(0).expand(self.num_steps, -1, -1, -1)
        if self.cfg.toggles.data.fix_data_centering:
            lsm = fix_latitude_alignment(lsm)
        
        # geo-potential
        path_z = path / "geopotential.nc"
        assert path_z.exists(), f"Geopotential file not found in {path_z}"
        z = torch.as_tensor(nc.Dataset(path_z)["z"][:])
        z = (z - z.mean()) / z.std()
        # (1, 721, 1440) -> (1, 1, 721, 1440) -> (num_steps, 1, 721, 1440)
        z = z.unsqueeze(0).expand(self.num_steps, -1, -1, -1)
        if self.cfg.toggles.data.fix_data_centering:
            z = fix_latitude_alignment(z)

        return dali.types.Constant(
            torch.cat([lsm, z], dim=1)
        )
    
    def _generate_spatial_info(self):
        _sample_height = self.cfg.dataset.sample.height
        _sample_width = self.cfg.dataset.sample.width

        # cos latitudes
        cos_lat = torch.cos(torch.deg2rad(self.model.latitudes)) # shape: (721,)
        cos_lat = (cos_lat - cos_lat.mean()) / cos_lat.std()
        # (721,) -> (1, 1, 721, 1) -> (num_steps, 1, 721, 1440)
        cos_lat = cos_lat.view(1, 1, -1, 1).expand(self.num_steps, 1, -1, _sample_width)
        
        # sin longitudes
        sin_lon = torch.sin(torch.deg2rad(self.model.longitudes)) # shape: (1440,)
        sin_lon = (sin_lon - sin_lon.mean()) / sin_lon.std()
        # (721,) -> (1, 1, 1, 1440) -> (num_steps, 1, 721, 1440)
        sin_lon = sin_lon.view(1, 1, 1, -1).expand(self.num_steps, 1, _sample_height, -1)
        
        # cos longitudes
        cos_lon = torch.cos(torch.deg2rad(self.model.longitudes)) # shape: (1440,)
        cos_lon = (cos_lon - cos_lon.mean()) / cos_lon.std()
        # (721,) -> (1, 1, 1, 1440) -> (num_steps, 1, 721, 1440)
        cos_lon = cos_lon.view(1, 1, 1, -1).expand(self.num_steps, 1, _sample_height, -1)

        return dali.types.Constant(
            torch.cat([cos_lat, sin_lon, cos_lon], dim=1),
            dtype=dali.types.FLOAT
        )

    def _create_pipeline(self, seed, iterator, prefetch_queue_depth, num_threads):
        pipe = dali.Pipeline(
            batch_size=1,
            py_start_method="spawn",
            prefetch_queue_depth=prefetch_queue_depth,
            num_threads=num_threads,
            device_id=DM.device().index,
            set_affinity=True,
        )

        with pipe:
            source = ERA5DaliExternalSource(
                cfg=self.cfg,
                num_samples=len(self),
                file_paths=self.file_paths,
                num_steps=self.num_steps,
                seed=seed,
                iterator=iterator
            )

            (
                epoch_idx, 
                idx_in_epoch,
                global_sample_id,
                timestamps,
                temporal_features,
                data
            ) = dali.fn.external_source(
                source,
                num_outputs=6,
                parallel=True,
                batch=False
            )

            # normalize data
            data = dali.fn.normalize(data, mean=self.means, stddev=self.stds)

            # other channels
            other_channels = list()
            if self.cfg.toggles.model.include_static_data:
                other_channels.append(self.static_data)
            if self.cfg.toggles.model.include_spatial_info:
                other_channels.append(self.spatial_info)
            if self.cfg.toggles.model.include_solar_radiation:
                other_channels.append(
                    cos_zenith_angle(
                        timestamps,
                        latlon=self.map_latlon_to_grid
                    ) / .55  # standardization 
                )
            if self.cfg.toggles.model.include_temporal_info:
                temporal_features_template = dali.fn.ones(
                    shape=(1, 1, self.cfg.dataset.sample.height, self.cfg.dataset.sample.width),
                    dtype=dali.types.FLOAT
                )
                other_channels.append(
                    dali.fn.reshape(
                        temporal_features,
                        shape=(self.num_steps, 4, 1, 1)
                    ) * temporal_features_template
                )
            
            # concatenate, move, split and return
            data = dali.fn.cat(data, *other_channels, axis=1)
            data = data.gpu()
            
            pipe.set_outputs(epoch_idx, idx_in_epoch, global_sample_id, data, timestamps)
        return pipe

    def __iter__(self):
        self.pipe.reset()

        _pipeline = dali_pth.DALIGenericIterator([self.pipe], [
            "epoch_idx", "idx_in_epoch", "global_sample_id", "data", "timestamps"
        ])

        for sample in _pipeline:
            sample = sample[0] # remove batch dimension, as batch_size is always 1 in this project
            sample["data"] = sample["data"][0].to(dtype=self.model.dtype())

            # Prepare sample for utilization.
            # - batch_size is 1, no need to return that dimension
            # - dtype and device can be set right away (dtype in particular can't be done through the Dali pipeline)
            # - for the sample info, no need to return tensors, just the values
            yield {
                "epoch_idx": sample["epoch_idx"].item(),
                "idx_in_epoch": sample["idx_in_epoch"].item(),
                "global_sample_id": sample["global_sample_id"].item(),
                "data": sample["data"],
                "timestamps": sample["timestamps"][0].tolist()
            }

    def __len__(self):
        if self.cfg.toggles.data.fix_december_gap:
            return len(self.file_paths) * self.cfg.dataset.samples_per_file - self.num_steps + 1
        else:
            return len(self.file_paths) * (self.cfg.dataset.samples_per_file - self.num_steps) + 1

    def next_sample(iterator):
        """
        Prepares a ERA5HDF5Datapipe sample for utilization.
        """
        sample = next(iterator)[0] # remove batch dimension, as batch_size is always 1 in this project
        sample["data"] = sample["data"][0].to(dtype=dtype)

        # Prepare sample for utilization.
        # - batch_size is 1, no need to return that dimension
        # - dtype and device can be set right away (dtype in particular can't be done through the Dali pipeline)
        # - for the sample info, no need to return tensors, just the values
        return {
            "epoch_idx": sample["epoch_idx"].item(),
            "idx_in_epoch": sample["idx_in_epoch"].item(),
            "global_sample_id": sample["global_sample_id"].item(),
            "data": sample["data"],
            "timestamps": sample["timestamps"][0].tolist()
        }

class ERA5DaliExternalSource:
    def __init__(
        self,
        cfg,
        num_samples: int,
        file_paths: list,
        num_steps: int,
        seed: int,
        iterator: dict,
    ):
        self.cfg = cfg
        self.num_samples = num_samples
        self.num_steps = num_steps

        self.file_paths = file_paths
        self.files = None
        self.first_year = min(self.file_paths.keys())

        if self.cfg.toggles.data.fix_december_gap:
            self.samples_per_file = self.cfg.dataset.samples_per_file
        else:
            self.samples_per_file = self.cfg.dataset.samples_per_file - self.num_steps

        # Build indicates for distributed training
        self.seed = seed
        self.initial_epoch_idx = iterator["initial_epoch_idx"]
        self.initial_sample_idx = iterator["initial_sample_idx"]

        self.build_idx_map(epoch_idx=self.initial_epoch_idx)

    def build_idx_map(self, epoch_idx=0):
        idx_map = np.arange(self.num_samples, dtype=np.uint16)
        np.random.default_rng(seed=self.seed + epoch_idx).shuffle(idx_map)
        self.idx_map = np.array_split(idx_map, DM.world_size())[DM.rank()]
        self.idx_map_length = len(self.idx_map)

    def open_files(self):
        return {
            year: h5py.File(path, "r", libver="latest", swmr=True) 
            for year, path in self.file_paths.items()
        }

    def parse_sample_info(self, sample_info: dali.types.SampleInfo):
        epoch_idx = sample_info.epoch_idx + self.initial_epoch_idx
        idx_in_epoch = sample_info.idx_in_epoch + self.initial_sample_idx

        if idx_in_epoch >= self.idx_map_length:
            self.initial_sample_idx = 0 # in the next epoch, start from the beginning!
            self.build_idx_map(epoch_idx=epoch_idx + 1) # prepare map for the next epoch
            raise StopIteration()

        # Effective reading index
        global_sample_idx = self.idx_map[idx_in_epoch]
        year = self.first_year + global_sample_idx // self.samples_per_file
        idx_start = global_sample_idx % self.samples_per_file
        idx_end = idx_start + self.num_steps

        return epoch_idx, idx_in_epoch, year, idx_start, idx_end

    def extract_data(self, idx_start, idx_end, year):
        # Extract data
        if self.cfg.toggles.data.fix_december_gap:
            if idx_end < self.samples_per_file:
                data = torch.from_numpy(self.files[year]["fields"][idx_start:idx_end])
            else:
                data_left = self.files[year]["fields"][idx_start:]
                data_right = self.files[year+1]["fields"][:idx_end - self.samples_per_file]
                data = torch.from_numpy(np.concatenate([data_left, data_right], axis=0))
        else:
            data = torch.from_numpy(self.files[year]["fields"][idx_start:idx_end])
        
        # Fix sst by filling missing data with t2m data
        if self.cfg.toggles.data.include_sst_channel and self.cfg.toggles.data.fix_sst_data:
            data[:, 20] = torch.where(data[:, 20] == -32767.0, data[:, 2], data[:, 20])

        # Fix latitude alignment: dataset is from 0 to 360 (Europe on the left), but should be -180 to 180 (Europe at the centre)
        if self.cfg.toggles.data.fix_data_centering:
            data = fix_latitude_alignment(data)

        if not self.cfg.toggles.data.include_sst_channel:
            data = data[:, :-1].contiguous()

        return data

    def extract_timestamps(self, idx_start, idx_end, year):
        timestamps = list()
        temporal_features = list()
        for idx in range(idx_start, idx_end):
            _date = datetime(year, 1, 1) + timedelta(hours=6) * idx
            timestamps.append(_date.timestamp())

            if self.cfg.toggles.data.fix_temporal_info:
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
            else:
                # Calculate the adjusted time index
                adjusted_time_idx = idx_start % self.samples_per_file
                # Compute hour of the year and its decomposition into day of year and time of day
                hour_of_year = adjusted_time_idx * 6
                day_of_year = hour_of_year // 24
                time_of_day = hour_of_year % 24
                # Normalize to the range [0, pi/2]
                normalized_day_of_year = torch.tensor(
                    (day_of_year / 365) * (np.pi / 2), 
                    dtype=torch.float32, 
                    #device=DM.device()
                )
                normalized_time_of_day = torch.tensor(
                    (time_of_day / (24 - 6)) * (np.pi / 2),
                    dtype=torch.float32,
                    #device=DM.device(),
                )
                temporal_features.append([
                    np.sin(normalized_day_of_year),
                    np.cos(normalized_day_of_year),
                    np.sin(normalized_time_of_day),
                    np.cos(normalized_time_of_day)
                ])
        return timestamps, temporal_features

    def __call__(self, sample_info: dali.types.SampleInfo):
        if self.files is None:
            self.files = self.open_files()

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
