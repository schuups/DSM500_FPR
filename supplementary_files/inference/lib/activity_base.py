import h5py
import torch
import yaml
import sys
import importlib
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

class ActivityBase:
    def __init__(
        self,
        id,
        inference_rollout_steps,

        model_name,
        model_type,
        model_code_path,
        model_config_path,

        weights_file_path,

        dataset_metadata,
        dataset_file_path,
        dataset_initial_condition_i
    ):
        self.id = id
        self.inference_rollout_steps = inference_rollout_steps

        self.model_name = model_name
        self.model_type = model_type
        self.model_code_path = model_code_path
        self.model_config_path = model_config_path

        self.weights_file_path = weights_file_path

        self.dataset_metadata = dataset_metadata
        self.dataset_file_path = dataset_file_path
        self.dataset_initial_condition_i = dataset_initial_condition_i

        self.file_path_data, self.file_path_metrics = self.get_file_paths()

        self.channel_name_to_index_map = None
        self.device = None

    def set_device(self, device):
        self.device = device

    def _build_metadata_maps(self):
        with open("/iopsstor/scratch/cscs/stefschu/DSM500_FPR/data/FCN_ERA5_data_v0/metadata.yaml", "r") as f:
            metadata = yaml.safe_load(f)
        
        self.metadata = metadata
        self.channel_name_to_index_map = {v["key"]: k for k, v in enumerate(metadata)}

    def _get_model_for_gc_baseline(self):
        assert self.device is not None, "Device not set yet."

        # Import modulus-baseline code
        sys.path.insert(0, self.model_code_path)
        importlib.invalidate_caches()
        from modulus.models.graph_cast_net import GraphCastNet        

        # Prepare model from checkpoint
        cp_dict = torch.load(
            self.weights_file_path, 
            map_location=self.device,
            weights_only=True
        )
        # Build model
        cfg = OmegaConf.create(cp_dict["cfg"])
        model = GraphCastNet(cfg=cfg, device=self.device)
        # Load weights
        weights = cp_dict["model"]
        model.load_state_dict(weights)
        model.eval()

        # Reset imports
        sys.path.pop(0)

        return model

    def _get_model_for_fcn(self):
        assert self.device is not None, "Device not set yet."

        # Import modulus-baseline code
        sys.path.insert(0, self.model_code_path)
        importlib.invalidate_caches()
        from networks.afnonet import AFNONet
        from utils.YParams import YParams

        # Build model
        params = YParams(self.model_config_path, "afno_backbone")
        params['world_size'] = 1
        params['rank'] = 0
        params['local_rank'] = 0
        params['global_batch_size'] = 1
        params['batch_size'] = 1
        params['in_channels'] = np.array(params['in_channels'])
        params['out_channels'] = np.array(params['out_channels'])
        params['N_in_channels'] = len(params['in_channels'])
        params['N_out_channels'] = len(params['out_channels'])
        self.params = params
        model = AFNONet(params).to(self.device)

        # Load weights
        cp_dict = torch.load(
            self.weights_file_path, 
            map_location=self.device,
            weights_only=False
        )

        # Fix keys because it was checkpointed from DDP
        cp_dict["model_state"] = {k.replace("module.", ""): v for k, v in cp_dict["model_state"].items()}

        model.load_state_dict(cp_dict["model_state"])
        model.eval()

        # Reset imports
        sys.path.pop(0)

        return model

    def _get_sample_for_gc_baseline(self):
        assert self.device is not None, "Device not set yet."
        assert self.model is not None, "Model not loaded yet."

        # Import modulus-baseline code
        sys.path.insert(0, self.model_code_path)
        importlib.invalidate_caches()
        from modulus.datapipes.era5_hdf5 import ERA5HDF5Datapipe
        
        # Prepare data
        datapipe = ERA5HDF5Datapipe(
            cfg=self.model.cfg,
            dataset_folder="out_of_sample",
            num_output_steps=self.inference_rollout_steps,
            model=self.model,
            iterator={
                "initial_epoch_idx": 0,
                "initial_sample_idx": self.dataset_initial_condition_i
            },
            prefetch_queue_depth=1,
            num_threads=1,
            
            shuffle=False,
            seed=0, # No use given that shuffle=False
            device=self.device,
            rank=0,
            world_size=1,
        )

        # Get what is needede, then delete pipeline
        means, stds = datapipe.means, datapipe.stds
        sample = next(iter(datapipe))
        del datapipe

        # Reset imports
        sys.path.pop(0)

        return sample, means, stds

    def _get_sample_for_fcn(self):
        assert self.device is not None, "Device not set yet."
        assert self.model is not None, "Model not loaded yet."

        # Import modulus-baseline code
        sys.path.insert(0, self.model_code_path)
        importlib.invalidate_caches()
        from utils.data_loader_multifiles import get_data_loader
        from utils.YParams import YParams

        # Build model
        params = YParams(self.model_config_path, "afno_backbone")
        params['world_size'] = 1
        params['rank'] = 0
        params['local_rank'] = 0
        params['global_batch_size'] = 1
        params['batch_size'] = 1
        params['in_channels'] = np.array(params['in_channels'])
        params['out_channels'] = np.array(params['out_channels'])
        params['N_in_channels'] = len(params['in_channels'])
        params['N_out_channels'] = len(params['out_channels'])

        valid_data_loader, valid_dataset = get_data_loader(params, params.valid_data_path, False, train=False)
        means = np.load(params.global_means_path)[:, params['in_channels']]
        stds = np.load(params.global_means_path)[:, params['in_channels']]

        sample = torch.empty((self.inference_rollout_steps + 1, 20, 720, 1440))
        for i in range(self.inference_rollout_steps + 1):
            data, _ = valid_dataset[self.dataset_initial_condition_i + 1]
            sample[i] = data

        sample = sample.to(device=self.device)

        # Reset imports
        sys.path.pop(0)

        return sample, means, stds

    def _run_inference_for_gc_baseline(self):
        assert self.device is not None, "Device not set yet."
        assert self.model is not None, "Model not loaded yet."
        assert self.sample is not None, "Sample not loaded yet."

        _channels_dataset = self.model.input_channels_count_dataset()
        _channels_generated = self.model.input_channels_count_generated()

        data = self.sample["data"]
        reanalysis, generated = torch.split(data, [_channels_dataset, _channels_generated], dim=1)
        forecasts = torch.empty_like(reanalysis)
        forecasts[0] = reanalysis[0]

        model_input = torch.empty_like(data[0])
        for step_i in range(self.inference_rollout_steps):
            if step_i == 0:
                model_input.copy_(data[0], non_blocking=True)
            else:
                model_input[:_channels_dataset].copy_(forecasts[step_i], non_blocking=True)
                model_input[_channels_dataset:].copy_(generated[step_i], non_blocking=True)
            
            with torch.no_grad():
                output = self.model(model_input)
            
            forecasts[step_i+1] = output

        return reanalysis, forecasts

    def _run_inference_for_fcn(self):
        assert self.device is not None, "Device not set yet."
        assert self.model is not None, "Model not loaded yet."
        assert self.sample is not None, "Sample not loaded yet."

        reanalysis = self.sample.clone()
        forecasts = torch.empty_like(self.sample)
        forecasts[0] = reanalysis[0]

        for step_i in range(1, self.inference_rollout_steps):
            model_input = forecasts[step_i-1]
            
            with torch.no_grad():
                output = self.model(model_input.unsqueeze(0)).squeeze(0)
            
            forecasts[step_i] = output

        return reanalysis, forecasts

    def get_file_paths(self):
        file_name = f"{self.id:05}.npy"

        dir_base = Path("/iopsstor/scratch/cscs/stefschu/DSM500_FPR/cache/inference")
        dir_data = dir_base / "data"
        dir_metrics = dir_base / "metrics"

        dir_data.mkdir(parents=True, exist_ok=True)
        dir_metrics.mkdir(parents=True, exist_ok=True)

        file_path_data = dir_data / file_name
        file_path_metrics = dir_metrics / file_name

        return file_path_data, file_path_metrics

    def is_already_computed(self):
        return self.file_path_data.exists() and self.file_path_metrics.exists()
        
    def save_results(self):
        # Store data
        torch.save({
            "id": self.id,
            "reanalysis": self.reanalysis.to(dtype=torch.float16),
            "forecast": self.forecast.to(dtype=torch.float16),
        }, self.file_path_data.with_suffix(".tmp"))

        # Store metrics
        torch.save({
            "id": self.id,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "code_path": self.model_code_path,
            "config_path": self.model_config_path,
            "weights_file_path": self.weights_file_path,
            "metrics": self.metrics,
            "ic": self.dataset_initial_condition_i,
            "cfg": self.params.params if self.model_type == "fcn" else OmegaConf.to_container(self.model.cfg),
        }, self.file_path_metrics.with_suffix(".tmp"))

        # Move into place
        self.file_path_data.with_suffix(".tmp").rename(self.file_path_data)
        self.file_path_metrics.with_suffix(".tmp").rename(self.file_path_metrics)
    