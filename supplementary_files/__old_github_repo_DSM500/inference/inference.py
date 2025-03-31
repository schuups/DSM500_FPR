import os
import sys
sys.path.append('/iopsstor/scratch/cscs/stefschu/DSM500/github/fourcastnet-93360c1')

import numpy as np
import torch
import json
import h5py

import netCDF4 as nc
from scipy.ndimage import sobel
from pathlib import Path
from hydra.utils import to_absolute_path

from modulus.distributed import DistributedManager

from omegaconf import OmegaConf
from modulus.datapipes.climate.era5_hdf5 import ERA5HDF5Datapipe, fix_latitude_alignment

class Inference:
    def __init__(self, cfg):
        self.cfg = cfg

        # Setup artifacts folders
        self.output_dir = Path(to_absolute_path(self.cfg.artifacts_folder))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir = self.output_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)

        # Load channels metadata
        with open(Path(self.cfg.dataset.base_path) / "metadata.json", "r") as f:
            metadata = json.loads(f.read())
            self.channel_titles = metadata["coords"]["channel_titles"]
            self.channel_tick_format = metadata["coords"]["channel_tick_format"]

        # Distributed context
        if DistributedManager.is_initialized():
            dist = DistributedManager()
            self.rank = dist.rank
            self.world_size = dist.world_size
        elif "LOCAL_RANK" in os.environ and "WORLD_SIZE" in os.environ:
            self.rank = int(os.environ.get("LOCAL_RANK"))
            self.world_size = int(os.environ.get("WORLD_SIZE"))
        else:
            self.rank = int(os.environ.get("SLURM_PROCID"))
            self.world_size = int(os.environ.get("SLURM_NTASKS"))
        self.device = "cuda:0"

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

    def load_model_and_datapipe(self, case):
        if case == "gc_baseline":
            return self._load_gc_baseline()
        elif case == "gc_improved":
            return self._load_gc_improved()
        elif case == "fcn":
            return self._load_fcn()
        else:
            raise ValueError(f"Unknown case: {case}")

    def _load_gc_baseline(self):
        cfg = OmegaConf.load(self.cfg.models.gc_baseline.config_file_path)

        from modulus.utils_new.caching import Cache
        from modulus.models_baseline.graphcast.graph_cast_net import GraphCastNetBaseline
        from modulus.datapipes.climate.era5_hdf5 import ERA5HDF5Datapipe

        Cache.destroy()
        Cache.initialize(dir=cfg.cache_dir)

        model = GraphCastNetBaseline(
            sample_height=cfg.sample.height,
            sample_width=cfg.sample.width,
            sample_channels=cfg.sample.channels,

            include_static_data=cfg.include.static_data,
            include_spatial_info=cfg.include.spatial_info,
            include_temporal_info=cfg.include.temporal_info,
            include_solar_radiation=cfg.include.solar_radiation,

            batch_size=cfg.datapipe.batch_size,
            mesh_level=cfg.mesh_level,
            activation_fn=cfg.activation_fn,
            hidden_dim=cfg.hidden_dim,
            hidden_layers=cfg.hidden_layers,
            aggregation_op=cfg.aggregation_op,
            processor_layers=cfg.processor_layers,
        )
        dtype = torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float32
        model = model.to(dtype=dtype).to(self.device)

        checkpoint = torch.load(
            self.cfg.models.gc_baseline.checkpoint_file_path,
            map_location=self.device,
            weights_only=True
        )
        model.check_args(checkpoint["metadata"])
        model.load_state_dict(checkpoint["model"])
        model.eval()

        # Load datapipe
        datapipe = ERA5HDF5Datapipe(
            model=model,
            dataset_base_path=cfg.dataset.base_path,
            dataset_folder="out_of_sample",
            dataset_samples_per_file=cfg.dataset.samples_per_file,

            num_output_steps=self.cfg.rollout_steps,

            iterator_seed=self.cfg.seed,
            iterator_offset_epoch_idx=0,
            iterator_offset_sample_idx=0,

            num_threads=1,
            prefetch_queue_depth=1
        )

        return model, datapipe

    def _load_gc_improved(self):
        cfg = OmegaConf.load(self.cfg.models.gc_improved.config_file_path)
        
        from modulus.utils_new.caching import Cache
        from modulus.models_baseline.graphcast.graph_cast_net import GraphCastNetBaseline
        from modulus.datapipes.climate.era5_hdf5 import ERA5HDF5Datapipe

        Cache.destroy()
        Cache.initialize(dir=cfg.cache_dir)

        model = GraphCastNetBaseline(
            sample_height=cfg.sample.height,
            sample_width=cfg.sample.width,
            sample_channels=cfg.sample.channels,

            include_static_data=cfg.include.static_data,
            include_spatial_info=cfg.include.spatial_info,
            include_temporal_info=cfg.include.temporal_info,
            include_solar_radiation=cfg.include.solar_radiation,

            batch_size=cfg.datapipe.batch_size,
            mesh_level=cfg.mesh_level,
            activation_fn=cfg.activation_fn,
            hidden_dim=cfg.hidden_dim,
            hidden_layers=cfg.hidden_layers,
            aggregation_op=cfg.aggregation_op,
            processor_layers=cfg.processor_layers,
        )
        dtype = torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float32
        model = model.to(dtype=dtype).to(self.device)

        checkpoint = torch.load(
            self.cfg.models.gc_improved.checkpoint_file_path,
            map_location=self.device,
            weights_only=True
        )
        # FIXME: uncomment this
        # model.check_args(checkpoint["metadata"])
        model.load_state_dict(checkpoint["model"])
        model.eval()

        # Load datapipe
        datapipe = ERA5HDF5Datapipe(
            model=model,
            dataset_base_path=cfg.dataset.base_path,
            dataset_folder="out_of_sample",
            dataset_samples_per_file=cfg.dataset.samples_per_file,

            num_output_steps=self.cfg.rollout_steps,

            iterator_seed=self.cfg.seed,
            iterator_offset_epoch_idx=0,
            iterator_offset_sample_idx=0,

            num_threads=1,
            prefetch_queue_depth=1
        )

        return model, datapipe

    def _load_fcn(self):
        """
        The following is extrapolated from https://github.com/NVlabs/fcn/blob/master/inference/inference.py
        """

        from collections import OrderedDict
        from fourcastnet.utils.YParams import YParams
        from fourcastnet.networks.afnonet import AFNONet
        from fourcastnet.inference.inference import load_model

        # Build model
        params = YParams(self.cfg.models.fcn.config_file_path, "afno_backbone")
        # The parameters required for the model construction are:
        # - patch_size (already present in the yaml file)
        # - N_in_channels
        # - N_out_channels
        # - num_blocks (already present in the yaml file)
        # - means
        # - stds
        params.N_in_channels = len(self.cfg.channels)
        params.N_out_channels = len(self.cfg.channels)
        
        # Load for normalization purposes
        params.means = np.load(params.global_means_path)[0, self.cfg.channels]
        params.stds = np.load(params.global_stds_path)[0, self.cfg.channels]

        torch.backends.cudnn.benchmark = True
        model = AFNONet(params).to(self.device)
        model = load_model(model, params, self.cfg.models.fcn.checkpoint_file_path)
        model = model.to(self.device)

        # Build datapipe. Need to mock a model...
        mocked_model = OrderedDict()
        mocked_model.includes_static_data = False
        mocked_model.includes_spatial_info = False
        mocked_model.includes_temporal_info = False
        mocked_model.includes_solar_radiation = False
        mocked_model.batch_size = 1
        datapipe = ERA5HDF5Datapipe(
            model=mocked_model,
            dataset_base_path=self.cfg.dataset.base_path,
            dataset_folder="out_of_sample",
            dataset_samples_per_file=self.cfg.dataset.samples_per_file,

            num_output_steps=self.cfg.rollout_steps,

            iterator_seed=self.cfg.seed,
            iterator_offset_epoch_idx=0,
            iterator_offset_sample_idx=0,

            num_threads=1,
            prefetch_queue_depth=1
        )

        my_means = datapipe.means.clone()[:, :20, :, :].to("cuda:0")
        my_stds = datapipe.stds.clone()[:, :20, :, :].to("cuda:0")

        their_means = torch.tensor(params.means).to("cuda:0")
        their_stds = torch.tensor(params.stds).to("cuda:0")

        def wrap_model(input):
            """
            Makes the model compatible with the rest of the input and evaluation
            """
            # [1, 20, 720, 1440] <- [1, 1, 20, 721, 1440]
            input = input[..., :720, :]
            # denormalize from my values
            input = input * my_stds + my_means
            # normalise with their values
            input = (input - their_means) / their_stds
            # cast back to float32
            input = input.to(dtype=torch.float32)
            # put US in the middle
            input = fix_latitude_alignment(input)

            with torch.no_grad():
                output = model(input)

            # Change output to be compatible with the rest of the evaluation
            # put EU in the middle
            output = fix_latitude_alignment(output)
            # denormalise from their values
            output = output * their_stds + their_means
            # normalise with my values
            output = (output - my_means) / my_stds
            # cast back to float32
            output = output.to(dtype=torch.float32)
            return output

        return wrap_model, datapipe

    def load_samples(self, case, datapipe):
        data = []
        global_sample_ids = []
        timestamps = []
        
        for sample_i, sample in enumerate(datapipe):
            # dict_keys(['epoch_idx', 'idx_in_epoch', 'global_sample_id', 'data', 'input'])
            # data['data'].shape, data["global_sample_id"].shape, data["timestamps"].shape
            # (torch.Size([1, 25, 31, 721, 1440]),
            #  torch.Size([1]),
            #  torch.Size([1, 25]))
            # type(data['input']), type(data["output"]), type(data["global_sample_id"]), type(data["timestamps"])
            # (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor)
            sample = sample[0]

            data.append(sample["data"])
            global_sample_ids.append(sample["global_sample_id"].item())
            timestamps.append(sample["timestamps"])

            if len(data) == self.cfg.samples:
                break
        
        data = torch.concatenate(data, axis=0)
        global_sample_ids = torch.tensor(global_sample_ids, dtype=torch.int32)
        timestamps = torch.concatenate(timestamps, axis=0)

        # torch.Size([3, 25, 21, 721, 1440]), torch.Size([3, 25, 10, 721, 1440]) <- outputs torch.Size([3, 25, 31, 721, 1440])
        reanalysis, generated = torch.split(data, [datapipe.channels_count, datapipe.generated_channels_count], dim=2)

        # Fix data for the particular case of fcn
        if case == "fcn":
            reanalysis = reanalysis[:, :, :20, :720].cpu()
            generated = generated[:, :, :20, :720].cpu()
        
        # Fix dtype
        if case != "fcn":
            reanalysis = reanalysis.to(dtype=torch.bfloat16).cpu()
            generated = generated.to(dtype=torch.bfloat16).cpu()

        # Store means and stds for later
        if case != "fcn":
            means = datapipe.means.to(dtype=torch.bfloat16)
            stds = datapipe.stds.to(dtype=torch.bfloat16)
        else:
            means = datapipe.means[:, :20]
            stds = datapipe.stds[:, :20]

        # Load climatology
        climatology = torch.zeros(reanalysis.shape, device="cpu", dtype=reanalysis.dtype)
        with h5py.File(Path(self.cfg.dataset.base_path) / "climatology.h5", "r") as f:
            for sample_i, global_sample_id in enumerate(global_sample_ids):
                idx_start = global_sample_id
                idx_end = idx_start + self.cfg.rollout_steps + 1
                if case != "fcn":
                    climatology[sample_i] = torch.tensor(f["climatology"][idx_start:idx_end])
                else:
                    climatology[sample_i] = torch.tensor(f["climatology"][idx_start:idx_end, :20, :720])
        climatology = fix_latitude_alignment(climatology)

        # Prepare forecasts container
        forecast = torch.zeros(reanalysis.shape, device="cpu", dtype=reanalysis.dtype)

        return {
            "means": means,
            "stds": stds,
            "reanalysis": reanalysis,
            "forecast": forecast,
            "generated": generated,
            "global_sample_ids": global_sample_ids,
            "timestamps": timestamps,
            "climatology": climatology
        }

    def run_inference(self, case, model, samples):
        with torch.no_grad():
            for sample_i in range(self.cfg.samples):
                _reanalysis = samples["reanalysis"][sample_i].to(self.device)
                _forecast = samples["forecast"][sample_i].to(self.device)
                _generated = samples["generated"][sample_i].to(self.device)

                for step_i in range(self.cfg.rollout_steps + 1):
                    if step_i == 0:
                        _forecast[0] = _reanalysis[0]
                    else:
                        if case != "fcn":
                            model_input = torch.cat([_forecast[step_i-1], _generated[step_i]], dim=0)
                        else:
                            model_input = _forecast[step_i-1]

                        # e.g., torch.Size([1, 21, 721, 1440]) <- model(torch.Size([31, 721, 1440]))
                        model_output = model(model_input)

                        if case != "fcn":
                            _forecast[step_i] = model_output[0]
                        else:
                            _forecast[step_i] = model_output[0]
                samples["forecast"][sample_i] = _forecast.cpu()
        del model_input, model_output, _reanalysis, _forecast, _generated, samples["generated"]
        torch.cuda.empty_cache()

        return samples

    def compute_metrics(self, case, samples):
        # Denormalise
        samples["reanalysis"] = samples["reanalysis"] * samples["stds"] + samples["means"]
        samples["forecast"] = samples["forecast"] * samples["stds"] + samples["means"]
    
        # MSE
        samples["mse"] = torch.square(samples["forecast"] - samples["reanalysis"]).mean(dim=(-2, -1))

        # ACC
        forecast_anomalies = samples["forecast"] - samples["climatology"]
        reanalysis_anomalies = samples["reanalysis"] - samples["climatology"]
        forecast_anomalies_std = torch.sqrt(torch.sum(torch.square(forecast_anomalies), dim=(-2, -1)))
        reanalysis_anomalies_std = torch.sqrt(torch.sum(torch.square(reanalysis_anomalies), dim=(-2, -1)))
        samples["acc"] = torch.sum(forecast_anomalies * reanalysis_anomalies, dim=(-2, -1)) / (forecast_anomalies_std * reanalysis_anomalies_std)

        # Additionally, store vmin a vmax for the upcoming shared colorbar rendering
        samples["vmin"] = torch.amin(samples["forecast"], dim=(-2, -1))
        samples["vmax"] = torch.amax(samples["forecast"], dim=(-2, -1))

        # No longer needed
        del samples["climatology"]

        return samples
    
    def store_results(self, samples):
        # Consistency checks
        assert torch.equal(samples["gc_baseline"]["global_sample_ids"], samples["gc_improved"]["global_sample_ids"])
        assert torch.equal(samples["gc_baseline"]["global_sample_ids"], samples["fcn"]["global_sample_ids"])
        
        vmin = torch.amin(samples["gc_baseline"]["reanalysis"][..., :20, :, :], dim=(-2, -1))
        vmin = torch.minimum(vmin, samples["gc_baseline"]["vmin"][..., :20])
        vmin = torch.minimum(vmin, samples["gc_improved"]["vmin"][..., :20])
        vmin = torch.minimum(vmin, samples["fcn"]["vmin"])

        vmax = torch.amax(samples["gc_baseline"]["reanalysis"][..., :20, :, :], dim=(-2, -1))
        vmax = torch.maximum(vmax, samples["gc_baseline"]["vmax"][..., :20])
        vmax = torch.maximum(vmax, samples["gc_improved"]["vmax"][..., :20])
        vmax = torch.maximum(vmax, samples["fcn"]["vmax"])

        # Store results
        with h5py.File(self.container_file_path, "w") as f:
            for case in ["gc_baseline", "gc_improved", "fcn"]:
                for key in ["reanalysis", "forecast", "mse", "acc"]:
                    f.create_dataset(f"{case}/{key}", data=np.array(samples[case][key].to(dtype=torch.float32)))
                    #f[f"{case}/{key}"] = samples[case][key].to(torch.float32).numpy()
            
            # f["timestamps"] = samples["gc_baseline"]["timestamps"].numpy()
            # f["global_sample_ids"] = samples["gc_baseline"]["global_sample_ids"].numpy()
            # f["vmin"] = vmin.to(torch.float32).numpy()
            # f["vmax"] = vmax.to(torch.float32).numpy()
            f.create_dataset("timestamps", data=np.array(samples["gc_baseline"]["timestamps"]))
            f.create_dataset("global_sample_ids", data=np.array(samples["gc_baseline"]["global_sample_ids"]))
            f.create_dataset("vmin", data=np.array(vmin))
            f.create_dataset("vmax", data=np.array(vmax))

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
