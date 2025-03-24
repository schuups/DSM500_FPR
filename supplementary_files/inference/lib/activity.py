from lib.activity_base import ActivityBase

from pathlib import Path
import torch
import h5py

class Activity(ActivityBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
        self.model = None
        self.sample = None

        self.reanalysis = None
        self.forecast = None
        self.climatology = None

        self.metrics = dict()

    def load_model(self):
        if self.model_type == "gc-baseline":
            self.model = self._get_model_for_gc_baseline()
        elif self.model_type == "fcn":
            self.model = self._get_model_for_fcn()
        else:
            raise NotImplementedError()

    def load_sample(self):
        if self.model_type == "gc-baseline":
            self.sample, means, stds = self._get_sample_for_gc_baseline()
            self.means = means.to(dtype=torch.bfloat16, device=self.device)
            self.stds = stds.to(dtype=torch.bfloat16, device=self.device)
        elif self.model_type == "fcn":
            self.sample, means, stds = self._get_sample_for_fcn()
            self.means = torch.tensor(means, dtype=torch.float32, device=self.device)
            self.stds = torch.tensor(stds, dtype=torch.float32, device=self.device)
        else:
            raise NotImplementedError()        

    def load_climatology(self):
        assert self.model is not None, "Model not loaded yet."
        assert self.sample is not None, "Sample not loaded yet."

        # Import modulus-baseline code
        if self.model_type != "fcn" and self.model.cfg.toggles.data.fix_sst_data:
            climatology_file_path = "/iopsstor/scratch/cscs/stefschu/DSM500_FPR/data/FCN_ERA5_data_v0/climatology_with_sst_fix.h5"
        else:
            climatology_file_path = "/iopsstor/scratch/cscs/stefschu/DSM500_FPR/data/FCN_ERA5_data_v0/climatology.h5"

        index_start = self.dataset_initial_condition_i
        index_end = index_start + self.inference_rollout_steps + 1

        with h5py.File(climatology_file_path, "r") as f:
            if self.model_type != "fcn" and self.model.cfg.toggles.data.include_sst_channel:
                self.climatology = torch.tensor(f["climatology"][index_start:index_end], dtype=torch.bfloat16, device=self.device)
            else:
                self.climatology = torch.tensor(f["climatology"][index_start:index_end, :20], dtype=torch.bfloat16, device=self.device)

        if self.model_type == "fcn":
            self.climatology = self.climatology.to(dtype=torch.float32)

    def run_inference(self):
        if self.model_type == "gc-baseline":
            reanalysis, forecast = self._run_inference_for_gc_baseline()
        elif self.model_type == "fcn":
            reanalysis, forecast = self._run_inference_for_fcn()
        else:
            raise NotImplementedError()

        # Denormalize
        self.reanalysis = reanalysis * self.stds + self.means
        self.forecast = forecast * self.stds + self.means

    def _get_slice(self, var, region):
        reanalysis = self.reanalysis
        forecast = self.forecast
        climatology = self.climatology

        # If data is not centered, center it now
        if self.model_type == "fcn" or self.model.cfg.toggles.data.fix_data_centering:
            reanalysis = torch.cat((reanalysis[..., 720:], reanalysis[..., :720]), axis=-1)
            forecast = torch.cat((forecast[..., 720:], forecast[..., :720]), axis=-1)
        # Climatology was computed without centering, thus it always need to be centered
        climatology = torch.cat((climatology[..., 720:], climatology[..., :720]), axis=-1)

        # FCN works on 720 latitude levels
        if self.model_type == "fcn":
            climatology = climatology[:, :, :720]

        # Select variable slice
        if var.name != "all":
            if self.channel_name_to_index_map is None:
                self._build_metadata_maps()

            var_index = self.channel_name_to_index_map[var.name]
            reanalysis = reanalysis[:, [var_index]]
            forecast = forecast[:, [var_index]]
            climatology = climatology[:, [var_index]]

        # Select region slice
        n_longitudes = 1440
        n_latitudes = 720 if self.model_type == "fcn" else 721
        limit_left = 0
        limit_right = n_longitudes
        limit_top = 0
        limit_bottom = n_latitudes

        if region.name != "global":
            if hasattr(region, "longitude_range"):
                lon_limit_left = region.longitude_range[0]
                lon_limit_right = region.longitude_range[1]

                limit_left = int(round((lon_limit_left + 180) / 360 * n_longitudes))
                limit_right = int(round((lon_limit_right + 180) / 360 * n_longitudes))

            if hasattr(region, "latitude_range"):
                lat_limit_top = region.latitude_range[0]
                lat_limit_bottom = region.latitude_range[1]

                limit_top = int(round((lat_limit_top + 90) / 180 * n_latitudes))
                limit_bottom = int(round((lat_limit_bottom + 90) / 180 * n_latitudes))

            reanalysis = reanalysis[:, :, limit_top:limit_bottom, limit_left:limit_right]
            forecast = forecast[:, :, limit_top:limit_bottom, limit_left:limit_right]
            climatology = climatology[:, :, limit_top:limit_bottom, limit_left:limit_right]

        return {
            "reanalysis": reanalysis,
            "forecast": forecast,
            "climatology": climatology,
            "slice": {
                "variable": var.name,
                "region": region.name,
                "box": {
                    "top": limit_top,
                    "bottom": limit_bottom,
                    "left": limit_left,
                    "right": limit_right
                }
            }
            
        }

    def compute_metrics(self, variable, region):
        _slice = self._get_slice(variable, region)

        reanalysis = _slice["reanalysis"]
        forecast = _slice["forecast"]
        climatology = _slice["climatology"]
        slice = _slice["slice"]

        # The following will produce a value for each timestep

        # Compute MSE
        rmse = torch.sqrt(torch.square(reanalysis - forecast).mean(dim=(1, 2, 3))).tolist()

        # Compute ACC
        forecast_anomalies = forecast - climatology
        reanalysis_anomalies = reanalysis - climatology
        forecast_anomalies_std = torch.sqrt(torch.sum(torch.square(forecast_anomalies), dim=(1, 2, 3)))
        reanalysis_anomalies_std = torch.sqrt(torch.sum(torch.square(reanalysis_anomalies), dim=(1, 2, 3)))
        acc = torch.sum(forecast_anomalies * reanalysis_anomalies, dim=(1, 2, 3)) / (forecast_anomalies_std * reanalysis_anomalies_std)
        acc = torch.clamp(acc, min=0.0, max=1.0)
        acc = acc.tolist()

        self.metrics[(variable.name, region.name)] = {
            "rmse": rmse,
            "acc": acc,
            "slice": slice
        }

    def destroy(self):
        del self.model
        del self.sample
        del self.reanalysis
        del self.forecast
        del self.climatology
        del self.metrics
        del self.means
        del self.stds
        torch.cuda.empty_cache()