import sys
sys.path.append('/iopsstor/scratch/cscs/stefschu/DSM500/github/modulus-a5275d8')

import torch
import numpy as np
import hydra
import h5py
import json
from omegaconf import DictConfig
from collections import defaultdict
from pathlib import Path
from hydra.utils import to_absolute_path
from modulus.distributed import DistributedManager

from modulus.launch.logging import PythonLogger

# FIXIT: Move it under modulus
from inference import Inference

class Plan:
    def __init__(self, cfg):
        self.cfg = cfg
        self.frames = list()
        self.build()

    def build(self):
        frame_id = 0
        channels = range(self.cfg.inference.channels) if isinstance(self.cfg.inference.channels, int) else self.cfg.inference.channels
        for channel in channels:
            for sample in range(self.cfg.inference.samples):
                self.frames.append({
                    "type": "ic",
                    "frame_id": frame_id,
                    "channel": channel,
                    "sample": sample,
                    "filename": f"channel{channel:02}_sample{sample:02}_ic.png"
                })
                frame_id += 1

                for step in range(self.cfg.inference.rollout_steps):
                    self.frames.append({
                        "type": "inference",
                        "frame_id": frame_id,
                        "channel": channel,
                        "sample": sample,
                        "step": step,
                        "filename": {
                            "reanalysis": f"channel{channel:02}_sample{sample:02}_step{step:02}_reanalysis.png",
                            "forecast": f"channel{channel:02}_sample{sample:02}_step{step:02}_forecast.png"
                        }
                    })
                    # Is last step? Add pause
                    # if step == self.cfg.inference.rollout_steps - 1:
                    #     self.frames.append({
                    #         "type": "pause",
                    #         "frame_id": frame_id,
                    #     })
                    frame_id += 1

    def update(self, predicate, update):
        for i, f, in enumerate(self.frames):
            if predicate(f):
                key = list(update.keys())[0]
                assert key not in f, f"Key {key} already exists"
                self.frames[i] |= update

    def save(self):
        with open(Path(to_absolute_path(self.cfg.inference.artifacts_folder)) / f"plan.json", "w") as f:
            json.dump(self.frames, f, indent=4)

class DataGenerator:
    def __init__(self, logger, inference, cfg, plan):
        self.logger = logger
        self.inference = inference
        self.cfg = cfg
        self.plan = plan
        # load model and data pipe
        self.model, self.datapipe = inference.load_model_and_datapipe()
        # load means and stds
        self.means, self.stds = inference.load_means_and_stds()
        # produced data container
        self.container = defaultdict(list)

    def run(self):
        for sample_i, data in enumerate(self.datapipe):
            if sample_i >= self.cfg.inference.samples:
                break
            self.process_sample(sample_i, data)

    def process_sample(self, sample_i, data):
        ic = data[0]["input"]
        reanalysis = data[0]["output"]
        global_sample_id = int(data[0]["global_sample_id"].item())
        timestamps = data[0]["timestamps"][0].numpy()
        timestamp_ic, timestamps_rollout = timestamps[0], timestamps[1:]
        _means = self.means.squeeze()
        _stds = self.stds.squeeze()

        # Update ic frames
        channels = range(self.cfg.inference.channels) if isinstance(self.cfg.inference.channels, int) else self.cfg.inference.channels
        for channel in channels:
            p = lambda f: f["type"] == "ic" and f["sample"] == sample_i and f["channel"] == channel
            self.plan.update(p, {
                "global_sample_id": int(global_sample_id),
                "timestamp": int(timestamp_ic),
                "vmin": ic[0, 0, channel].min().item() * _stds[channel] + _means[channel],
                "vmax": ic[0, 0, channel].max().item() * _stds[channel] + _means[channel],
            })
        
        ic = ic.to(dtype=self.inference.dtype)
        reanalysis = reanalysis.to(dtype=self.inference.dtype)
        forecast = reanalysis.clone()
        with torch.no_grad():
            next_ic = ic
            for step in range(self.cfg.inference.rollout_steps):
                _forecast = self.model(next_ic)
                forecast[0, step, :21] = _forecast
                next_ic = forecast[0, step]

                # Update inference frames
                channels = range(self.cfg.inference.channels) if isinstance(self.cfg.inference.channels, int) else self.cfg.inference.channels
                for channel in channels:
                    p = lambda f: f["type"] == "inference" and f["sample"] == sample_i and f["channel"] == channel and f["step"] == step
                    self.plan.update(p, {
                        "global_sample_id": int(global_sample_id),
                        "timestamp": int(timestamps_rollout[step]),
                        "vmin": min(
                            reanalysis[0, step, channel].min().item(),
                            forecast[0, step, channel].min().item(),
                        ) * _stds[channel] + _means[channel],
                        "vmax": max(
                            reanalysis[0, step, channel].max().item(),
                            forecast[0, step, channel].max().item(), 
                        ) * _stds[channel] + _means[channel],
                    })

        # Denormalize
        ic = ic[0, :, :self.cfg.output_channels]
        ic = ic.to(dtype=torch.float32).cpu().numpy()
        ic = ic * self.stds + self.means

        reanalysis = reanalysis[0, :, :self.cfg.output_channels]
        reanalysis = reanalysis.to(dtype=torch.float32).cpu().numpy()
        reanalysis = reanalysis * self.stds + self.means

        forecast = forecast[0, :, :self.cfg.output_channels]
        forecast = forecast.to(dtype=torch.float32).cpu().numpy()
        forecast = forecast * self.stds + self.means

        self.container["ic"].append(ic)
        self.container["reanalysis"].append(reanalysis)
        self.container["forecast"].append(forecast)

        self.logger.info(f"Sample {sample_i+1} processed")

    def save(self):
        # Save inference samples container
        self.inference.container_file_path.unlink(missing_ok=True)
        with h5py.File(self.inference.container_file_path, "w") as f:
            for key, value in self.container.items():
                f.create_dataset(key, data=np.array(value))
        self.logger.info("Inference samples container saved. Done.")

@hydra.main(version_base="1.3", config_path=None, config_name=None)
def main(cfg: DictConfig):
    logger = PythonLogger("main")
    logger.file_logging()
    DistributedManager.initialize()

    inference = Inference(cfg, logger)
    assert inference.world_size == 1, "This stage of Inference is not supposed to be run in distributed mode"
    
    if inference.container_file_path.exists() and inference.plan_file_path.exists():
        logger.info("Stage 1 files already exists. Skipping.")
        return

    # Load n samples
    for sample_i, data in enumerate(self.datapipe):
        if sample_i >= self.cfg.inference.samples:
            break

    # Run inference steps

    # Store inference samples to file, to be used on stage 2 for visualization rendering

    # Build cohesive plan


    plan = Plan(cfg)
    data_generator = DataGenerator(logger, inference, cfg, plan)
    data_generator.run()
    data_generator.save()
    plan.save()

    logger.info("Done")

if __name__ == "__main__":
    main()

    # Supress [rank0]:[W303 12:53:55.312091090 ProcessGroupNCCL.cpp:1262] Warning: WARNING: process group has NOT been destroyed before we destruct ProcessGroupNCCL. On normal program exit, the application should call destroy_process_group to ensure that any pending NCCL operations have finished in this process. In rare cases this process can exit before this point and block the progress of another member of the process group. This constraint has always been present,  but this warning has only been added since PyTorch 2.4 (function operator())
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()

    # Suppress Exception ignored in atexit callback: <function dump_compile_times at 0x400186acc1f0>
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    import os
    os.environ["TORCH_COMPILE_DEBUG"] = "0"
    import atexit
    import torch._dynamo.utils
    atexit.unregister(torch._dynamo.utils.dump_compile_times)

