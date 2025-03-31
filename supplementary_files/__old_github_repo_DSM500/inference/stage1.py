import sys
sys.path.append('/iopsstor/scratch/cscs/stefschu/DSM500/github/modulus-a5275d8')

import torch

import h5py
import json
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from modulus.distributed import DistributedManager

from collections import defaultdict

from modulus.launch.logging import PythonLogger

from inference import Inference

@hydra.main(version_base="1.3", config_path=".", config_name="inference_config")
def main(cfg: DictConfig):
    DistributedManager.initialize()
    inference = Inference(cfg)
    logger = PythonLogger("main")

    assert inference.world_size == 1, "This stage of Inference is not supposed to be run in distributed mode"

    if inference.container_file_path.exists() and inference.plan_file_path.exists():
        logger.info("Stage 1 files already exists. Skipping.")
        return

    # Load model, data and run inference
    logger.info("Starting inference computation ...")
    samples = dict()
    for case in ["gc_baseline", "gc_improved", "fcn"]:
        model, datapipe = inference.load_model_and_datapipe(case)
        samples[case] = inference.load_samples(case, datapipe)
        samples[case] = inference.run_inference(case, model, samples[case])
        samples[case] = inference.compute_metrics(case, samples[case])
        logger.info(f" Inference computation for '{case}' done.")
    # Store results for later usage
    inference.store_results(samples)
    del samples
    logger.info("Inference and metrics computation completed.")

    # Build video frame plan
    with h5py.File(inference.container_file_path, "r") as f:
        vmin = f["vmin"][()]
        vmax = f["vmax"][()]
        timestamps = f["timestamps"][()]
        global_sample_ids = f["global_sample_ids"][()]

        frame_id = 0
        frames = list()
        for channel in cfg.channels:
            accumulator_mse = defaultdict(list)
            accumulator_acc = defaultdict(list)
            for sample in range(cfg.samples):
                mse_max = float(max(
                    f["gc_baseline/mse"][:, :, channel].max(),
                    f["gc_improved/mse"][:, :, channel].max(),
                    f["fcn/mse"][:, :, channel].max()
                ))
                acc_min = float(min(
                    f["gc_baseline/acc"][:, :, channel].min(),
                    f["gc_improved/acc"][:, :, channel].min(),
                    f["fcn/acc"][:, :, channel].min()
                ))
                for step in range(cfg.rollout_steps + 1):
                    accumulator_mse["gc_baseline"].append(float(f["gc_baseline/mse"][sample, step, channel]))
                    accumulator_mse["gc_improved"].append(float(f["gc_improved/mse"][sample, step, channel]))
                    accumulator_mse["fcn"].append(float(f["fcn/mse"][sample, step, channel]))

                    accumulator_acc["gc_baseline"].append(float(f["gc_baseline/acc"][sample, step, channel]))
                    accumulator_acc["gc_improved"].append(float(f["gc_improved/acc"][sample, step, channel]))
                    accumulator_acc["fcn"].append(float(f["fcn/acc"][sample, step, channel]))

                    frame = {
                        "frame_id": frame_id,
                        "channel": channel,
                        "sample": sample,
                        "step": step,
                        "vmin": float(vmin[sample, step, channel]),
                        "vmax": float(vmax[sample, step, channel]),
                        "timestamp": int(timestamps[sample][step]),
                        "global_sample_id": int(global_sample_ids[sample]),
                        "mse": {
                            "gc_baseline": accumulator_mse["gc_baseline"].copy(),
                            "gc_improved": accumulator_mse["gc_improved"].copy(),
                            "fcn": accumulator_mse["fcn"].copy(),
                            "max": mse_max,
                        },
                        "acc": {
                            "gc_baseline": accumulator_acc["gc_baseline"].copy(),
                            "gc_improved": accumulator_acc["gc_improved"].copy(),
                            "fcn": accumulator_acc["fcn"].copy(),
                            "min": acc_min,
                        },
                    }

                    if step == 0:
                        frames.append(frame | {
                            "files": {
                                "reanalysis": f"channel{channel:02}_sample{sample:02}_step{step:02}_reanalysis.png",
                            }
                        })
                        frame_id += 1
                    else:
                        frames.append(frame | {
                            "files": {
                                "reanalysis": f"channel{channel:02}_sample{sample:02}_step{step:02}_reanalysis.png",
                                "gc_baseline": f"channel{channel:02}_sample{sample:02}_step{step:02}_gc_baseline.png",
                                "gc_improved": f"channel{channel:02}_sample{sample:02}_step{step:02}_gc_improved.png",
                                "fcn": f"channel{channel:02}_sample{sample:02}_step{step:02}_fcn.png",
                            }
                        })
                        frame_id += 1
    
    with open(inference.plan_file_path, "w") as f:
        json.dump(frames, f, indent=4)
    
    logger.info("Frame plan created.")

if __name__ == "__main__":
    main()
