import os
import torch
import hydra
import random
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from itertools import product

from lib.utils import get_logger, set_seed, get_distribution_context, build_activities

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    rank, world_size, device = get_distribution_context()
    logger = get_logger(rank, world_size)

    # Build list of all possible activities, then select those for this rank
    activities = build_activities(cfg)
    # Select activities for this rank
    local_activities = activities[rank::world_size]
    logger.info(f"Initialized {len(activities)} total activities, {len(local_activities)} assigned to this rank")

    # Run activities
    for activity_i, activity in enumerate(local_activities):
        if activity.is_already_computed():
            logger.info(f"Activity {activity.id} already computed, skipping")
            continue

        logger.info(f"Processing activity {activity.id} ({activity.model_name}) ...")
        activity.set_device(device)
        activity.load_model()
        activity.load_sample()
        activity.load_climatology()
        activity.run_inference()

        # Compute metrics
        for variable, region in product(cfg.metrics.slices.variables, cfg.metrics.slices.regions):
            activity.compute_metrics(variable, region)

        # Store results
        activity.save_results()
        logger.info(f"Activity {activity.id} completed (Local progress: {(activity_i+1)/len(local_activities):.2%})")
        
        # Free up memory
        activity.destroy()
        del activity

    logger.info("All activities completed")

if __name__ == "__main__":
    main()
