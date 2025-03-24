import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from lib.activity import Activity
from lib.engine import Engine
from lib.utils import get_logger

def build_activities(cfg):
    models = cfg.models
    data = cfg.data
    activities = []
    for model in models:
        for weights in model.weights:
            for ic_i in data.initial_conditions_idx:
                activities.append(
                    Activity(
                        id=len(activities),
                        inference_rollout_steps=cfg.inference.rollout_steps,

                        model_name=model.name,
                        model_type=model.type,
                        model_code_path=model.code_path,
                        model_config_path=model.config_path,

                        weights_file_path=weights,

                        dataset_metadata=data.metadata,
                        dataset_file_path=data.file_path,
                        dataset_initial_condition_i=ic_i,
                    )
                )
    return activities

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    engine = Engine(cfg=cfg)
    logger = get_logger(engine.rank, engine.world_size)

    # Build activities list
    activities = build_activities(cfg)
    # Select activities for this rank
    local_activities = activities[engine.rank::engine.world_size]
    local_activities_count = len(local_activities)
    logger.info(f"Initialized {len(activities)} total activities, {local_activities_count} assigned to this rank")

    # Run activities
    for activity_i, activity in enumerate(local_activities):
        if activity.is_already_computed():
            logger.info(f"Activity {activity.id} already computed, skipping")
            continue

        logger.info(f"Processing activity {activity.id} ...")
        engine.run(activity)

        # Store results
        activity.save_results()
        activity.destroy()

        logger.info(f"Activity {activity.id} completed (Local progress: {(activity_i+1)/local_activities_count:.2%})")

        if activity_i >= 5:
            break

    logger.info("All activities completed")

if __name__ == "__main__":
    main()
