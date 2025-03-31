import os
import wandb

import hydra
from omegaconf import DictConfig

@hydra.main(version_base="1.3", config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    rank = int(os.environ.get("LOCAL_RANK"))
    # world_size = int(os.environ.get("WORLD_SIZE"))

    if rank == 0:
        wandb.init()
    
    import random
    # generate a random number between 0 and 1
    random_number = random.random()

    if rank == 0:
        wandb.log({"loss": random_number})

    print(f"[RANK {rank}] hydra config: {cfg}")

if __name__ == "__main__":
    main()

    

    
