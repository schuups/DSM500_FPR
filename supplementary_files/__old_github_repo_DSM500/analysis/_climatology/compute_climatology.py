import hydra
from omegaconf import DictConfig

import os
import glob
import numpy as np
import h5py
from pathlib import Path

def list_h5_files(directory):
    return glob.glob(os.path.join(directory, '**', '*.h5'), recursive=True)

@hydra.main(version_base="1.3", config_path=None, config_name=None)
def main(cfg: DictConfig):
    h5_files = list_h5_files(cfg.dataset.base_path)

    climatology = np.zeros((cfg.dataset.samples_per_file, cfg.sample.channels, cfg.sample.height, cfg.sample.width), dtype=np.float32)

    temp_container = np.zeros((len(h5_files), cfg.sample.channels, cfg.sample.height, cfg.sample.width), dtype=np.float32)

    # For each timestep in a year
    for idx_in_year in range(cfg.dataset.samples_per_file):
        print(f"{idx_in_year=} {idx_in_year/cfg.dataset.samples_per_file*100:.2f}%")
        # Open each file and pull that time of the year
        for file_i, file in enumerate(h5_files):
            #print("\t", file_i)
            with h5py.File(file, 'r') as f:
                temp_container[file_i] = f["fields"][idx_in_year]
        climatology[idx_in_year] = np.mean(temp_container, axis=0)

    with h5py.File(Path(cfg.dataset.base_path) / "climatology.h5", "w") as f:
        f.create_dataset("climatology", data=climatology)
    
    print("Done")

if __name__ == '__main__':
    main()