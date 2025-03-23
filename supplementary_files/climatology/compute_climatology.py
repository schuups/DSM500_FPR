import hydra
from omegaconf import DictConfig

import os
import glob
import numpy as np
import h5py
from pathlib import Path

def list_h5_files(directory):
    files = glob.glob(os.path.join(directory, '**', '*.h5'), recursive=True)
    return [f for f in files if "out_of_sample" not in f]

@hydra.main(version_base="1.3", config_path=None, config_name=None)
def main(cfg: DictConfig):
    h5_files = list_h5_files(cfg.dataset.base_path)
    print(f"Found {len(h5_files)} files")

    # Create a container to hold the climatology 
    climatology = np.zeros((
        cfg.dataset.samples_per_file, 
        cfg.dataset.sample.channels, 
        cfg.dataset.sample.height, 
        cfg.dataset.sample.width
    ), dtype=np.float32)

    temp_container = np.zeros((
        len(h5_files),
        cfg.dataset.sample.channels, 
        cfg.dataset.sample.height, 
        cfg.dataset.sample.width
    ), dtype=np.float32)

    # For each timestep in a year
    for idx_in_year in range(cfg.dataset.samples_per_file):
        print(f"{idx_in_year=} {idx_in_year/cfg.dataset.samples_per_file*100:.2f}%")

        # Open each file and pull that time of the year into the teamp container
        for file_i, file in enumerate(h5_files):
            with h5py.File(file, 'r') as f:
                temp_container[file_i] = f["fields"][idx_in_year]
        
        climatology[idx_in_year] = np.mean(temp_container, axis=0)

    output_path = '/iopsstor/scratch/cscs/stefschu/DSM500_FPR/supplementary_files/climatology/data'
    with h5py.File(f"{output_path}/climatology.h5", "w") as f:
        f.create_dataset("climatology", data=climatology)
    
    print("Done")

if __name__ == '__main__':
    main()