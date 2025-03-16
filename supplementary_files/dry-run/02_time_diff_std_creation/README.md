# Setup notes


## Prepare files
```sh
cd /iopsstor/scratch/cscs/stefschu/DSM500_FPR/supplementary_files/dry-run/02_time_diff_std_creation

cp /iopsstor/scratch/cscs/stefschu/DSM500_FPR/modulus-a5275d8/examples/weather/graphcast/compute_time_diff_std.py compute_time_diff_std.py
```

## Execute

```sh
sbatch compute_time_diff_std.sbatch
```

## Install files

```sh
cd /iopsstor/scratch/cscs/stefschu/DSM500_FPR/data/dry-run/stats

ln -s /iopsstor/scratch/cscs/stefschu/DSM500_FPR/supplementary_files/dry-run/02_time_diff_std_creation/data/time_diff_std.npy time_diff_std.npy
```
