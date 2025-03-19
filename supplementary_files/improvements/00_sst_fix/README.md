# Setup notes


## Position the new files at the right location
```sh
mkdir -p /iopsstor/scratch/cscs/stefschu/DSM500_FPR/data/FCN_ERA5_data_v0/stats
cd /iopsstor/scratch/cscs/stefschu/DSM500_FPR/data/FCN_ERA5_data_v0/stats

ln -s /iopsstor/scratch/cscs/stefschu/DSM500_FPR/supplementary_files/improvements/00_sst_fix/data/global_means_with_sst_fix.npy global_means_with_sst_fix.npy
ln -s /iopsstor/scratch/cscs/stefschu/DSM500_FPR/supplementary_files/improvements/00_sst_fix/data/global_stds_with_sst_fix.npy global_stds_with_sst_fix.npy


ln -s /iopsstor/scratch/cscs/stefschu/DSM500_FPR/supplementary_files/improvements/00_sst_fix/data/time_diff_std_with_sst_fix.npy time_diff_std_with_sst_fix.npy
```


