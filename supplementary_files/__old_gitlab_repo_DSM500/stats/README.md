# Reproducitibility info

```sh
1. srun --environment=/iopsstor/scratch/cscs/stefschu/DSM500/gitlab/env_arm64.toml --job-name gtest --partition debug --pty bash
2. cd /iopsstor/scratch/cscs/stefschu/DSM500/gitlab/stats
3. jupyter lab --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.ip='0.0.0.0' --NotebookApp.port=8888 --NotebookApp.open_browser=False --notebook-dir=$(pwd)

... run computations + check results

cd /iopsstor/scratch/cscs/stefschu/DSM500/data
ln -s /iopsstor/scratch/cscs/stefschu/DSM500/gitlab/stats/global_means_no_sst_fix.npy global_means.npy
ln -s /iopsstor/scratch/cscs/stefschu/DSM500/gitlab/stats/global_stds_no_sst_fix.npy global_stds.npy
ln -s /iopsstor/scratch/cscs/stefschu/DSM500/gitlab/stats/time_diff_mean_new.npy time_diff_mean.npy
ln -s /iopsstor/scratch/cscs/stefschu/DSM500/gitlab/stats/time_diff_std_new.npy time_diff_std.npy
```