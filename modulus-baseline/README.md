# Setup notes

The baseline starts from the latest dry-run version.

```sh
cp -r /iopsstor/scratch/cscs/stefschu/DSM500_FPR/modulus-dry-run /iopsstor/scratch/cscs/stefschu/DSM500_FPR/modulus-baseline

cd /iopsstor/scratch/cscs/stefschu/DSM500_FPR/modulus-baseline

rm -rf runs/*

# Rename and adapt as needed (from dry-run references), e.g.:
# - sbatch file
# - hydra config file
```