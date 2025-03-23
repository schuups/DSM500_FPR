# Setup and execution

```sh
cd /iopsstor/scratch/cscs/stefschu/DSM500_FPR/supplementary_files/schedule_tuning
```

1. Create sweep yaml definition file 

2. Register it to get sweep id

```sh
> wandb sweep --project DSM500_FPR sweep.yaml
wandb: Creating sweep from: sweep.yaml
wandb: Creating sweep with ID: 28709atw
wandb: View sweep at: https://wandb.ai/schups/DSM500_FPR/sweeps/28709atw
wandb: Run sweep agent with: wandb agent schups/DSM500_FPR/28709atw
```

3. Update the sweep id in the sbatch file
4. Submit the sbatch file
