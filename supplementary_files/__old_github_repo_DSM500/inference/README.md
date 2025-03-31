# Inference

## Stage 1
Purpose:
- Load data from dataloader
- Load multiple models (e.g. baseline graphcast, improved graphcast, FourCastNet)
- Run inference and collect rollout outputs
- Compute metrics
- Store results
- Build video plan (frame by frame, what to show) - in case next stages are executed
- Produce inference report

## Stage 2
Purpose:
- Render globe images

## Stage 3
Purpose:
- Render video frames, incl. metric plots

## Stage 4:
Purpose:
- Assemble video

## Run manually
- Once in the container and with the venv activated:
```sh
python -m torch.distributed.run --nnodes=1 --nproc_per_node=1 stage1.py
python -m torch.distributed.run --nnodes=1 --nproc_per_node=200 stage2.py
python -m torch.distributed.run --nnodes=1 --nproc_per_node=100 stage3.py
python -m torch.distributed.run --nnodes=1 --nproc_per_node=1 stage4.py
```