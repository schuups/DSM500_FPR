# Inference

## Aim
Results of different nature (e.g. metrics, images) need to be collected following a fixed protocol for different models (e.g. GraphCast and FourCastNet) and versions of models (GraphCast baseline, GraphCast improved).

## Goal
Enable fair and robust results comparison.

## Design

### Component 1: Inference Engine script
1. It receives a list of models to be measured (defined by files and config file paths, and dataloader specific configurations) and for each of them also a list of checkpoint file paths.
2. It loops through those checkpoints by:
    - Loading the checkpoint and preparing the model for inference
    - Loading the out-of-sample data
    - Looping through the out-of-sample data, at interval of 29 temporal steps for a total of 49 of them (out of sample sample id: 0, 28, 57, ...) resulting on 49 initial conditions on which to initialise inference rollouts.
    - For each such initial condition, a inference rollout is executed.
    - A hook system allow to keep other logic separated, e.g. data transformation (e.g. to adapt the dataloader output to the model input), for metrics computation and collection and for images production
3. It supports hooks on the following points:
    - After the data is loaded from the dataloader and before is fed into the model
    - After the model has produced an output
4. To better utilise the compute nodes capabilities (4 GPUs) it supports parallelization.
    - A list of activities is made at the beginning of the execution. This is then split by world_size and each rank processes only its own activities. No communication among these ranks is necessary.
    - To cope with the 30 minutes runtime limit, a caching meachanism is implmented to allow for the skipping of previously completed activities. Care is paid to only considered done activities that were truly completed.
    - Each activity is defined by:
        (model version, model checkpoint, out-of-sample sample id for the initial condition)

## Guide

### Execute inference with
```sh
python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 run_inference.py
```