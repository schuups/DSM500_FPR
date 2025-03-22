# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2025 schups@gmail.com (Modifications)
# SPDX-License-Identifier: Apache-2.0

import os
import torch
from warnings import warn
from datetime import timedelta

class DistributedManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        raise RuntimeError("Can't be utilized as Cache()")

    @classmethod
    def initialize(cls):
        if cls._instance is not None:
            raise RuntimeError("DistributedManager already initialized.")
        if torch.distributed.is_initialized():
            raise RuntimeError("torch.distributed already initialized")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        _instance = super().__new__(cls)
   
        if "SLURM_PROCID" in os.environ and "SLURM_NPROCS" in os.environ and "SLURM_LOCALID" in os.environ:
            _instance._rank = int(os.environ["SLURM_PROCID"])
            _instance._world_size = int(os.environ["SLURM_NPROCS"])
            _instance._local_rank = int(os.environ.get("SLURM_LOCALID"))
        elif "RANK" in os.environ and "WORLD_SIZE" in os.environ and "LOCAL_RANK" in os.environ:
            _instance._rank = int(os.environ["RANK"])
            _instance._world_size = int(os.environ["WORLD_SIZE"])
            _instance._local_rank = int(os.environ.get("LOCAL_RANK"))
        else:
            _instance._rank = 0
            _instance._world_size = 1
            _instance._local_rank = 0
            warn("DistributedManager: running in single process mode!")

        assert _instance._local_rank < torch.cuda.device_count(), f"Local rank {_instance._local_rank} is greater than available devices"
        _instance._device = torch.device(f"cuda:{_instance._local_rank}")
        # Set device for this process and empty cache to optimize memory usage
        torch.cuda.set_device(_instance._device)
        torch.cuda.device(_instance._device)
        torch.cuda.empty_cache()

        # https://pytorch.org/docs/master/notes/cuda.html#id5
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"
        os.environ["NCCL_IB_DISABLE"] = "1"
        os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", os.getenv("SLURM_LAUNCH_NODE_IPADDR", "localhost"))
        os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")

        torch.distributed.init_process_group(
            "nccl",
            rank=_instance._rank,
            world_size=_instance._world_size,
            device_id=_instance._device
        )

        cls._instance = _instance

    @classmethod
    def is_initialized(cls):
        return cls._instance is not None and torch.distributed.is_initialized()

    @classmethod
    def destroy(cls):
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()
        cls._instance = None
    
    @classmethod
    def is_rank_zero(cls):
        assert cls.is_initialized(), "DistributedManager is not initialized."
        return cls._instance._rank == 0

    @classmethod
    def rank(cls):
        assert cls.is_initialized(), "DistributedManager is not initialized."
        return cls._instance._rank

    @classmethod
    def local_rank(cls):
        assert cls.is_initialized(), "DistributedManager is not initialized."
        return cls._instance._local_rank

    @classmethod
    def world_size(cls):
        assert cls.is_initialized(), "DistributedManager is not initialized."
        return cls._instance._world_size

    @classmethod
    def device(cls):
        assert cls.is_initialized(), "DistributedManager is not initialized."
        return cls._instance._device

    @classmethod
    def dist(cls):
        assert cls.is_initialized(), "DistributedManager is not initialized."
        assert torch.distributed.is_initialized(), "torch.distributed is not initialized"
        return torch.distributed

