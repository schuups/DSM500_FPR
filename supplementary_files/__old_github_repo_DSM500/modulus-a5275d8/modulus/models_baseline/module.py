# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2025 schups@gmail.com (Modifications)
# SPDX-License-Identifier: Apache-2.0

import importlib
import inspect
import json
import logging
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict, Union

import torch

import modulus
from modulus.models.meta import ModelMetaData
from modulus.registry import ModelRegistry
from modulus.utils.filesystem import _download_cached, _get_fs


class Module(torch.nn.Module):
    __model_checkpoint_version__ = (
        "0.1.0n"  # Used for file versioning and is not the same as modulus version
    )

    def __new__(cls, *args, **kwargs):
        out = super().__new__(cls)

        # Get signature of __init__ function
        sig = inspect.signature(cls.__init__)

        # Bind args and kwargs to signature
        bound_args = sig.bind_partial(
            *([None] + list(args)), **kwargs
        )  # Add None to account for self
        bound_args.apply_defaults()

        # Get args and kwargs (excluding self and unroll kwargs)
        instantiate_args = {}
        for param, (k, v) in zip(sig.parameters.values(), bound_args.arguments.items()):
            # Skip self
            if k == "self":
                continue

            # Add args and kwargs to instantiate_args
            if param.kind == param.VAR_KEYWORD:
                instantiate_args.update(v)
            else:
                instantiate_args[k] = v

        # Store args needed for instantiation
        out._args = {
            "__name__": cls.__name__,
            "__module__": cls.__module__,
            "__args__": instantiate_args,
        }
        return out

    def __init__(self, meta: Union[ModelMetaData, None] = None):
        super().__init__()
        self.meta = meta
        self.register_buffer("device_buffer", torch.empty(0))

    @property
    def device(self) -> torch.device:
        return self.device_buffer.device

    @property
    def args(self) -> Dict[str, Any]:
        return self._args

    def check_args(self, args: Dict[str, Any]) -> None:
        """
        Utility to compare model loaded from checkpoint with current model
        """
        for k, v in args.items():
            if k not in self.args:
                raise ValueError(f"Key {k} not found in model args")
            if self.args[k] != v:
                raise ValueError(f"Models configuration mismatch. Loaded {v} but expected {self.args[k]}")

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

