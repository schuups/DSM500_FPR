# SPDX-FileCopyrightText: Copyright (c) 2025 schups@gmail.com (Modifications)
# SPDX-License-Identifier: Apache-2.0

import os
import fcntl
import pickle
import threading
import torch
from warnings import warn
from pathlib import Path

from modulus.utils.distributed_manager import DistributedManager as DM
from modulus.utils.logging import Logger

from threading import Lock
from datetime import timedelta

class Cache:
    _instance = None
    _lock = Lock()

    MESHES = "meshes"

    _file_names = {
        MESHES: "icosahedron_meshes.pickled"
    }

    def __new__(cls, *args, **kwargs):
        raise RuntimeError("Can't be utilized as Cache()")

    @classmethod
    def initialize(cls, dir: str, verbose=False):
        if cls.is_initialized():
            cls.destroy()
            warn("Cache is already initialized. Reinitializing.")

        with cls._lock:
            if cls._instance is None:
                _instance = super().__new__(cls)

                # Check provided dir
                dir = Path(dir).absolute()
                assert dir.exists(), f"Cache directory '{dir}' does not exist"
                assert dir.is_dir(), f"Cache directory '{dir}' is not a valid directory"
                _instance.dir = dir
                # Store verbosity
                _instance.verbose = verbose
                _instance.logger = Logger("cache")
                
                # Set as initialized
                cls._instance = _instance

    @classmethod
    def is_initialized(cls):
        return cls._instance is not None

    @classmethod
    def destroy(cls):
        assert cls.is_initialized(), "Cache must be initialized before use"
        cls._instance = None

    @classmethod
    def log(cls, msg):
        assert cls.is_initialized(), "Cache must be initialized before use"
        if cls._instance.verbose:
            cls._instance.logger.info(msg)

    @classmethod
    def get_file_path(cls, key):
        assert cls.is_initialized(), "Cache must be initialized before use"
        assert key in cls._file_names, f"{key=} is not a valid key"
        return cls._instance.dir / cls._file_names[key]

    @classmethod
    def is_cached(cls, key):
        assert cls.is_initialized(), "Cache must be initialized before use"
        cls.log(f"Checking if '{key}' is cached.")
        file_path = cls.get_file_path(key)
        if not file_path.exists():
            cls.log(f"-> MISS! '{file_path}' does not exist.")
            return False
        else:
            cls.log(f"-> HIT! '{file_path}' exists.")
            return True

    @classmethod
    def store(cls, key, value):
        assert cls.is_initialized(), "Cache must be initialized before use"
        cls.log(f"Storing '{key}' in cache.")
        file_path = self._get_file_path(key)
        assert file_path.exists(), f"{file_path=} already exist"

        DM.dist().barrier()

        if DM.rank() == 0:
            with cls._lock:
                if not file_path.exists(): # Double check
                    with open(file_path, "wb") as f:
                        cls.log(f"-> STORE! File '{file_path}' written.")
                        fcntl.flock(f, fcntl.LOCK_EX)
                        pickle.dump(value, f)
                        f.flush()
                        os.fsync(f.fileno())
                        fcntl.flock(f, fcntl.LOCK_UN)

        DM.dist().barrier()

    @classmethod
    def load(cls, key, guards=[]):
        assert cls.is_initialized(), "Cache must be initialized before use"
        cls.log(f"Loading cache for '{key}'.")

        if not cls.is_cached(key):
            return None

        file_path = cls.get_file_path(key)
        with open(file_path, "rb") as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            value = pickle.load(f)
            fcntl.flock(f, fcntl.LOCK_UN)

        for guard in guards:
            cls.log(f"-> Checking guard '{guard.__class__.__name__}'.")
            guard.check(value)
        
        return value

    @classmethod
    def clear(cls, key):
        assert cls.is_initialized(), "Cache must be initialized before use"
        cls.log(f"Clearing cache for '{key}'.")
        file_path = cls.get_file_path(key)
        if os.path.exists(file_path):
            os.remove(file_path)


class MeshesCacheGuard:
    def __init__(self, list_length):
        self.list_length = list_length

    def check(self, value):
        assert len(value) == self.list_length, f"Expected {self.list_length} elements, got {len(value)}"
