# SPDX-FileCopyrightText: Copyright (c) 2025 schups@gmail.com (Modifications)
# SPDX-License-Identifier: Apache-2.0

import os
import fcntl
import pickle
import threading
from warnings import warn
from modulus.distributed import DistributedManager

_instance = None
_lock = threading.Lock()

class Cache(object):
    MESHES = "meshes"
    _file_names = {
        MESHES: "icosahedron_meshes.pickled"
    }

    def __new__(cls):
        global _instance
        if _instance is None:
            raise RuntimeError("Cache must be initialized using before use.")
        return _instance
    
    @classmethod
    def initialize(cls, dir: str, verbose=False):
        global _instance, _lock
        if not _instance:
            with _lock:
                if not _instance:
                    _instance = super().__new__(cls)

                    # Store distributed manager
                    dist = DistributedManager()
                    assert dist.is_initialized(), "DistributedManager not yet initialized"
                    _instance.dist = dist
                    # Store dir information
                    assert isinstance(dir, str) and len(dir) > 0, "cache_dir must be a string longer than 0"
                    assert os.path.exists(dir) and os.path.isdir(dir), f"{dir=} is not an existing folder"
                    _instance.dir = dir
                    # Store verbosity
                    _instance.verbose = verbose
                    # Set initialized
                    _initialized = True
        else:
            warn("Cache is already initialized, skipping re-initialization.")

    @classmethod
    def is_initialized(cls):
        global _instance
        return _instance is not None

    @classmethod
    def destroy(cls):
        global _instance
        _instance = None

    def _log(self, msg):
        if self.verbose:
            print(f"[CACHE] {msg}")

    def _get_file_path(self, key):
        assert self.is_initialized(), "Cache must be initialized before use"
        assert key in self._file_names, f"{key=} is not a valid key"
        file_name = self._file_names[key]
        file_path = os.path.join(self.dir, file_name)
        return file_path

    def is_cached(self, key):
        self._log(f"Checking if '{key}' is cached.")
        file_path = self._get_file_path(key)
        if not os.path.exists(file_path):
            self._log(f"-> MISS! '{file_path}' does not exist.")
            return False
        else:
            self._log(f"-> HIT! '{file_path}' exists.")
            return True

    def store(self, key, value):
        global _lock

        self._log(f"Storing '{key}' in cache.")
        file_path = self._get_file_path(key)
        assert not os.path.exists(file_path), f"{file_path=} already exist"

        if self.dist.distributed:
            self._log(f"-> Cache write skipped due to multi-rank context.")
            warn("Running in a distributed context: skipping cache creation")
            return
        elif os.path.exists(file_path):
            self._log(f"-> '{file_path}' aready exists, skipping.")
            warn(f"{file_path=} already exists, skipping cache")
            return

        with _lock:
            if not os.path.exists(file_path): # Double check
                with open(file_path, "wb") as f:
                    self._log(f"-> STORE! File '{file_path}' written.")
                    fcntl.flock(f, fcntl.LOCK_EX)
                    pickle.dump(value, f)
                    f.flush()
                    os.fsync(f.fileno())
                    fcntl.flock(f, fcntl.LOCK_UN)
                    return
        self._log(f"-> Unexpectedly reached end of method for file '{file_path}'.")

    def load(self, key, guards=[]):
        self._log(f"Loading cache for '{key}'.")
        file_path = self._get_file_path(key)

        with open(file_path, "rb") as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            value = pickle.load(f)
            fcntl.flock(f, fcntl.LOCK_UN)

        for guard in guards:
            self._log(f"-> Checking guard '{guard.__class__.__name__}'.")
            guard.check(value)
        
        return value

    def clear(self, key):
        self._log(f"Clearing cache for '{key}'.")
        file_path = self._get_file_path(key)

        if os.path.exists(file_path):
            os.remove(file_path)


class MeshesCacheGuard:
    def __init__(self, list_length):
        self.list_length = list_length

    def check(self, value):
        assert len(value) == self.list_length, f"Expected {self.list_length} elements, got {len(value)}"