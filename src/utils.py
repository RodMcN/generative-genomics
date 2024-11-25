import os
import logging
import pickle
from pathlib import Path
import uuid
from typing import Iterable
import cupy as cp
import torch
import gc


log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format='%(levelname)s: [%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def get_logger(name):
    return logging.getLogger(name)

logger = get_logger(__name__)

class StashableVar:
    """
    Wrapper for a variable that can be saved to disk and loaded later when needed.
    Provides an easy way to dump large variables to disk when not in use and reload later.
    Can save memory when using large arrays
    """
    def __init__(self, value, name=None, directory="."):
        if not name:
            name = str(uuid.uuid4())
        self.name = name
        self._value = value
        self._filename = Path(directory, f"{name}.pkl")
        self._stashed = False
        self._value_str = str(self._value)
    
    
    def __str__(self) -> str:
        if self._stashed:
            return f"Stashed variable: {self._value_str}"
        return str(self._value)
    
    
    def stash(self):
        if self._stashed:
            return
        try:
            with open(self._filename, "wb") as f:
                pickle.dump(self._value, f)
            self._value = None
            self._stashed = True
        except (IOError, pickle.PickleError) as e:
            raise RuntimeError(f"Failed to stash {self.name}: {e}")
        return self
    
    
    def load(self):
        if not self._stashed:
            return self._value
        try:
            with open(self._filename, "rb") as f:
                self._value = pickle.load(f)
            self._stashed = False
            return self._value
        except (IOError, pickle.PickleError) as e:
            raise RuntimeError(f"Failed to load {self.name}: {e}")

    
    def load_once(self):
        if not self._stashed:
            return self._value
        with open(self._filename, "rb") as f:
            return pickle.load(f)
    
    def delete_stash(self):
        if self._filename.exists():
            self._filename.unlink()
        self._stashed = False


    # def __del__(self):
    #     self.delete_stash()

    
    def is_stashed(self):
        return self._stashed


    @staticmethod
    def stash_multiple(vars: Iterable['StashableVar']):
        for var in vars:
            if not isinstance(var, StashableVar):
                raise TypeError(f"Expected StashableVar, got {type(var).__name__}")
            var.stash()


def clear_gpu_memory():
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()
