"""Config, seeding, per-epoch CSV logging, mixed precision, checkpoints."""
import contextlib
import csv
import os
import random
import time

import numpy as np
import torch
import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def device_of(cfg):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast(device, enabled):
    """bf16 autocast on CUDA (no grad scaler needed); no-op on CPU"""
    if enabled and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


class EpochLog:
    """one CSV row per epoch (epoch means of every logged quantity), plus a human-readable line on stdout"""

    def __init__(self, path):
        self.path, self.header = path, None
        if os.path.exists(path):
            with open(path) as f:
                first = f.readline().strip()
            self.header = first.split(",") if first else None

    def write(self, row):
        row = {k: (float(v) if torch.is_tensor(v) else v) for k, v in row.items()}
        if self.header is None:
            self.header = list(row)
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow(self.header)
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([row.get(k, "") for k in self.header])
        print("  ".join(f"{k}={v:.5g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)


class Meter:
    """running means of named scalars within one epoch"""

    def __init__(self):
        self.s, self.n = {}, {}

    def add(self, d):
        for k, v in d.items():
            v = float(v)
            self.s[k] = self.s.get(k, 0.0) + v
            self.n[k] = self.n.get(k, 0) + 1

    def means(self, prefix=""):
        return {prefix + k: self.s[k] / self.n[k] for k in self.s}


class Timer:
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.t = time.time()
        return self

    def __exit__(self, *a):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.dt = time.time() - self.t


@contextlib.contextmanager
def fixed_seed(seed):
    """run a block with a fixed RNG seed (CPU + CUDA) and restore the previous RNG state afterwards,
    so validation is identical every epoch without making the training randomness repeat"""
    cpu = torch.random.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(seed)
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


def save_ckpt(path, **kw):
    tmp = path + ".tmp"
    torch.save(kw, tmp)
    os.replace(tmp, path)
