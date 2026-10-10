"""Volume I/O, intensity normalisation, histogram matching, padding and patch / latent-crop datasets."""
import csv
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

LANDMARK_PCTS = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]


# ----------------------------------------------------------------------------- pair list
def read_pairs(csv_path, split=None):
    """CSV columns: subject, split (train/val/test), t3, t7 [, mask]. 3T must already be registered to the 7T grid."""
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            r = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in r.items()}
            if split is None or r["split"] == split:
                rows.append(r)
    if not rows:
        raise SystemExit(f"no rows with split={split!r} in {csv_path}")
    return rows


def load_nifti(path):
    import nibabel as nib
    img = nib.load(path)
    return np.asarray(img.dataobj, dtype=np.float32), img.affine, img.header.get_zooms()[:3]


# ----------------------------------------------------------------------------- intensity
def normalize01(vol, pct=99.5, fg_frac=0.01):
    """clip to [0, pct-th percentile of the foreground] and scale to [0, 1]. foreground = vol > fg_frac * max."""
    v = np.clip(vol, 0, None)
    fg = v[v > fg_frac * v.max()]
    hi = np.percentile(fg, pct) if fg.size else v.max()
    return np.clip(v / max(hi, 1e-8), 0, 1).astype(np.float32)


def landmarks(vol01, thr):
    fg = vol01[vol01 > thr]
    return np.percentile(fg, LANDMARK_PCTS).astype(np.float64)


def hist_match(vol01, ref_lm, thr):
    """Nyul-Udupa piecewise-linear standardisation of the foreground onto reference landmarks (values in [0,1])."""
    src = np.concatenate([[0.0], landmarks(vol01, thr), [1.0]])
    ref = np.concatenate([[0.0], np.asarray(ref_lm, dtype=np.float64), [1.0]])
    src = np.maximum.accumulate(src + np.arange(len(src)) * 1e-9)  # strictly increasing for np.interp
    out = vol01.copy()
    m = vol01 > thr
    out[m] = np.interp(vol01[m], src, ref)
    return np.clip(out, 0, 1).astype(np.float32)


# ----------------------------------------------------------------------------- padding
def pad_to_multiple(vol, m, value):
    pads = []
    for s in vol.shape:
        t = int(np.ceil(s / m) * m) - s
        pads.append((t // 2, t - t // 2))
    return np.pad(vol, pads, constant_values=value), pads


def unpad(vol, pads):
    sl = tuple(slice(a, vol.shape[i] - b) for i, (a, b) in enumerate(pads))
    return vol[sl]


# ----------------------------------------------------------------------------- prepared data
class Prepped:
    """Reads data_dir/prepped/meta.json written by scripts/prepare_data.py."""

    def __init__(self, data_dir):
        self.dir = os.path.join(data_dir, "prepped")
        self.meta = json.load(open(os.path.join(self.dir, "meta.json")))

    def subjects(self, split):
        return [s for s, m in self.meta["subjects"].items() if m["split"] == split]

    def volume(self, subject, mod, mmap=True):
        """mod in {'3t', '7t'}; values in [-1, 1], padded to a multiple of meta['pad_multiple']."""
        return np.load(os.path.join(self.dir, f"{subject}_{mod}.npy"), mmap_mode="r" if mmap else None)


def _rand_corner(shape, size, rng, center_pool=None):
    if center_pool is not None and len(center_pool[0]):
        i = rng.integers(len(center_pool[0]))
        c = [center_pool[d][i] for d in range(3)]
        return [int(np.clip(c[d] - size[d] // 2, 0, shape[d] - size[d])) for d in range(3)]
    return [int(rng.integers(0, shape[d] - size[d] + 1)) for d in range(3)]


class PatchDataset(Dataset):
    """random paired 3D image patches (for the autoencoder). Returns (x3t, x7t), each (1, P, P, P) in [-1, 1]."""

    def __init__(self, prepped, split, patch, samples_per_epoch, fg_bias=0.8, seed=None, mods=("3t", "7t")):
        self.p, self.subs, self.mods = prepped, prepped.subjects(split), mods
        self.size = [patch] * 3 if isinstance(patch, int) else list(patch)
        self.n, self.fg_bias, self.seed = samples_per_epoch, fg_bias, seed
        self.fg = {}
        for s in self.subs:  # subsampled foreground voxel coordinates for foreground-biased crops
            v = np.asarray(self.p.volume(s, "7t"))
            idx = np.nonzero(v[::4, ::4, ::4] > -0.9)
            self.fg[s] = tuple(a * 4 for a in idx)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng(None if self.seed is None else self.seed + i)
        s = self.subs[rng.integers(len(self.subs))]
        v0 = self.p.volume(s, self.mods[0])
        pool = self.fg[s] if rng.random() < self.fg_bias else None
        c = _rand_corner(v0.shape, self.size, rng, pool)
        sl = tuple(slice(c[d], c[d] + self.size[d]) for d in range(3))
        return tuple(torch.from_numpy(np.array(self.p.volume(s, m)[sl], dtype=np.float32))[None] for m in self.mods)


class LatentCropDataset(Dataset):
    """aligned crops of cached latents. keys: subset of {'z3t', 'z7t', 'zspade'}; each item is a dict of (C, L, L, L) tensors.
    seed=None -> random crops every epoch (train); seed=int -> the same crops every time (validation)."""

    def __init__(self, cache_dir, subjects, keys, crop, samples_per_epoch, seed=None, fg_bias=0.8):
        self.dir, self.subs, self.keys = cache_dir, list(subjects), list(keys)
        self.size = [crop] * 3 if isinstance(crop, int) else list(crop)
        self.n, self.seed, self.fg_bias = samples_per_epoch, seed, fg_bias
        self.fg = {}
        for s in self.subs:
            z = np.load(os.path.join(cache_dir, f"{s}_z7t.npy"), mmap_mode="r")
            e = np.abs(np.asarray(z, dtype=np.float32)).mean(0)
            idx = np.nonzero(e > np.percentile(e, 50))
            self.fg[s] = idx
        missing = [f"{s}_{k}" for s in self.subs for k in self.keys if not os.path.exists(os.path.join(cache_dir, f"{s}_{k}.npy"))]
        if missing:
            raise SystemExit(f"missing cached latents (run the cache scripts first): {missing[:5]} ...")

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng(None if self.seed is None else self.seed + i)
        s = self.subs[rng.integers(len(self.subs))]
        z0 = np.load(os.path.join(self.dir, f"{s}_{self.keys[0]}.npy"), mmap_mode="r")
        pool = self.fg[s] if rng.random() < self.fg_bias else None
        c = _rand_corner(z0.shape[1:], self.size, rng, pool)
        sl = (slice(None),) + tuple(slice(c[d], c[d] + self.size[d]) for d in range(3))
        return {k: torch.from_numpy(np.array(np.load(os.path.join(self.dir, f"{s}_{k}.npy"), mmap_mode="r")[sl], dtype=np.float32))
                for k in self.keys}
