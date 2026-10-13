"""Volume I/O, intensity normalisation, histogram matching, zero padding, patch grids and patch / latent-patch datasets."""
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


# ----------------------------------------------------------------------------- padding and patch grid
def pad_for_patches(vol, patch, stride, value):
    """symmetric constant padding so that every axis is >= patch and (size - patch) is a multiple of stride,
    i.e. a stride-`stride` grid of patch^3 windows covers the volume exactly (no voxel left out at the far border)."""
    pads = []
    for s in vol.shape:
        target = patch if s <= patch else patch + int(np.ceil((s - patch) / stride)) * stride
        t = target - s
        pads.append((t // 2, t - t // 2))
    return np.pad(vol, pads, constant_values=value), pads


def unpad(vol, pads):
    sl = tuple(slice(a, vol.shape[i] - b) for i, (a, b) in enumerate(pads))
    return vol[sl]


def grid_corners(shape, patch, stride):
    """corners of all patch^3 windows on a stride grid; the volume must have been padded with pad_for_patches"""
    axes = []
    for s in shape:
        if (s - patch) % stride:
            raise ValueError(f"axis of size {s} does not fit a {patch}/{stride} patch grid -- pad with pad_for_patches")
        axes.append(range(0, s - patch + 1, stride))
    return np.array([(a, b, c) for a in axes[0] for b in axes[1] for c in axes[2]], dtype=np.int64)


def extract_patches(vol, corners, patch):
    """vol (..., D, H, W) array or tensor -> stacked patches (N, ..., P, P, P)"""
    out = [vol[..., a:a + patch, b:b + patch, c:c + patch] for a, b, c in corners]
    return torch.stack(out) if torch.is_tensor(vol) else np.stack(out)


class PatchMerger:
    """accumulate overlapping patch outputs and average them (each voxel = mean of every patch that covers it)"""

    def __init__(self, shape, patch, device, channels=1):
        self.p = patch
        self.sum = torch.zeros((channels, *shape), device=device)
        self.cnt = torch.zeros((1, *shape), device=device)

    def add(self, patches, corners):
        for x, (a, b, c) in zip(patches, corners):
            sl = (slice(None), slice(a, a + self.p), slice(b, b + self.p), slice(c, c + self.p))
            self.sum[sl] += x.float()
            self.cnt[sl] += 1

    def result(self):
        if (self.cnt == 0).any():
            raise RuntimeError("some voxels are not covered by any patch")
        return self.sum / self.cnt


# ----------------------------------------------------------------------------- prepared data
class Prepped:
    """Reads data_dir/prepped/meta.json written by scripts/prepare_data.py."""

    def __init__(self, data_dir):
        self.dir = os.path.join(data_dir, "prepped")
        self.meta = json.load(open(os.path.join(self.dir, "meta.json")))

    def subjects(self, split):
        return [s for s, m in self.meta["subjects"].items() if m["split"] == split]

    def volume(self, subject, mod, mmap=True):
        """mod in {'3t', '7t'}; values in [-1, 1], zero-padded (-1) so the patch grids fit (meta['patch'], meta['pad_stride'])."""
        return np.load(os.path.join(self.dir, f"{subject}_{mod}.npy"), mmap_mode="r" if mmap else None)


class PatchDataset(Dataset):
    """paired 3D image patches on a stride grid (for the autoencoder). Returns (x3t, x7t), each (1, P, P, P) in [-1, 1].
    Every item draws a subject and a grid position; with probability fg_bias the position is one whose patch contains brain."""

    def __init__(self, prepped, split, patch, stride, samples_per_epoch, fg_bias=0.8, seed=None, mods=("3t", "7t")):
        self.p, self.subs, self.mods = prepped, prepped.subjects(split), mods
        self.patch, self.n, self.fg_bias, self.seed = int(patch), samples_per_epoch, fg_bias, seed
        self.corners, self.fg = {}, {}
        for s in self.subs:
            v = np.asarray(self.p.volume(s, "7t"))
            c = grid_corners(v.shape, self.patch, stride)
            frac = np.array([(v[a:a + self.patch, b:b + self.patch, d:d + self.patch][::4, ::4, ::4] > -0.9).mean() for a, b, d in c])
            self.corners[s], self.fg[s] = c, np.nonzero(frac > 0.05)[0]

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng(None if self.seed is None else self.seed + i)
        s = self.subs[rng.integers(len(self.subs))]
        pool = self.fg[s] if (rng.random() < self.fg_bias and len(self.fg[s])) else np.arange(len(self.corners[s]))
        a, b, c = self.corners[s][pool[rng.integers(len(pool))]]
        sl = (slice(a, a + self.patch), slice(b, b + self.patch), slice(c, c + self.patch))
        return tuple(torch.from_numpy(np.array(self.p.volume(s, m)[sl], dtype=np.float32))[None] for m in self.mods)


class LatentPatchDataset(Dataset):
    """cached per-patch latents (cache_latents3d.py): {subject}_{key}.npy has shape (N, C, l, l, l), one row per grid patch,
    {subject}_fg.npy the brain fraction of each patch. Each item: dict key -> (C, l, l, l) for one patch.
    seed=None -> random patches every epoch (train); seed=int -> the same patches every time (validation)."""

    def __init__(self, cache_dir, subjects, keys, samples_per_epoch, seed=None, fg_bias=0.8, min_fg=0.05):
        self.dir, self.subs, self.keys = cache_dir, list(subjects), list(keys)
        self.n, self.seed, self.fg_bias = samples_per_epoch, seed, fg_bias
        missing = [f"{s}_{k}" for s in self.subs for k in self.keys + ["fg"] if not os.path.exists(os.path.join(cache_dir, f"{s}_{k}.npy"))]
        if missing:
            raise SystemExit(f"missing cached latents (run the cache scripts first): {missing[:5]} ...")
        self.fg = {s: np.nonzero(np.load(os.path.join(cache_dir, f"{s}_fg.npy")) > min_fg)[0] for s in self.subs}
        self.count = {s: len(np.load(os.path.join(cache_dir, f"{s}_fg.npy"))) for s in self.subs}

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng(None if self.seed is None else self.seed + i)
        s = self.subs[rng.integers(len(self.subs))]
        pool = self.fg[s] if (rng.random() < self.fg_bias and len(self.fg[s])) else np.arange(self.count[s])
        j = int(pool[rng.integers(len(pool))])
        return {k: torch.from_numpy(np.array(np.load(os.path.join(self.dir, f"{s}_{k}.npy"), mmap_mode="r")[j], dtype=np.float32))
                for k in self.keys}
