"""Print shape / spacing / intensity of every volume in the pair list and flag 3T/7T grid mismatches.
usage: python scripts/inspect_volumes.py --config configs/base.yaml"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from mri3d.data import load_nifti, read_pairs  # noqa: E402
from mri3d.utils import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_config(ap.parse_args().config)
    rows = read_pairs(cfg["data"]["pairs_csv"])
    bad = 0
    print(f"{'subject':>12} {'split':>5}  {'shape 3T':>16} {'shape 7T':>16}  {'spacing 7T (mm)':>20}  {'3T max':>8} {'7T max':>8}")
    for r in rows:
        v3, a3, z3 = load_nifti(r["t3"])
        v7, a7, z7 = load_nifti(r["t7"])
        same = v3.shape == v7.shape and np.allclose(a3, a7, atol=1e-3)
        bad += not same
        print(f"{r['subject']:>12} {r['split']:>5}  {str(v3.shape):>16} {str(v7.shape):>16}  {str(tuple(round(float(z), 2) for z in z7)):>20}  "
              f"{v3.max():8.1f} {v7.max():8.1f}{'' if same else '   <-- 3T and 7T grids differ'}")
    P, S = cfg["data"]["patch"], cfg["data"]["pad_stride"]
    print(f"\n{len(rows)} pairs; {bad} with mismatched grids. Volumes will be zero-padded so that {P}^3 patches tile them with stride {S}.")
    if bad:
        sys.exit("register the 3T onto the 7T grid first (the pipeline assumes voxel-aligned pairs)")


if __name__ == "__main__":
    main()
