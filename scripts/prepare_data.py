"""Normalise + zero-pad every volume once, save as float16 .npy in [-1, 1], and build the histogram-matching reference.
Padding: symmetric, value 0 in [0, 1] (= -1 stored), so that size >= patch and (size - patch) % pad_stride == 0 on every axis
-> the AE grid (ae.train.stride) and the inference / latent grid (infer.stride) both tile the volume exactly.
usage: python scripts/prepare_data.py --config configs/base.yaml"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from mri3d.data import landmarks, load_nifti, normalize01, pad_for_patches, read_pairs  # noqa: E402
from mri3d.utils import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_config(ap.parse_args().config)
    d = cfg["data"]
    out = os.path.join(d["data_dir"], "prepped")
    os.makedirs(out, exist_ok=True)
    P, S = d["patch"], d["pad_stride"]
    for k, st in (("ae.train.stride", cfg["ae"]["train"]["stride"]), ("latent.stride", cfg["latent"]["stride"]), ("infer.stride", cfg["infer"]["stride"])):
        if S % st:
            sys.exit(f"data.pad_stride={S} must be a multiple of {k}={st} so that grid fits the padded volume")
    meta = {"patch": P, "pad_stride": S, "norm_pct": d["norm_pct"], "subjects": {}}
    lms = {"3t": [], "7t": []}
    for r in read_pairs(d["pairs_csv"]):
        vols = {}
        for mod, key in (("3t", "t3"), ("7t", "t7")):
            v, aff, zoom = load_nifti(r[key])
            v01 = normalize01(v, d["norm_pct"])
            if r["split"] == "train":
                lms[mod].append(landmarks(v01, d["brain_thr"]))
            vp, pads = pad_for_patches(v01 * 2 - 1, P, S, -1.0)
            np.save(os.path.join(out, f"{r['subject']}_{mod}.npy"), vp.astype(np.float16))
            vols[mod] = (v.shape, pads)
        if vols["3t"][0] != vols["7t"][0]:
            sys.exit(f"{r['subject']}: 3T {vols['3t'][0]} and 7T {vols['7t'][0]} have different shapes -- register first")
        meta["subjects"][r["subject"]] = {"split": r["split"], "shape": list(vols["7t"][0]), "pads": vols["7t"][1],
                                          "t7_path": os.path.abspath(r["t7"]), "t3_path": os.path.abspath(r["t3"])}
        print(f"{r['subject']:>12} {r['split']:>5} {vols['7t'][0]} -> padded {tuple(s + a + b for s, (a, b) in zip(vols['7t'][0], vols['7t'][1]))}")
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    ref = {"lm_3t": np.mean(lms["3t"], 0), "lm_7t": np.mean(lms["7t"], 0), "thr": d["brain_thr"]}
    np.savez(os.path.join(d["data_dir"], "hist_ref.npz"), **ref)
    print(f"\nsaved {len(meta['subjects'])} subjects to {out}; histogram reference from {len(lms['3t'])} training subjects")
    print("  3T landmarks:", np.round(ref["lm_3t"], 3))
    print("  7T landmarks:", np.round(ref["lm_7t"], 3))


if __name__ == "__main__":
    main()
