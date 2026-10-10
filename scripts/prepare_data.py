"""Normalise + pad every volume once, save as float16 .npy in [-1, 1], and build the histogram-matching reference.
usage: python scripts/prepare_data.py --config configs/base.yaml"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from mri3d.data import landmarks, load_nifti, normalize01, pad_to_multiple, read_pairs  # noqa: E402
from mri3d.utils import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_config(ap.parse_args().config)
    d = cfg["data"]
    out = os.path.join(d["data_dir"], "prepped")
    os.makedirs(out, exist_ok=True)
    meta = {"pad_multiple": d["pad_multiple"], "norm_pct": d["norm_pct"], "subjects": {}}
    lms = {"3t": [], "7t": []}
    for r in read_pairs(d["pairs_csv"]):
        vols = {}
        for mod, key in (("3t", "t3"), ("7t", "t7")):
            v, aff, zoom = load_nifti(r[key])
            v01 = normalize01(v, d["norm_pct"])
            if r["split"] == "train":
                lms[mod].append(landmarks(v01, d["brain_thr"]))
            vp, pads = pad_to_multiple(v01 * 2 - 1, d["pad_multiple"], -1.0)
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
