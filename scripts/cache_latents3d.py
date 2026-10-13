"""Encode every 64^3 patch of every volume with the frozen AE (stage ae), or add the frozen SPADE output (stage spade).
Patches lie on a stride-`latent.stride` grid (32 by default), exactly the grid used at inference, and each patch is
encoded on its own -- so SPADE / diffusion train on the same per-patch latents they will see at inference.
  --stage ae    : out_dir/latents/{subject}_z3t.npy, _z7t.npy  (N, C, l, l, l) posterior means, unscaled, float16
                  {subject}_corners.npy (N, 3) patch corners, {subject}_fg.npy (N,) brain fraction of each 7T patch
                  meta.json with scale = 1 / std(train z7t over brain patches)
  --stage spade : out_dir/latents/{subject}_zspade.npy = SPADE(z3t, class 1) + kappa_auto in meta.json
usage: python scripts/cache_latents3d.py --config configs/base.yaml --stage ae
       python scripts/cache_latents3d.py --config configs/base.yaml --stage spade"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from mri3d.data import Prepped, extract_patches, grid_corners  # noqa: E402
from mri3d.nets import ae_encode_mu, ae_factor, load_ae  # noqa: E402
from mri3d.spade3d import SPADEGenerator3D  # noqa: E402
from mri3d.utils import autocast, device_of, load_config  # noqa: E402

FG_MIN = 0.05   # patches with at least this brain fraction count as brain patches (scale, kappa, training sampler)


def batched(fn, x, bs):
    return torch.cat([fn(x[i:i + bs]) for i in range(0, len(x), bs)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", choices=["ae", "spade"], required=True)
    ap.add_argument("--ae_ckpt", default=None, help="default: out_dir/ae/ae_best.pth")
    ap.add_argument("--spade_ckpt", default=None, help="default: out_dir/spade/spade_best.pth")
    args = ap.parse_args()
    cfg = load_config(args.config)
    dev = device_of(cfg)
    P, stride, bs = cfg["data"]["patch"], cfg["latent"]["stride"], cfg["latent"]["batch"]
    cdir = os.path.join(cfg["out_dir"], "latents")
    os.makedirs(cdir, exist_ok=True)
    meta_p = os.path.join(cdir, "meta.json")
    pr = Prepped(cfg["data"]["data_dir"])
    subs = list(pr.meta["subjects"])
    thr = cfg["data"]["brain_thr"] * 2 - 1     # brain threshold in the stored [-1, 1] range

    if args.stage == "ae":
        ck = args.ae_ckpt or os.path.join(cfg["out_dir"], "ae", "ae_best.pth")
        ae = load_ae(cfg, ck, dev)
        enc = lambda x: ae_encode_mu(ae, x).float()
        t0 = time.time()
        for s in subs:
            v7 = np.asarray(pr.volume(s, "7t"), dtype=np.float32)
            corners = grid_corners(v7.shape, P, stride)
            fg = np.array([(v7[a:a + P, b:b + P, c:c + P] > thr).mean() for a, b, c in corners], dtype=np.float32)
            np.save(os.path.join(cdir, f"{s}_corners.npy"), corners)
            np.save(os.path.join(cdir, f"{s}_fg.npy"), fg)
            for mod in ("3t", "7t"):
                v = torch.from_numpy(np.asarray(pr.volume(s, mod), dtype=np.float32))
                x = extract_patches(v, corners, P)[:, None]                     # (N, 1, P, P, P)
                with torch.no_grad(), autocast(dev, cfg["bf16"]):
                    z = batched(lambda b: enc(b.to(dev)).cpu(), x, bs)
                np.save(os.path.join(cdir, f"{s}_z{mod}.npy"), z.numpy().astype(np.float16))
            print(f"  {s}: {len(corners)} patches ({int((fg > FG_MIN).sum())} brain), latent {tuple(z.shape[1:])}")
        tr = []
        for s in pr.subjects("train"):
            z = np.load(os.path.join(cdir, f"{s}_z7t.npy")).astype(np.float32)
            keep = np.load(os.path.join(cdir, f"{s}_fg.npy")) > FG_MIN
            tr.append(np.moveaxis(z[keep], 1, 0).reshape(z.shape[1], -1))
        scale = float(1.0 / np.concatenate(tr, 1).std())
        meta = {"ae_ckpt": os.path.abspath(ck), "scale": scale, "factor": ae_factor(cfg["ae"]), "patch": P, "stride": stride}
        json.dump(meta, open(meta_p, "w"), indent=1)
        print(f"encoded {len(subs)} subjects in {time.time() - t0:.0f} s; scale = 1/std(train z7t, brain patches) = {scale:.4f}")
        return

    meta = json.load(open(meta_p))
    ck = args.spade_ckpt or os.path.join(cfg["out_dir"], "spade", "spade_best.pth")
    sp = SPADEGenerator3D(cfg["spade"]["num_classes"], cfg["ae"]["latent_channels"], cfg["spade"]["nf"]).to(dev).eval()
    sp.load_state_dict(torch.load(ck, map_location=dev, weights_only=False)["spade"])
    for s in subs:
        z3 = torch.from_numpy(np.load(os.path.join(cdir, f"{s}_z3t.npy")).astype(np.float32))
        with torch.no_grad(), autocast(dev, cfg["bf16"]):
            zs = batched(lambda b: sp(b.to(dev), torch.ones(len(b), dtype=torch.long, device=dev)).float().cpu(), z3, bs)
        np.save(os.path.join(cdir, f"{s}_zspade.npy"), zs.numpy().astype(np.float16))
    res = []
    for s in pr.subjects("train"):
        a = np.load(os.path.join(cdir, f"{s}_zspade.npy")).astype(np.float32)
        b = np.load(os.path.join(cdir, f"{s}_z7t.npy")).astype(np.float32)
        keep = np.load(os.path.join(cdir, f"{s}_fg.npy")) > FG_MIN
        res.append(np.moveaxis((a[keep] - b[keep]) * meta["scale"], 1, 0).reshape(a.shape[1], -1))
    kappa = np.concatenate(res, 1).std(1)
    meta.update(spade_ckpt=os.path.abspath(ck), spade_mtime=os.path.getmtime(ck), kappa_auto=[round(float(k), 4) for k in kappa])
    json.dump(meta, open(meta_p, "w"), indent=1)
    print(f"cached SPADE latents for {len(subs)} subjects; kappa_auto (per channel std of scaled zspade - z7t, brain patches) = {meta['kappa_auto']}")


if __name__ == "__main__":
    main()
