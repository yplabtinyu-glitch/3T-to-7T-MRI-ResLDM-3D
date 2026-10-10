"""Encode every volume once with the frozen AE (stage ae), or add the frozen SPADE output (stage spade).
  --stage ae    : out_dir/latents/{subject}_z3t.npy, _z7t.npy  (posterior mean, unscaled, float16) + meta.json with scale
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

from mri3d.data import Prepped  # noqa: E402
from mri3d.nets import ae_encode_mu, ae_factor, load_ae, tiled  # noqa: E402
from mri3d.spade3d import SPADEGenerator3D  # noqa: E402
from mri3d.utils import autocast, device_of, load_config  # noqa: E402


def encode_volume(ae, cfg, vol, dev):
    x = torch.from_numpy(np.asarray(vol, dtype=np.float32))[None, None].to(dev)
    f, C = ae_factor(cfg["ae"]), cfg["ae"]["latent_channels"]
    tile = cfg["latent"].get("tile")
    with torch.no_grad(), autocast(dev, cfg["bf16"]):
        if tile:
            z = tiled(lambda t: ae_encode_mu(ae, t).float(), x, tile, cfg["latent"]["overlap"], 1 / f, C)
        else:
            z = ae_encode_mu(ae, x).float()
    return z[0].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", choices=["ae", "spade"], required=True)
    ap.add_argument("--ae_ckpt", default=None, help="default: out_dir/ae/ae_best.pth")
    ap.add_argument("--spade_ckpt", default=None, help="default: out_dir/spade/spade_best.pth")
    args = ap.parse_args()
    cfg = load_config(args.config)
    dev = device_of(cfg)
    cdir = os.path.join(cfg["out_dir"], "latents")
    os.makedirs(cdir, exist_ok=True)
    meta_p = os.path.join(cdir, "meta.json")
    pr = Prepped(cfg["data"]["data_dir"])
    subs = list(pr.meta["subjects"])

    if args.stage == "ae":
        ck = args.ae_ckpt or os.path.join(cfg["out_dir"], "ae", "ae_best.pth")
        ae = load_ae(cfg, ck, dev)
        t0 = time.time()
        for s in subs:
            for mod in ("3t", "7t"):
                z = encode_volume(ae, cfg, pr.volume(s, mod), dev)
                np.save(os.path.join(cdir, f"{s}_z{mod}.npy"), z.astype(np.float16))
            print(f"  {s}: latent {z.shape}")
        tr = [np.load(os.path.join(cdir, f"{s}_z7t.npy")).astype(np.float32) for s in pr.subjects("train")]
        scale = float(1.0 / np.concatenate([z.reshape(z.shape[0], -1) for z in tr], 1).std())
        meta = {"ae_ckpt": os.path.abspath(ck), "scale": scale, "factor": ae_factor(cfg["ae"])}
        json.dump(meta, open(meta_p, "w"), indent=1)
        print(f"encoded {len(subs)} subjects in {time.time() - t0:.0f} s; scale = 1/std(train z7t) = {scale:.4f}")
        return

    meta = json.load(open(meta_p))
    ck = args.spade_ckpt or os.path.join(cfg["out_dir"], "spade", "spade_best.pth")
    sp = SPADEGenerator3D(cfg["spade"]["num_classes"], cfg["ae"]["latent_channels"], cfg["spade"]["nf"]).to(dev).eval()
    sp.load_state_dict(torch.load(ck, map_location=dev, weights_only=False)["spade"])
    for s in subs:
        z3 = torch.from_numpy(np.load(os.path.join(cdir, f"{s}_z3t.npy")).astype(np.float32))[None].to(dev)
        with torch.no_grad(), autocast(dev, cfg["bf16"]):
            zs = sp(z3, torch.ones(1, dtype=torch.long, device=dev)).float()
        np.save(os.path.join(cdir, f"{s}_zspade.npy"), zs[0].cpu().numpy().astype(np.float16))
    res = []
    for s in pr.subjects("train"):
        a = np.load(os.path.join(cdir, f"{s}_zspade.npy")).astype(np.float32)
        b = np.load(os.path.join(cdir, f"{s}_z7t.npy")).astype(np.float32)
        res.append(((a - b) * meta["scale"]).reshape(a.shape[0], -1))
    kappa = np.concatenate(res, 1).std(1)
    meta.update(spade_ckpt=os.path.abspath(ck), spade_mtime=os.path.getmtime(ck), kappa_auto=[round(float(k), 4) for k in kappa])
    json.dump(meta, open(meta_p, "w"), indent=1)
    print(f"cached SPADE latents for {len(subs)} subjects; kappa_auto (per channel std of scaled zspade - z7t) = {meta['kappa_auto']}")


if __name__ == "__main__":
    main()
