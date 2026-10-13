"""Stage 2: 3D SPADE on cached per-patch latents (8^3 x 4 = one 64^3 patch). loss = lambda_latent * L1(SPADE(z3t), z7t) + lambda_mind * MIND(D(SPADE(z3t)), D(z3t)).
usage: python scripts/train_spade3d.py --config configs/base.yaml [--resume]"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from mri3d.data import LatentPatchDataset, Prepped  # noqa: E402
from mri3d.imageloss import ImageLosses  # noqa: E402
from mri3d.nets import load_ae  # noqa: E402
from mri3d.spade3d import SPADEGenerator3D  # noqa: E402
from mri3d.utils import EpochLog, Meter, autocast, device_of, fixed_seed, load_config, save_ckpt, seed_all  # noqa: E402


def step(sp, il, b, dev, cfg, t):
    z3, z7 = b["z3t"].to(dev, non_blocking=True), b["z7t"].to(dev, non_blocking=True)
    y = torch.ones(z3.shape[0], dtype=torch.long, device=dev)
    with autocast(dev, cfg["bf16"]):
        zs = sp(z3, y).float()
        l_lat = F.l1_loss(zs, z7)
        l_img, parts = il(zs, z3, z3, z7)
    return t["lambda_latent"] * l_lat + l_img, {"latent": l_lat.detach(), **parts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed_all(cfg["seed"])
    dev = device_of(cfg)
    t = cfg["spade"]["train"]
    out = os.path.join(cfg["out_dir"], "spade")
    os.makedirs(out, exist_ok=True)
    cdir = os.path.join(cfg["out_dir"], "latents")
    meta = json.load(open(os.path.join(cdir, "meta.json")))
    pr = Prepped(cfg["data"]["data_dir"])

    keys = ["z3t", "z7t"]
    tr = DataLoader(LatentPatchDataset(cdir, pr.subjects("train"), keys, t["samples_per_epoch"]),
                    batch_size=t["batch_size"], num_workers=cfg["num_workers"], drop_last=True)
    va = DataLoader(LatentPatchDataset(cdir, pr.subjects("val"), keys, t["val_samples"], seed=4321),
                    batch_size=t["batch_size"], num_workers=cfg["num_workers"])
    ae = load_ae(cfg, meta["ae_ckpt"], dev)
    sp = SPADEGenerator3D(cfg["spade"]["num_classes"], cfg["ae"]["latent_channels"], cfg["spade"]["nf"]).to(dev)
    il = ImageLosses(ae, 1.0, cfg["spade"].get("losses", {}), meta["factor"]).to(dev)   # SPADE works on unscaled latents
    opt = torch.optim.AdamW(sp.parameters(), lr=t["lr"])
    print(f"SPADE params: {sum(p.numel() for p in sp.parameters()) / 1e6:.2f} M | image losses: mind={il.l_mind}")

    start, best, bad = 0, float("inf"), 0
    if args.resume and os.path.exists(os.path.join(out, "spade_latest.pth")):
        ck = torch.load(os.path.join(out, "spade_latest.pth"), map_location=dev, weights_only=False)
        sp.load_state_dict(ck["spade"]); opt.load_state_dict(ck["opt"])
        start, best, bad = ck["epoch"] + 1, ck["best"], ck["bad"]
    log = EpochLog(os.path.join(out, "log.csv"))
    for ep in range(start, t["epochs"]):
        sp.train()
        m = Meter()
        for b in tr:
            loss, parts = step(sp, il, b, dev, cfg, t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            m.add({"loss": loss.detach(), **parts})
        sp.eval()
        mv = Meter()
        with torch.no_grad(), fixed_seed(0):   # same image-loss sub-crops every validation
            for b in va:
                loss, parts = step(sp, il, b, dev, cfg, t)
                mv.add({"loss": loss, **parts})
        val = mv.means()["loss"]
        log.write({"epoch": ep, **m.means("train_"), **mv.means("val_")})
        if val < best:
            best, bad = val, 0
            save_ckpt(os.path.join(out, "spade_best.pth"), spade=sp.state_dict(), epoch=ep, val_loss=val, config=cfg)
        else:
            bad += 1
        save_ckpt(os.path.join(out, "spade_latest.pth"), spade=sp.state_dict(), opt=opt.state_dict(), epoch=ep, best=best, bad=bad, config=cfg)
        if bad >= t["patience"]:
            print(f"early stop at epoch {ep} (best val {best:.5f})")
            break


if __name__ == "__main__":
    main()
