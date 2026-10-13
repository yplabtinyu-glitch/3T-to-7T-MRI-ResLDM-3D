"""Stage 3: ResShift latent diffusion in 3D on cached per-patch latents (AE and SPADE frozen, read from the cache).
loss = lambda_simple * MSE(x0_hat, x0) + image losses (MIND vs 3T, LF vs SPADE, optional registration-corrected L1 vs 7T).
usage: python scripts/train_diffusion3d.py --config configs/base.yaml [--name p4_mindlf] [--resume]"""
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
from mri3d.nets import build_unet, load_ae  # noqa: E402
from mri3d.resshift3d import CONDITIONS, ResShift3D  # noqa: E402
from mri3d.utils import EpochLog, Meter, autocast, device_of, fixed_seed, load_config, save_ckpt, seed_all  # noqa: E402


def resolve_kappa(cfg, meta):
    d = cfg["diffusion"]
    k = meta.get("kappa_auto") if d["kappa"] == "auto" else d["kappa"]
    if k is None:
        raise SystemExit("kappa: auto needs kappa_auto in latents/meta.json -- run cache_latents3d.py --stage spade first")
    k = [float(v) * float(d.get("kappa_mult", 1.0)) for v in (k if isinstance(k, list) else [k])]
    return k if len(k) > 1 else k[0]


def build_model(cfg, kappa):
    d, C = cfg["diffusion"], cfg["ae"]["latent_channels"]
    unet = build_unet(d["unet"], C * (1 + CONDITIONS[d["condition"]]), C)
    return ResShift3D(unet, C, d["T"], d["p"], kappa, d["condition"])


def step(model, il, b, scale, dev, cfg):
    z3, z7, zs = (b[k].to(dev, non_blocking=True) * scale for k in ("z3t", "z7t", "zspade"))
    t = torch.randint(0, model.T, (z3.shape[0],), device=dev)
    with autocast(dev, cfg["bf16"]):
        zt = model.q_sample(z7, zs, t, torch.randn_like(z7))
        x0 = model.predict_x0(zt, t, model.cond(z3, zs)).float()
        mse = F.mse_loss(x0, z7)
        l_img, parts = il(x0, z3, zs, z7)
    loss = cfg["diffusion"]["train"].get("lambda_simple", 1.0) * mse + l_img
    return loss, {"mse": mse.detach(), **parts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", default="diffusion", help="sub-folder of out_dir for this run")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed_all(cfg["seed"])
    dev = device_of(cfg)
    t = cfg["diffusion"]["train"]
    out = os.path.join(cfg["out_dir"], args.name)
    os.makedirs(out, exist_ok=True)
    cdir = os.path.join(cfg["out_dir"], "latents")
    meta = json.load(open(os.path.join(cdir, "meta.json")))
    if "spade_ckpt" not in meta:
        raise SystemExit("no SPADE latents cached -- run cache_latents3d.py --stage spade first")
    scale, kappa = meta["scale"], resolve_kappa(cfg, meta)
    pr = Prepped(cfg["data"]["data_dir"])

    keys = ["z3t", "z7t", "zspade"]
    tr = DataLoader(LatentPatchDataset(cdir, pr.subjects("train"), keys, t["samples_per_epoch"]),
                    batch_size=t["batch_size"], num_workers=cfg["num_workers"], drop_last=True,
                    persistent_workers=cfg["num_workers"] > 0)
    va = DataLoader(LatentPatchDataset(cdir, pr.subjects("val"), keys, t["val_samples"], seed=999),
                    batch_size=t["batch_size"], num_workers=cfg["num_workers"])

    model = build_model(cfg, kappa).to(dev)
    il = ImageLosses(load_ae(cfg, meta["ae_ckpt"], dev), scale, cfg["diffusion"].get("losses", {}), meta["factor"]).to(dev)
    params = list(model.unet.parameters()) + (list(il.reg.parameters()) if il.reg is not None else [])
    opt = torch.optim.AdamW(params, lr=t["lr"])
    if t.get("compile"):
        model.unet = torch.compile(model.unet, mode="reduce-overhead")
    print(f"UNet params {sum(p.numel() for p in model.unet.parameters()) / 1e6:.1f} M | condition {model.condition} | T {model.T} | "
          f"kappa {kappa} | scale {scale:.4f} | image losses mind={il.l_mind} lf={il.l_lf} corr={il.l_corr}"
          + (f" (reg net {sum(p.numel() for p in il.reg.parameters())} params)" if il.reg is not None else ""))

    start, best, bad = 0, float("inf"), 0
    if args.resume and os.path.exists(os.path.join(out, "latest.pth")):
        ck = torch.load(os.path.join(out, "latest.pth"), map_location=dev, weights_only=False)
        model.unet.load_state_dict(ck["unet"]); opt.load_state_dict(ck["opt"])
        if il.reg is not None:
            il.reg.load_state_dict(ck["reg"])
        start, best, bad = ck["epoch"] + 1, ck["best"], ck["bad"]
    log = EpochLog(os.path.join(out, "log.csv"))

    def state(ep, with_opt):
        s = dict(unet=model.unet.state_dict(), epoch=ep, best=best, bad=bad, kappa=kappa, scale=scale, config=cfg,
                 spade_ckpt=meta["spade_ckpt"], ae_ckpt=meta["ae_ckpt"])
        if il.reg is not None:
            s["reg"] = il.reg.state_dict()
        if with_opt:
            s["opt"] = opt.state_dict()
        return s

    for ep in range(start, t["epochs"]):
        model.train()
        m = Meter()
        for b in tr:
            loss, parts = step(model, il, b, scale, dev, cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            m.add({"loss": loss.detach(), **parts})
        model.eval()
        mv = Meter()
        with torch.no_grad(), fixed_seed(0):    # same timesteps / noise / sub-crops every validation
            for b in va:
                loss, parts = step(model, il, b, scale, dev, cfg)
                mv.add({"loss": loss, **parts})
        val = mv.means()["loss"]
        log.write({"epoch": ep, **m.means("train_"), **mv.means("val_")})
        if val < best:
            best, bad = val, 0
            save_ckpt(os.path.join(out, "best.pth"), **state(ep, False))
        else:
            bad += 1
        save_ckpt(os.path.join(out, "latest.pth"), **state(ep, True))
        if t.get("ckpt_every") and (ep + 1) % t["ckpt_every"] == 0:
            save_ckpt(os.path.join(out, f"epoch{ep}.pth"), **state(ep, False))
        if bad >= t["patience"]:
            print(f"early stop at epoch {ep} (best val {best:.5f})")
            break


if __name__ == "__main__":
    main()
