"""Stage 1: shared 3D AutoencoderKL for 3T and 7T, trained on random 3D patches.
loss = L1 + kl_weight * KL (+ adv_weight * 3D PatchGAN after adv_start_epoch). One CSV row per epoch.
usage: python scripts/train_ae3d.py --config configs/base.yaml [--resume]"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from monai.networks.nets import PatchDiscriminator  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from mri3d.data import PatchDataset, Prepped  # noqa: E402
from mri3d.nets import build_ae  # noqa: E402
from mri3d.utils import EpochLog, Meter, autocast, device_of, load_config, save_ckpt, seed_all  # noqa: E402


def kl_loss(mu, sigma):
    return 0.5 * torch.mean(torch.sum(mu.pow(2) + sigma.pow(2) - torch.log(sigma.pow(2) + 1e-8) - 1, dim=list(range(1, mu.ndim))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed_all(cfg["seed"])
    dev = device_of(cfg)
    c, t = cfg["ae"], cfg["ae"]["train"]
    out = os.path.join(cfg["out_dir"], "ae")
    os.makedirs(out, exist_ok=True)

    pr = Prepped(cfg["data"]["data_dir"])
    tr = DataLoader(PatchDataset(pr, "train", t["patch"], t["samples_per_epoch"]), batch_size=t["batch_size"],
                    shuffle=False, num_workers=cfg["num_workers"], drop_last=True, persistent_workers=cfg["num_workers"] > 0)
    va = DataLoader(PatchDataset(pr, "val", t["patch"], t["val_samples"], seed=1234), batch_size=t["batch_size"],
                    num_workers=cfg["num_workers"])

    ae = build_ae(c).to(dev)
    opt = torch.optim.AdamW(ae.parameters(), lr=t["lr"])
    use_adv = t.get("adv_weight", 0) > 0
    if use_adv:
        disc = PatchDiscriminator(spatial_dims=3, channels=32, in_channels=1, num_layers_d=t.get("adv_layers", 3)).to(dev)
        opt_d = torch.optim.AdamW(disc.parameters(), lr=t["lr"])
    print(f"AE params: {sum(p.numel() for p in ae.parameters()) / 1e6:.1f} M | device {dev}")

    start, best, bad = 0, float("inf"), 0
    if args.resume and os.path.exists(os.path.join(out, "ae_latest.pth")):
        ck = torch.load(os.path.join(out, "ae_latest.pth"), map_location=dev, weights_only=False)
        ae.load_state_dict(ck["ae"]); opt.load_state_dict(ck["opt"])
        if use_adv and "disc" in ck:
            disc.load_state_dict(ck["disc"]); opt_d.load_state_dict(ck["opt_d"])
        start, best, bad = ck["epoch"] + 1, ck["best"], ck.get("bad", 0)
        print(f"resumed at epoch {start}")
    log = EpochLog(os.path.join(out, "log.csv"))

    for ep in range(start, t["epochs"]):
        ae.train()
        m = Meter()
        adv_on = use_adv and ep >= t.get("adv_start_epoch", 0)
        for x3, x7 in tr:
            x = torch.cat([x3, x7]).to(dev, non_blocking=True)
            with autocast(dev, cfg["bf16"]):
                rec, mu, sigma = ae(x)
                l1 = F.l1_loss(rec.float(), x)
                kl = kl_loss(mu.float(), sigma.float())
                loss = l1 + t["kl_weight"] * kl
                if adv_on:
                    g_adv = ((disc(rec.float())[-1] - 1) ** 2).mean()
                    loss = loss + t["adv_weight"] * g_adv
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            rec_d = {"loss": loss.detach(), "l1": l1.detach(), "kl": kl.detach()}
            if adv_on:
                with autocast(dev, cfg["bf16"]):
                    d_loss = 0.5 * (((disc(x)[-1] - 1) ** 2).mean() + (disc(rec.detach().float())[-1] ** 2).mean())
                opt_d.zero_grad(set_to_none=True)
                d_loss.backward()
                opt_d.step()
                rec_d.update(g_adv=g_adv.detach(), d_loss=d_loss.detach())
            m.add(rec_d)
        ae.eval()
        mv = Meter()
        with torch.no_grad():
            for x3, x7 in va:
                x = torch.cat([x3, x7]).to(dev)
                with autocast(dev, cfg["bf16"]):
                    rec, _, _ = ae(x)
                mv.add({"l1": F.l1_loss(rec.float(), x)})
        val = mv.means()["l1"]
        row = {"epoch": ep, **m.means("train_"), "val_l1": val, "adv_on": int(adv_on)}
        log.write(row)
        state = dict(ae=ae.state_dict(), opt=opt.state_dict(), epoch=ep, best=min(best, val), bad=bad, config=cfg)
        if use_adv:
            state.update(disc=disc.state_dict(), opt_d=opt_d.state_dict())
        if val < best:
            best, bad = val, 0
            save_ckpt(os.path.join(out, "ae_best.pth"), ae=ae.state_dict(), epoch=ep, val_l1=val, config=cfg)
        else:
            bad += 1
        state["bad"] = bad
        save_ckpt(os.path.join(out, "ae_latest.pth"), **state)
        if bad >= t["patience"]:
            print(f"early stop: no val improvement for {bad} epochs (best {best:.5f})")
            break


if __name__ == "__main__":
    main()
