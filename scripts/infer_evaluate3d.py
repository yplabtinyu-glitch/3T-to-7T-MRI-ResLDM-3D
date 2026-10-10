"""Synthesise whole 7T volumes for a split and evaluate against the real 7T.
pipeline per subject: [hist match] -> tiled AE encode -> SPADE -> ResShift sampling (whole latent or tiles) -> tiled decode
metrics (on the unpadded volume, [0,1]): SSIM3D (whole / brain), PSNR, MAE, MAE(brain), mean axial 2D SSIM (last axis),
plus seconds per volume for each stage. Rows for SPADE-only and for the diffusion output (and the AE ceiling with --ceiling).
usage: python scripts/infer_evaluate3d.py --config configs/base.yaml --ckpt runs/base/diffusion/best.pth"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from mri3d.data import Prepped, hist_match, unpad  # noqa: E402
from mri3d.losses import _gauss1d, ssim3d_map  # noqa: E402
from mri3d.nets import ae_encode_mu, load_ae, tiled  # noqa: E402
from mri3d.spade3d import SPADEGenerator3D  # noqa: E402
from mri3d.utils import Timer, autocast, device_of, load_config  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_diffusion3d import build_model  # noqa: E402


def ssim2d_slices(a, b, mask, size=7, sigma=1.5):
    """mean 2D SSIM over slices along the last axis that contain brain; a, b, mask: (D,H,W) tensors"""
    x, y = a.permute(2, 0, 1)[:, None], b.permute(2, 0, 1)[:, None]
    keep = mask.permute(2, 0, 1).flatten(1).any(1)
    x, y = x[keep], y[keep]
    g = _gauss1d(size, sigma, a.device, a.dtype)
    w = (g[:, None] * g[None, :])[None, None]
    blur = lambda t: F.conv2d(t, w, padding=size // 2)
    mx, my = blur(x), blur(y)
    sxx, syy, sxy = blur(x * x) - mx ** 2, blur(y * y) - my ** 2, blur(x * y) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (sxx + syy + c2))
    return float(s.mean())


def metrics(pred, gt, thr):
    pred, gt = pred.clamp(0, 1), gt.clamp(0, 1)
    bm = (gt > thr).float()
    smap = ssim3d_map(pred[None, None], gt[None, None])[0, 0]
    mse = ((pred - gt) ** 2).mean()
    return {"ssim3d": float(smap.mean()), "ssim3d_brain": float((smap * bm).sum() / bm.sum()),
            "psnr": float(10 * torch.log10(1.0 / mse.clamp_min(1e-10))), "mae": float((pred - gt).abs().mean()),
            "mae_brain": float(((pred - gt).abs() * bm).sum() / bm.sum()), "ssim2d_axial": ssim2d_slices(pred, gt, bm > 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True, help="diffusion checkpoint (best.pth)")
    ap.add_argument("--hist_match", choices=["none", "3t", "both"], default=None)
    ap.add_argument("--split", default=None)
    ap.add_argument("--out", default=None, help="default: <ckpt dir>/eval_<split>_hm<mode>")
    ap.add_argument("--ceiling", action="store_true", help="also evaluate the AE reconstruction of the 7T (upper bound)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    dev = device_of(cfg)
    ic = cfg["infer"]
    mode, split = args.hist_match or ic["hist_match"], args.split or ic["split"]
    thr = cfg["data"]["brain_thr"]
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    out = args.out or os.path.join(os.path.dirname(args.ckpt), f"eval_{split}_hm{mode}")
    os.makedirs(out, exist_ok=True)

    ae = load_ae(cfg, ck["ae_ckpt"], dev)
    sp = SPADEGenerator3D(cfg["spade"]["num_classes"], cfg["ae"]["latent_channels"], cfg["spade"]["nf"]).to(dev).eval()
    sp.load_state_dict(torch.load(ck["spade_ckpt"], map_location=dev, weights_only=False)["spade"])
    model = build_model(ck["config"], ck["kappa"]).to(dev).eval()
    model.unet.load_state_dict(ck["unet"])
    scale, C = ck["scale"], cfg["ae"]["latent_channels"]
    meta_lat = json.load(open(os.path.join(cfg["out_dir"], "latents", "meta.json")))
    f = meta_lat["factor"]
    tile, ov = cfg["latent"].get("tile"), cfg["latent"]["overlap"]
    ref = np.load(os.path.join(cfg["data"]["data_dir"], "hist_ref.npz")) if mode != "none" else None
    pr = Prepped(cfg["data"]["data_dir"])
    print(f"split={split} hist_match={mode} | T={model.T} condition={model.condition} kappa={ck['kappa']} | ckpt epoch {ck['epoch']}")

    enc = lambda x: tiled(lambda t: ae_encode_mu(ae, t).float(), x, tile, ov, 1 / f, C) if tile else ae_encode_mu(ae, x).float()
    dec = lambda z: tiled(lambda t: ae.decode(t).float(), z, tile // f, ov // f, f, 1) if tile else ae.decode(z).float()

    def sample(y0, cond):
        g = torch.Generator(device=y0.device).manual_seed(ic["seed"])
        noise = torch.randn(y0.shape, generator=g, device=y0.device)
        lt = ic.get("latent_tile")
        if not lt:
            return model.sample(y0, cond, noise)
        n = len(cond)
        packed = torch.cat([y0, noise] + cond, 1)
        fn = lambda p: model.sample(p[:, :C], [p[:, (2 + i) * C:(3 + i) * C] for i in range(n)], p[:, C:2 * C])
        return tiled(fn, packed, lt, ic["latent_overlap"], 1, C)

    rows = []
    for s in pr.subjects(split):
        sm = pr.meta["subjects"][s]
        v3 = (np.asarray(pr.volume(s, "3t"), dtype=np.float32) + 1) / 2
        v7 = (np.asarray(pr.volume(s, "7t"), dtype=np.float32) + 1) / 2
        if mode in ("3t", "both"):
            v3 = hist_match(v3, ref["lm_3t"], thr)
        if mode == "both":
            v7 = hist_match(v7, ref["lm_7t"], thr)
        x3 = torch.from_numpy(v3 * 2 - 1)[None, None].to(dev)
        times = {}
        with torch.no_grad(), autocast(dev, cfg["bf16"]):
            with Timer() as tm:
                z3 = enc(x3)
            times["t_encode"] = tm.dt
            with Timer() as tm:
                zs = sp(z3, torch.ones(1, dtype=torch.long, device=dev)).float()
            times["t_spade"] = tm.dt
            with Timer() as tm:
                zd = sample(zs * scale, model.cond(z3 * scale, zs * scale)) / scale
            times["t_diffusion"] = tm.dt
            with Timer() as tm:
                img_d = (dec(zd) + 1) / 2
            times["t_decode"] = tm.dt
            img_s = (dec(zs) + 1) / 2
            img_c = (dec(enc(torch.from_numpy(v7 * 2 - 1)[None, None].to(dev))) + 1) / 2 if args.ceiling else None
        pads = sm["pads"]
        gt = torch.from_numpy(unpad(v7, pads)).to(dev)
        cut = lambda im: torch.from_numpy(unpad(im[0, 0].float().cpu().numpy(), pads)).to(dev)
        total = sum(times.values())
        for name, im in (("spade", img_s), ("diffusion", img_d), ("ae_ceiling", img_c)):
            if im is None:
                continue
            r = {"subject": s, "method": name, **metrics(cut(im), gt, thr)}
            if name == "diffusion":
                r.update({k: round(v, 3) for k, v in times.items()}, t_total=round(total, 3))
            rows.append(r)
            print(f"  {s} {name:>10}: " + "  ".join(f"{k}={v:.4f}" for k, v in r.items() if isinstance(v, float) and not k.startswith("t_")))
        print(f"  {s} time: " + "  ".join(f"{k}={v:.2f}s" for k, v in times.items()) + f"  total={total:.2f}s")
        if ic.get("save_nifti"):
            import nibabel as nib
            aff = nib.load(sm["t7_path"]).affine
            nib.save(nib.Nifti1Image(cut(img_d).cpu().numpy().astype(np.float32), aff), os.path.join(out, f"{s}_synth7t.nii.gz"))

    keys = [k for k in rows[0] if k not in ("subject", "method")]
    for name in dict.fromkeys(r["method"] for r in rows):
        sub = [r for r in rows if r["method"] == name]
        rows.append({"subject": "overall", "method": name,
                     **{k: float(np.mean([r[k] for r in sub if k in r])) for k in keys if any(k in r for r in sub)}})
    allk = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(out, "metrics.csv"), "w", newline="") as fcsv:
        w = csv.DictWriter(fcsv, allk)
        w.writeheader()
        w.writerows(rows)
    print("\noverall:")
    for r in rows:
        if r["subject"] == "overall":
            print(f"  {r['method']:>10}: " + "  ".join(f"{k}={v:.4f}" for k, v in r.items() if isinstance(v, float)))
    print(f"saved {out}/metrics.csv")


if __name__ == "__main__":
    main()
