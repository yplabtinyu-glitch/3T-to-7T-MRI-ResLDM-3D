"""Synthesise whole 7T volumes for a split and evaluate against the real 7T.
pipeline per subject: [hist match] -> 64^3 patches on a stride-32 grid of the zero-padded volume ->
  per patch: AE encode -> SPADE -> ResShift (T steps) -> AE decode -> overlapping patches averaged voxel-wise -> unpad
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

from mri3d.data import PatchMerger, Prepped, extract_patches, grid_corners, hist_match, unpad  # noqa: E402
from mri3d.losses import _gauss1d, ssim3d_map  # noqa: E402
from mri3d.nets import ae_encode_mu, load_ae  # noqa: E402
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
    scale = ck["scale"]
    P, S, B = cfg["data"]["patch"], ic["stride"], ic["batch"]
    if ic.get("merge", "mean") != "mean":
        raise ValueError("only infer.merge: mean is implemented")
    ref = np.load(os.path.join(cfg["data"]["data_dir"], "hist_ref.npz")) if mode != "none" else None
    pr = Prepped(cfg["data"]["data_dir"])
    print(f"split={split} hist_match={mode} | T={model.T} condition={model.condition} kappa={ck['kappa']} | "
          f"patch {P}^3 stride {S} batch {B} merge mean | ckpt epoch {ck['epoch']}")

    def run_patches(vol, want_spade):
        """vol: padded (D,H,W) numpy in [-1,1] -> merged diffusion (and SPADE-only) volumes in [0,1], stage times"""
        corners = grid_corners(vol.shape, P, S)
        x = torch.from_numpy(np.ascontiguousarray(vol))[None].to(dev)          # (1,D,H,W)
        m_d = PatchMerger(vol.shape, P, dev)
        m_s = PatchMerger(vol.shape, P, dev) if want_spade else None
        times = dict.fromkeys(("t_encode", "t_spade", "t_diffusion", "t_decode"), 0.0)
        g = torch.Generator(device=dev).manual_seed(ic["seed"])
        torch.manual_seed(ic["seed"])                                             # per-step noise inside sample()
        for i in range(0, len(corners), B):
            cb = corners[i:i + B]
            xb = extract_patches(x, cb, P)                                         # (b,1,P,P,P)
            with torch.no_grad(), autocast(dev, cfg["bf16"]):
                with Timer() as tm:
                    z3 = ae_encode_mu(ae, xb).float()
                times["t_encode"] += tm.dt
                with Timer() as tm:
                    zs = sp(z3, torch.ones(len(cb), dtype=torch.long, device=dev)).float()
                times["t_spade"] += tm.dt
                with Timer() as tm:
                    noise = torch.randn(zs.shape, generator=g, device=dev)
                    zd = model.sample(zs * scale, model.cond(z3 * scale, zs * scale), noise) / scale
                times["t_diffusion"] += tm.dt
                with Timer() as tm:
                    m_d.add((ae.decode(zd).float() + 1) / 2, cb)
                times["t_decode"] += tm.dt
                if m_s is not None:
                    m_s.add((ae.decode(zs).float() + 1) / 2, cb)
        times["n_patches"] = len(corners)
        return m_d.result(), (m_s.result() if m_s is not None else None), times

    def ae_ceiling(vol):
        corners = grid_corners(vol.shape, P, S)
        x = torch.from_numpy(np.ascontiguousarray(vol))[None].to(dev)
        m = PatchMerger(vol.shape, P, dev)
        for i in range(0, len(corners), B):
            cb = corners[i:i + B]
            with torch.no_grad(), autocast(dev, cfg["bf16"]):
                m.add((ae.decode(ae_encode_mu(ae, extract_patches(x, cb, P))).float() + 1) / 2, cb)
        return m.result()

    rows = []
    for s in pr.subjects(split):
        sm = pr.meta["subjects"][s]
        v3 = (np.asarray(pr.volume(s, "3t"), dtype=np.float32) + 1) / 2
        v7 = (np.asarray(pr.volume(s, "7t"), dtype=np.float32) + 1) / 2
        if mode in ("3t", "both"):
            v3 = hist_match(v3, ref["lm_3t"], thr)
        if mode == "both":
            v7 = hist_match(v7, ref["lm_7t"], thr)
        img_d, img_s, times = run_patches(v3 * 2 - 1, True)
        n_patches = times.pop("n_patches")
        img_d, img_s = img_d[None], img_s[None]
        img_c = ae_ceiling(v7 * 2 - 1)[None] if args.ceiling else None
        pads = sm["pads"]
        gt = torch.from_numpy(unpad(v7, pads)).to(dev)
        cut = lambda im: torch.from_numpy(unpad(im[0, 0].float().cpu().numpy(), pads)).to(dev)
        total = sum(times.values())
        for name, im in (("spade", img_s), ("diffusion", img_d), ("ae_ceiling", img_c)):
            if im is None:
                continue
            r = {"subject": s, "method": name, **metrics(cut(im), gt, thr)}
            if name == "diffusion":
                r.update({k: round(v, 3) for k, v in times.items()}, t_total=round(total, 3), n_patches=n_patches)
            rows.append(r)
            print(f"  {s} {name:>10}: " + "  ".join(f"{k}={v:.4f}" for k, v in r.items() if isinstance(v, float) and not k.startswith("t_")))
        print(f"  {s} time: " + "  ".join(f"{k}={v:.2f}s" for k, v in times.items()) + f"  total={total:.2f}s  ({n_patches} patches)")
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
