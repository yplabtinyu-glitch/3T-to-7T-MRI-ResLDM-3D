"""End-to-end smoke test on tiny synthetic volumes (CPU, a few minutes). Checks the math helpers, then runs every script once.
usage: python tests/smoke_test.py            (writes to a temporary folder, deleted afterwards unless --keep)"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import yaml  # noqa: E402

from mri3d.data import hist_match, landmarks  # noqa: E402
from mri3d.losses import mind3d, warp3d  # noqa: E402
from mri3d.nets import tiled  # noqa: E402
from mri3d.resshift3d import ResShift3D  # noqa: E402


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        sys.exit(1)


def unit_tests():
    torch.manual_seed(0)
    # tiled processing == whole-volume processing when overlap >= receptive field
    enc = torch.nn.Sequential(torch.nn.Conv3d(1, 4, 3, padding=1), torch.nn.Conv3d(4, 4, 3, padding=1), torch.nn.AvgPool3d(4)).eval()
    x = torch.randn(1, 1, 48, 64, 40)
    with torch.no_grad():
        full = enc(x)
    t = tiled(enc, x, 32, 8, 1 / 4, 4)
    check("tiled encode == whole volume", torch.allclose(full, t, atol=1e-5), f"(max diff {(full - t).abs().max():.2e})")
    dec = torch.nn.Sequential(torch.nn.Upsample(scale_factor=4), torch.nn.Conv3d(4, 1, 3, padding=1)).eval()
    with torch.no_grad():
        full = dec(t)
    t2 = tiled(dec, t, 8, 2, 4, 1)
    check("tiled decode == whole volume", torch.allclose(full, t2, atol=1e-5), f"(max diff {(full - t2).abs().max():.2e})")
    # warp: zero displacement = identity; integer shift is undone
    img = torch.rand(1, 1, 20, 24, 28)
    check("warp3d identity", torch.allclose(warp3d(img, torch.zeros(1, 3, 20, 24, 28)), img, atol=1e-5))
    d = torch.zeros(1, 3, 20, 24, 28)
    d[:, 2] = 2.0
    sh = torch.roll(img, 2, dims=4)
    check("warp3d undoes a 2-voxel shift along w", torch.allclose(warp3d(sh, d)[..., 3:-3, 3:-3, 3:-3], img[..., 3:-3, 3:-3, 3:-3], atol=1e-5))
    # MIND is invariant to a linear intensity change (as long as the local variance is above eps=1e-3)
    v = torch.rand(1, 1, 24, 24, 24)
    check("MIND3D invariant to a*x+b", torch.allclose(mind3d(v), mind3d(2.5 * v + 0.3), atol=1e-4))
    # ResShift schedule identical to the 2D pipeline's formula
    T, p, k = 4, 0.3, 0.6
    eta_1 = min((0.04 / k) ** 2, 0.001)
    b0 = np.exp(np.log(0.999 / eta_1) / (2 * (T - 1)))
    ref = np.array([(np.sqrt(eta_1) * b0 ** (((t - 1) / (T - 1)) ** p * (T - 1))) ** 2 for t in range(1, T + 1)])
    ref[-1] = 0.999
    rs = ResShift3D(torch.nn.Identity(), 4, T, p, k, "none")
    check("ResShift eta schedule", np.allclose(rs.etas.numpy(), ref, atol=1e-7), str(np.round(ref, 5)))
    # histogram matching moves the foreground landmarks onto the reference
    a = np.clip(np.random.default_rng(0).gamma(2, 0.15, (30, 30, 30)), 0, 1).astype(np.float32)
    lm_ref = landmarks(np.clip(a ** 0.7, 0, 1), 0.02)
    m = hist_match(a, lm_ref, 0.02)
    check("histogram matching", np.allclose(landmarks(m, 0.02), lm_ref, atol=0.02), f"(max landmark error {np.abs(landmarks(m, 0.02) - lm_ref).max():.3f})")


def make_data(root, n=4, shape=(40, 48, 36)):
    import nibabel as nib
    rng = np.random.default_rng(1)
    rows = ["subject,split,t3,t7"]
    splits = ["train", "train", "val", "test"][:n]
    for i, sp in enumerate(splits):
        x = torch.from_numpy(rng.random((1, 1, *shape)).astype(np.float32))
        for _ in range(3):
            x = F.avg_pool3d(x, 5, 1, 2)
        x = (x - x.min()) / (x.max() - x.min())
        brain = (x > 0.35).float()
        v7 = (x * brain * 1000).numpy()[0, 0]
        v3 = ((x ** 1.5) * brain * 800).numpy()[0, 0]          # different contrast, same anatomy
        aff = np.diag([0.7, 0.7, 0.7, 1.0])
        for mod, v in (("3t", v3), ("7t", v7)):
            nib.save(nib.Nifti1Image(v, aff), os.path.join(root, f"s{i}_{mod}.nii.gz"))
        rows.append(f"s{i},{sp},{root}/s{i}_3t.nii.gz,{root}/s{i}_7t.nii.gz")
    open(os.path.join(root, "pairs.csv"), "w").write("\n".join(rows) + "\n")


def tiny_config(root):
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "base.yaml")))
    cfg.update(out_dir=os.path.join(root, "runs"), num_workers=0, bf16=False)
    cfg["data"].update(pairs_csv=os.path.join(root, "pairs.csv"), data_dir=root, pad_multiple=16)
    cfg["ae"].update(channels=[8, 16, 16], attention_levels=[False, False, False], norm_num_groups=8, use_checkpoint=False)
    cfg["ae"]["train"].update(patch=32, adv_layers=2, batch_size=2, samples_per_epoch=4, val_samples=2, epochs=2, adv_start_epoch=1)
    cfg["latent"].update(tile=32, overlap=8)
    cfg["spade"]["nf"] = 8
    cfg["spade"]["train"].update(crop=8, batch_size=2, samples_per_epoch=4, val_samples=2, epochs=2)
    cfg["spade"]["losses"].update(margin=2, sub_crop=6)
    cfg["diffusion"]["condition"] = "all"
    cfg["diffusion"]["unet"].update(channels=[16, 16, 32], attention_levels=[False, False, True], norm_num_groups=8, num_head_channels=8)
    cfg["diffusion"]["train"].update(crop=8, batch_size=2, samples_per_epoch=4, val_samples=2, epochs=2, ckpt_every=1)
    cfg["diffusion"]["losses"].update(lambda_corr=1.0, margin=2, sub_crop=6, lf_factor=4)
    cfg["infer"].update(latent_tile=8, latent_overlap=2)
    p = os.path.join(root, "tiny.yaml")
    yaml.safe_dump(cfg, open(p, "w"))
    return p, cfg


def run(args):
    print("\n$ " + " ".join(args), flush=True)
    r = subprocess.run([sys.executable] + args, cwd=ROOT)
    check(os.path.basename(args[0]), r.returncode == 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    unit_tests()
    root = tempfile.mkdtemp(prefix="resldm3d_smoke_")
    try:
        make_data(root)
        cp, cfg = tiny_config(root)
        run(["scripts/inspect_volumes.py", "--config", cp])
        run(["scripts/prepare_data.py", "--config", cp])
        run(["scripts/train_ae3d.py", "--config", cp])
        run(["scripts/train_ae3d.py", "--config", cp, "--resume"])
        run(["scripts/cache_latents3d.py", "--config", cp, "--stage", "ae"])
        run(["scripts/train_spade3d.py", "--config", cp])
        run(["scripts/cache_latents3d.py", "--config", cp, "--stage", "spade"])
        run(["scripts/train_diffusion3d.py", "--config", cp, "--name", "p5_test"])
        ck = os.path.join(cfg["out_dir"], "p5_test", "best.pth")
        run(["scripts/infer_evaluate3d.py", "--config", cp, "--ckpt", ck, "--ceiling"])
        ev = os.path.join(cfg["out_dir"], "p5_test", "eval_test_hmboth")
        check("metrics.csv and NIfTI written", os.path.exists(os.path.join(ev, "metrics.csv")) and os.path.exists(os.path.join(ev, "s3_synth7t.nii.gz")))
        import nibabel as nib
        out_shape = nib.load(os.path.join(ev, "s3_synth7t.nii.gz")).shape
        check("output has the original (unpadded) shape", out_shape == (40, 48, 36), str(out_shape))
        log = open(os.path.join(cfg["out_dir"], "p5_test", "log.csv")).read().strip().splitlines()
        check("one log row per epoch", len(log) == 1 + cfg["diffusion"]["train"]["epochs"], log[0])
        print(f"\nall smoke tests passed ({root})")
    finally:
        if not a.keep:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
