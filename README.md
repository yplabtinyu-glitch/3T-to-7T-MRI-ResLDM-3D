# 3T-to-7T MRI ResLDM — 3D

Full-3D version of the unpaired 3T→7T brain MRI synthesis pipeline in
[3T-to-7T-MRI-ResLDM](https://github.com/yplabtinyu-glitch/3T-to-7T-MRI-ResLDM) (2D, slice-wise).
Same three stages and the same losses, but every network works on 3D volumes, which removes the
slice-to-slice inconsistency of the 2D model.

```
3T volume ──► 3D AE encoder ──► z_3T ──► 3D SPADE ──► y0 = z_SPADE ──► 3D ResShift UNet (T steps) ──► 3D AE decoder ──► synthetic 7T
                (frozen)                  (frozen)                       condition: z_3T / z_SPADE / both / none     (frozen)
```

| stage | network | training data | loss |
|---|---|---|---|
| 1. AE | MONAI `AutoencoderKL`, 3D, 8x per axis, 4 latent channels | random 3D patches of 3T and 7T | L1 + KL (+ 3D PatchGAN) |
| 2. SPADE | 3D SPADE generator (ported from ALDM) | random crops of cached latents | latent L1 to z_7T + MIND(D(SPADE), D(z_3T)) |
| 3. ResShift | MONAI `DiffusionModelUNet`, 3D | random crops of cached latents | latent MSE + MIND vs 3T + low-frequency L1 vs SPADE (+ optional registration-corrected L1 vs 7T) |

What is carried over from the 2D pipeline unchanged: the ResShift schedule and sampler (checked numerically
against the 2D code in the smoke test), `y0 = SPADE latent`, per-channel kappa from `std(z_SPADE − z_7T)`,
MIND structure loss (now 6-neighbour 3D), the low-frequency anchor to SPADE, the RegGAN-style registration-corrected
L1 (`lambda_corr`, off by default), Nyúl histogram matching with the `none / 3t / both` evaluation modes, and
one CSV row per epoch for every training log.

## Install

```bash
conda create -n resldm3d python=3.10 -y && conda activate resldm3d
pip install torch --index-url https://download.pytorch.org/whl/cu121   # match the machine's CUDA
pip install -r requirements.txt
python tests/smoke_test.py      # a few minutes on CPU: math checks + every script once on tiny synthetic volumes
```

## Data

`data/pairs.csv` (see `data/pairs_example.csv`):

```
subject,split,t3,t7
sub001,train,/path/sub001_3T_to7T.nii.gz,/path/sub001_7T.nii.gz
...
```

* `t3` must already be registered onto the grid of `t7` (same shape and affine) — this is the output of the
  separate SynthMorph registration pipeline. `inspect_volumes.py` checks this.
* `split`: `train` / `val` / `test`. Test subjects should be same-subject 3T/7T pairs.

## Run order

```bash
C=configs/base.yaml
python scripts/inspect_volumes.py   --config $C                  # shapes, spacing, grid check
python scripts/prepare_data.py      --config $C                  # normalise + pad -> data/prepped/*.npy, data/hist_ref.npz
python scripts/train_ae3d.py        --config $C                  # -> runs/base/ae/ae_best.pth
python scripts/cache_latents3d.py   --config $C --stage ae       # -> runs/base/latents/*_z3t.npy, *_z7t.npy, scale
python scripts/train_spade3d.py     --config $C                  # -> runs/base/spade/spade_best.pth
python scripts/cache_latents3d.py   --config $C --stage spade    # -> runs/base/latents/*_zspade.npy, kappa_auto
python scripts/train_diffusion3d.py --config $C --name p4_mindlf # -> runs/base/p4_mindlf/best.pth
python scripts/infer_evaluate3d.py  --config $C --ckpt runs/base/p4_mindlf/best.pth --ceiling
```

Every training script accepts `--resume` and stops early after `patience` epochs without validation improvement.
`scripts/run_all.sh configs/base.yaml` runs the whole chain.

Experiments are config changes: copy `configs/base.yaml`, change e.g. `diffusion.condition`, `diffusion.T`,
`diffusion.kappa_mult` or `diffusion.losses.lambda_corr`, and train with a new `--name`.
The AE / SPADE / latent cache are shared by every diffusion run with the same `out_dir`.

## Outputs

* `runs/<exp>/<stage>/log.csv` — one row per epoch: epoch means of every loss term, train and val.
* `runs/<exp>/<name>/eval_<split>_hm<mode>/metrics.csv` — per subject and overall, for SPADE-only, diffusion
  and (with `--ceiling`) the AE reconstruction of the 7T: `ssim3d`, `ssim3d_brain`, `psnr`, `mae`, `mae_brain`,
  `ssim2d_axial` (mean 2D SSIM over axial slices, for comparison with the 2D pipeline), and seconds per volume for
  encode / SPADE / diffusion / decode.
* `<subject>_synth7t.nii.gz` — synthetic 7T in the original 7T grid (values in [0, 1]).

## Memory and speed knobs

The defaults are a starting point for an 80–141 GB GPU; **they have not yet been run at full scale on a GPU**
(only the CPU smoke test). Things to adjust first if memory runs out:

| knob | effect |
|---|---|
| `ae.train.patch`, `ae.train.batch_size` | AE activation memory (3D patches scale with patch³) |
| `ae.use_checkpoint` | gradient checkpointing in the AE; needed for the image-space losses of stages 2–3 |
| `*.losses.sub_crop` | image losses are computed on a `sub_crop³` latent sub-crop (12 latent = 96³ voxels) |
| `latent.tile` / `latent.overlap` | tiled encode / decode of whole volumes (exact when overlap ≥ receptive field) |
| `infer.latent_tile` | run the UNet on latent tiles instead of the whole latent volume |
| `diffusion.train.compile` | `torch.compile(mode="reduce-overhead")` (CUDA graphs) against CPU launch overhead — untested |
| `bf16` | bf16 autocast on CUDA |

## For reviewers — design choices worth checking

1. **Patch-based training, whole-volume inference.** The AE trains on 96³ patches, SPADE / diffusion on 16³ latent
   crops (128³ voxels); inference encodes / decodes the whole volume in overlapping tiles and runs the UNet on the
   whole latent volume (attention only at the deepest level). Is this train/test size mismatch acceptable, or
   should inference also be tiled?
2. **Image-space losses on decoded crops.** MIND / LF / registration losses decode the predicted latent through the
   frozen decoder with gradient. Crop borders are excluded (`margin`) because decoded crops have edge artefacts.
3. **Latent cache.** AE and SPADE are frozen, so their outputs are cached once; diffusion training reads only
   latents (the training-side acceleration of the 2D pipeline, now also for SPADE).
4. **Scale factor.** `scale = 1 / std(z_7T)` over the training set, applied before diffusion; kappa is measured on
   the scaled latents.
5. **Data volume.** 3D training sees one sample per subject per crop; with few subjects the AE may need strong
   augmentation (none implemented yet).

## Repository layout

```
mri3d/
  data.py        pair list, NIfTI I/O, normalisation, Nyúl histogram matching, padding, patch / latent-crop datasets
  nets.py        MONAI AE / UNet builders, posterior-mean encode, tiled whole-volume processing
  spade3d.py     3D SPADE generator (from ALDM)
  resshift3d.py  ResShift schedule, forward process, sampler, conditioning
  losses.py      3D MIND, low-frequency L1, registration net + warp, SSIM
  imageloss.py   image-space losses on decoded latent crops
  utils.py       config, seeding, per-epoch CSV log, bf16 autocast, checkpoints
scripts/         inspect / prepare / train_ae3d / cache_latents3d / train_spade3d / train_diffusion3d / infer_evaluate3d
configs/base.yaml
tests/smoke_test.py
```

## Acknowledgments

* [ALDM](https://github.com/jongdory/ALDM) — Kim & Park, WACV 2024 (3D SPADE + latent diffusion; `mri3d/spade3d.py` is ported from it)
* [ResShift](https://arxiv.org/abs/2307.12348) — Yue et al., NeurIPS 2023
* [MIND](https://doi.org/10.1016/j.media.2012.05.006) — Heinrich et al., Medical Image Analysis 2012
* [RegGAN](https://arxiv.org/abs/2110.06465) — Kong et al., NeurIPS 2021 (registration-corrected loss)
* Nyúl, Udupa & Zhang, IEEE TMI 2000 (histogram standardisation)
* [MONAI](https://monai.io/) `AutoencoderKL`, `DiffusionModelUNet`, `PatchDiscriminator`
