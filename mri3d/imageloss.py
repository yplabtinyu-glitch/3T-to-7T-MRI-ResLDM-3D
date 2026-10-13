"""Image-space losses on decoded latent patches (shared by SPADE and diffusion training).

  MIND  : structure of D(pred) vs the same-subject 3T, D(z_3T)          (lambda_mind)
  LF    : low-frequency intensity of D(pred) vs a reference D(z_ref)     (lambda_lf; reference = SPADE for diffusion)
  CORR  : RegGAN-style registration-corrected L1 to the 7T, D(z_7T)      (lambda_corr; + smoothness of the displacement)
All references are decoded without gradient; the prediction is decoded with gradient (decoder weights stay frozen).
With `sub_crop: null` (default) the losses use the whole decoded 64³ patch; an int restricts them to a random latent sub-crop. `margin` voxels at the border are ignored."""
import torch
import torch.nn as nn

from .losses import RegNet3D, border_mask, lf_loss, masked_mean, mind_loss, smoothness, warp3d


class ImageLosses(nn.Module):
    def __init__(self, ae, scale, c, factor):
        super().__init__()
        self.ae, self.scale, self.f = ae, float(scale), int(factor)
        self.l_mind = float(c.get("lambda_mind", 0.0))
        self.l_lf = float(c.get("lambda_lf", 0.0))
        self.lf_factor = int(c.get("lf_factor", 8))
        self.l_corr = float(c.get("lambda_corr", 0.0))
        self.corr_smooth = float(c.get("corr_smooth", 1.0))
        self.thr = float(c.get("brain_thr", 0.02))
        self.margin = int(c.get("margin", 8))
        self.sub = c.get("sub_crop", None)
        self.reg = RegNet3D(max_disp=c.get("corr_max_disp", 4.0)) if self.l_corr > 0 else None

    @property
    def active(self):
        return self.l_mind > 0 or self.l_lf > 0 or self.l_corr > 0

    def _img(self, z):
        return (self.ae.decode(z / self.scale).float() + 1) / 2

    def _crop(self, tensors):
        L = tensors[0].shape[-1]
        s = min(int(self.sub or L), L)
        if s == L:
            return tensors
        o = [int(torch.randint(0, L - s + 1, (1,))) for _ in range(3)]
        sl = (slice(None), slice(None)) + tuple(slice(a, a + s) for a in o)
        return [t[sl] for t in tensors]

    def forward(self, pred, z3t, zref_lf, z7t):
        """all inputs are scaled latents (B,C,L,L,L). returns (weighted sum, dict of detached components)"""
        out = {}
        if not self.active:
            return pred.new_zeros(()), out
        pred, z3t, zref_lf, z7t = self._crop([pred, z3t, zref_lf, z7t])
        img = self._img(pred)
        with torch.no_grad():
            i3 = self._img(z3t).clamp(0, 1)
        edge = border_mask(img.shape, self.margin, img.device)
        bm = (i3 > self.thr).float() * edge
        total = img.new_zeros(())
        if self.l_mind > 0:
            l = mind_loss(img, i3, bm)
            total = total + self.l_mind * l
            out["mind"] = l.detach()
        if self.l_lf > 0:
            with torch.no_grad():
                ir = self._img(zref_lf).clamp(0, 1)
            l = lf_loss(img, ir, bm, self.lf_factor)
            total = total + self.l_lf * l
            out["lf"] = l.detach()
        if self.l_corr > 0:
            with torch.no_grad():
                i7 = self._img(z7t).clamp(0, 1)
            disp = self.reg(img, i7)
            m7 = (i7 > self.thr).float() * edge
            l = masked_mean((warp3d(img, disp) - i7).abs(), m7)
            ls = smoothness(disp)
            total = total + self.l_corr * (l + self.corr_smooth * ls)
            out["corr"], out["smooth"], out["disp"] = l.detach(), ls.detach(), disp.detach().abs().mean()
        return total, out
