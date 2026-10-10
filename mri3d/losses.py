"""3D losses and metrics: MIND (6-neighbour), low-frequency L1, registration-corrected L1 (RegGAN-style), SSIM."""
import torch
import torch.nn as nn
import torch.nn.functional as F

_SHIFTS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


def mind3d(x, eps=1e-3):
    """MIND descriptor (Heinrich et al., MedIA 2012), 6-neighbourhood, 3x3x3 patch distance. x: (B,1,D,H,W) -> (B,6,D,H,W)."""
    D, H, W = x.shape[-3:]
    p = F.pad(x, (1, 1, 1, 1, 1, 1), mode="replicate")
    dist = torch.cat([F.avg_pool3d((x - p[:, :, 1 + a:1 + a + D, 1 + b:1 + b + H, 1 + c:1 + c + W]) ** 2, 3, stride=1, padding=1)
                      for a, b, c in _SHIFTS], 1)
    var = dist.mean(1, keepdim=True).clamp_min(eps)
    m = torch.exp(-dist / var)
    return m / m.amax(1, keepdim=True).clamp_min(eps)


def masked_mean(t, mask):
    return (t * mask).sum() / mask.sum().clamp_min(1.0)


def mind_loss(pred, ref, mask):
    """mean over mask of mean_k |MIND_k(pred) - MIND_k(ref)|; pred/ref in [0,1]"""
    return masked_mean((mind3d(pred) - mind3d(ref)).abs().mean(1, keepdim=True), mask)


def lf_loss(pred, ref, mask, factor=8):
    """L1 between factor^3 average-pooled images inside the (pooled) mask: anchors large-scale intensity / contrast"""
    m = (F.avg_pool3d(mask, factor) > 0.5).float()
    return masked_mean((F.avg_pool3d(pred, factor) - F.avg_pool3d(ref, factor)).abs(), m)


def border_mask(shape, margin, device):
    """1 inside, 0 within `margin` voxels of the crop border (decoded latent crops have edge artefacts)"""
    m = torch.zeros(shape, device=device)
    if margin <= 0:
        return m + 1
    m[..., margin:-margin, margin:-margin, margin:-margin] = 1
    return m


# ----------------------------------------------------------------------------- registration-corrected L1
class RegNet3D(nn.Module):
    """small 3D U-Net: [moving, fixed] (B,2,D,H,W) -> displacement (B,3,D,H,W) in voxels (d, h, w order).
    last conv is zero-initialised, so training starts from the identity transform."""

    def __init__(self, ch=(8, 16, 32), max_disp=4.0):
        super().__init__()

        def blk(i, o, s=1):
            return nn.Sequential(nn.Conv3d(i, o, 3, s, 1), nn.LeakyReLU(0.2, inplace=True))
        self.e0, self.e1, self.e2 = blk(2, ch[0]), blk(ch[0], ch[1], 2), blk(ch[1], ch[2], 2)
        self.d1, self.d0 = blk(ch[2] + ch[1], ch[1]), blk(ch[1] + ch[0], ch[0])
        self.out = nn.Conv3d(ch[0], 3, 3, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.max_disp = float(max_disp)

    def forward(self, moving, fixed):
        e0 = self.e0(torch.cat([moving, fixed], 1))
        e1 = self.e1(e0)
        e2 = self.e2(e1)
        up = lambda a, ref: F.interpolate(a, size=ref.shape[-3:], mode="trilinear", align_corners=False)
        d1 = self.d1(torch.cat([up(e2, e1), e1], 1))
        d0 = self.d0(torch.cat([up(d1, e0), e0], 1))
        return self.max_disp * torch.tanh(self.out(d0))


def warp3d(img, disp):
    """warp img (B,C,D,H,W) by voxel displacements disp (B,3,D,H,W) ordered (d, h, w)"""
    B, _, D, H, W = img.shape
    zs, ys, xs = torch.meshgrid(*(torch.linspace(-1, 1, n, device=img.device, dtype=img.dtype) for n in (D, H, W)), indexing="ij")
    grid = torch.stack([xs, ys, zs], -1).unsqueeze(0).expand(B, D, H, W, 3)
    off = torch.stack([disp[:, 2] * 2 / max(W - 1, 1), disp[:, 1] * 2 / max(H - 1, 1), disp[:, 0] * 2 / max(D - 1, 1)], -1)
    return F.grid_sample(img, grid + off, mode="bilinear", padding_mode="border", align_corners=True)


def smoothness(disp):
    return sum((disp.diff(dim=d)).pow(2).mean() for d in (2, 3, 4))


# ----------------------------------------------------------------------------- SSIM / metrics
def _gauss1d(size, sigma, device, dtype):
    x = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    return g / g.sum()


def ssim3d_map(a, b, data_range=1.0, size=7, sigma=1.5):
    """local SSIM map (B,1,D,H,W) with a separable 3D Gaussian window; inputs (B,1,D,H,W)"""
    g = _gauss1d(size, sigma, a.device, a.dtype)
    def blur(x):
        x = F.conv3d(x, g.view(1, 1, -1, 1, 1), padding=(size // 2, 0, 0))
        x = F.conv3d(x, g.view(1, 1, 1, -1, 1), padding=(0, size // 2, 0))
        return F.conv3d(x, g.view(1, 1, 1, 1, -1), padding=(0, 0, size // 2))
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mu_a, mu_b = blur(a), blur(b)
    saa, sbb, sab = blur(a * a) - mu_a ** 2, blur(b * b) - mu_b ** 2, blur(a * b) - mu_a * mu_b
    return ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (saa + sbb + c2))
