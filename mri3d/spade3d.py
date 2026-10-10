"""3D SPADE generator for latent-to-latent domain translation.

Ported from ALDM (Kim & Park, WACV 2024; github.com/jongdory/ALDM, ldm/models/normalization.py).
Same blocks; the per-sample class loop is vectorised by class, and num_classes defaults to 2 (1 = 7T, as in the 2D pipeline)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc, kernel_size=3, nhidden=64):
        super().__init__()
        pw = kernel_size // 2
        self.param_free_norm = nn.InstanceNorm3d(norm_nc, affine=False)
        self.mlp_shared = nn.Sequential(nn.Conv3d(label_nc, nhidden, kernel_size, padding=pw), nn.ReLU())
        self.mlp_gamma = nn.Conv3d(nhidden, norm_nc, kernel_size, padding=pw)
        self.mlp_beta = nn.Conv3d(nhidden, norm_nc, kernel_size, padding=pw)

    def forward(self, x):
        actv = self.mlp_shared(x)
        return self.param_free_norm(x) * (1 + self.mlp_gamma(actv)) + self.mlp_beta(actv)


class SPADEMultiClass(nn.Module):
    """one SPADE per target class; samples are routed by their class label y (B,)"""

    def __init__(self, num_classes, norm_nc, label_nc):
        super().__init__()
        self.spades = nn.ModuleList([SPADE(norm_nc, label_nc) for _ in range(num_classes)])

    def forward(self, x, y):
        out = torch.empty_like(x)
        for c in torch.unique(y).tolist():
            if c >= len(self.spades):
                raise ValueError(f"class {c} out of range ({len(self.spades)} classes)")
            idx = (y == c).nonzero(as_tuple=True)[0]
            out[idx] = self.spades[c](x[idx])
        return out


class SPADEResnetBlock(nn.Module):
    def __init__(self, num_classes, fin, fout):
        super().__init__()
        fmid = min(fin, fout)
        self.conv_0 = nn.Conv3d(fin, fmid, 3, padding=1)
        self.conv_1 = nn.Conv3d(fmid, fout, 3, padding=1)
        self.conv_s = nn.Conv3d(fin, fout, 1, bias=False) if fin != fout else None
        self.norm_0 = SPADEMultiClass(num_classes, fin, fin)
        self.norm_1 = SPADEMultiClass(num_classes, fmid, fmid)

    def forward(self, x, y):
        xs = self.conv_s(x) if self.conv_s is not None else x
        dx = self.conv_0(F.leaky_relu(self.norm_0(x, y), 0.2))
        dx = self.conv_1(F.leaky_relu(self.norm_1(dx, y), 0.2))
        return xs + dx


class SPADEGenerator3D(nn.Module):
    def __init__(self, num_classes=2, z_dim=4, nf=64):
        super().__init__()
        self.block = nn.ModuleList([
            SPADEResnetBlock(num_classes, z_dim, nf),
            SPADEResnetBlock(num_classes, nf, nf * 2),
            SPADEResnetBlock(num_classes, nf * 2, nf),
            SPADEResnetBlock(num_classes, nf, z_dim),
        ])

    def forward(self, x, y):
        for b in self.block:
            x = b(x, y)
        return x
