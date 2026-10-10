"""Network builders (MONAI AutoencoderKL / DiffusionModelUNet, 3D) and tiled encode / decode for whole volumes."""
import itertools

import torch
from monai.networks.nets import AutoencoderKL, DiffusionModelUNet


def build_ae(c):
    """c = cfg['ae']. len(channels)-1 downsamplings -> factor 2**(len-1) per axis."""
    return AutoencoderKL(spatial_dims=3, in_channels=1, out_channels=1,
                         channels=tuple(c["channels"]), num_res_blocks=c.get("num_res_blocks", 2),
                         attention_levels=tuple(c.get("attention_levels", [False] * len(c["channels"]))),
                         latent_channels=c["latent_channels"], norm_num_groups=c.get("norm_num_groups", 32),
                         with_encoder_nonlocal_attn=c.get("nonlocal_attn", False),
                         with_decoder_nonlocal_attn=c.get("nonlocal_attn", False),
                         use_checkpoint=c.get("use_checkpoint", False))


def ae_factor(c):
    return 2 ** (len(c["channels"]) - 1)


def build_unet(c, in_channels, out_channels):
    """c = cfg['diffusion']['unet']"""
    return DiffusionModelUNet(spatial_dims=3, in_channels=in_channels, out_channels=out_channels,
                              channels=tuple(c["channels"]), num_res_blocks=c.get("num_res_blocks", 2),
                              attention_levels=tuple(c["attention_levels"]),
                              norm_num_groups=c.get("norm_num_groups", 32),
                              num_head_channels=c.get("num_head_channels", 32))


def load_ae(cfg, ckpt, device):
    ae = build_ae(cfg["ae"]).to(device).eval()
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    ae.load_state_dict(sd["ae"] if "ae" in sd else sd)
    for p in ae.parameters():
        p.requires_grad_(False)
    return ae


def ae_encode_mu(ae, x):
    """posterior mean (deterministic), as in the 2D pipeline"""
    h = ae.encoder(x)
    return ae.quant_conv_mu(h)


def _windows(size, tile, ov):
    """yield (window_start, window_len, keep_start, keep_len) along one axis, keep = window core without the overlap"""
    if size <= tile:
        yield 0, size, 0, size
        return
    core = tile - 2 * ov
    assert core > 0, "tile must be larger than 2 * overlap"
    s = 0
    while s < size:
        k_len = min(core, size - s)
        w0 = min(max(s - ov, 0), size - tile)
        yield w0, tile, s, k_len
        s += core


@torch.no_grad()
def tiled(fn, x, tile, overlap, scale, out_channels):
    """apply fn tile-by-tile over a (1,C,D,H,W) volume and stitch the cores.
    scale = output size / input size per axis (1/8 for encode, 8 for decode). tile, overlap in input voxels."""
    assert x.shape[0] == 1
    S = x.shape[2:]
    out_size = [int(round(s * scale)) for s in S]
    out = None
    axes = [list(_windows(S[d], tile, overlap)) for d in range(3)]
    for wins in itertools.product(*axes):
        sl_in = (slice(None), slice(None)) + tuple(slice(w0, w0 + wl) for w0, wl, _, _ in wins)
        y = fn(x[sl_in])
        if out is None:
            out = torch.zeros((1, out_channels, *out_size), device=y.device, dtype=y.dtype)
        src = (slice(None), slice(None)) + tuple(slice(int(round((ks - w0) * scale)), int(round((ks - w0 + kl) * scale)))
                                                for w0, _, ks, kl in wins)
        dst = (slice(None), slice(None)) + tuple(slice(int(round(ks * scale)), int(round((ks + kl) * scale))) for _, _, ks, kl in wins)
        out[dst] = y[src]
    return out
