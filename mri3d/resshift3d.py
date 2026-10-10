"""ResShift (Yue et al., NeurIPS 2023) in 3D latent space, same schedule / sampler as the 2D pipeline.

forward : z_t = x0 + eta_t (y0 - x0) + kappa sqrt(eta_t) eps,   x0 = scaled 7T latent, y0 = scaled SPADE latent
model   : UNet([z_t || condition], t) -> x0_hat
sampler : z_T = y0 + kappa sqrt(eta_T) eps;  z_{t-1} ~ N(eta_{t-1}/eta_t z_t + alpha_t/eta_t x0_hat, kappa^2 eta_{t-1}/eta_t alpha_t)
"""
import numpy as np
import torch
import torch.nn as nn

CONDITIONS = {"none": 0, "3t": 1, "spade": 1, "all": 2}  # number of latent blocks concatenated to z_t


class ResShift3D(nn.Module):
    def __init__(self, unet, latent_channels, T=4, p=0.3, kappa=1.0, condition="3t"):
        super().__init__()
        assert condition in CONDITIONS, condition
        self.unet, self.C, self.T, self.condition = unet, latent_channels, int(T), condition
        k = torch.as_tensor(list(kappa) if hasattr(kappa, "__iter__") else kappa, dtype=torch.float32)
        self.register_buffer("kappa", k, persistent=False)
        self._schedule(self.T, p, float(k.mean()))

    def _schedule(self, T, p, kappa):
        eta_1 = min((0.04 / kappa) ** 2, 0.001)
        eta_T = 0.999
        b0 = np.exp(np.log(eta_T / eta_1) / (2 * (T - 1)))
        etas = np.array([(np.sqrt(eta_1) * b0 ** (((t - 1) / (T - 1)) ** p * (T - 1))) ** 2 for t in range(1, T + 1)])
        etas[-1] = eta_T
        prev = np.concatenate([[0.0], etas[:-1]])
        alphas = etas - prev
        assert np.all(alphas > 0), "non-increasing eta schedule -- check T / p / kappa"
        f = lambda a: torch.tensor(a, dtype=torch.float32)
        self.register_buffer("etas", f(etas))
        self.register_buffer("etas_prev", f(prev))
        self.register_buffer("alphas", f(alphas))

    def _k(self, ndim):
        return self.kappa if self.kappa.ndim == 0 else self.kappa.view(1, -1, *([1] * (ndim - 2)))

    @staticmethod
    def _at(arr, t, ndim):
        return arr.gather(0, t).view(-1, *([1] * (ndim - 1)))

    def q_sample(self, x0, y0, t, noise):
        eta = self._at(self.etas, t, x0.ndim)
        return x0 + eta * (y0 - x0) + self._k(x0.ndim) * eta.sqrt() * noise

    def cond(self, z3t, zsp):
        return {"none": [], "3t": [z3t], "spade": [zsp], "all": [z3t, zsp]}[self.condition]

    def predict_x0(self, z_t, t, cond_list):
        return self.unet(torch.cat([z_t] + cond_list, 1), t)

    @torch.no_grad()
    def sample(self, y0, cond_list, noise=None):
        k = self._k(y0.ndim)
        z = y0 + k * self.etas[-1].sqrt() * (torch.randn_like(y0) if noise is None else noise)
        for i in range(self.T - 1, -1, -1):
            t = torch.full((y0.shape[0],), i, device=y0.device, dtype=torch.long)
            x0 = self.predict_x0(z, t, cond_list)
            if i > 0:
                eta, prev, a = self.etas[i], self.etas_prev[i], self.alphas[i]
                z = (prev / eta) * z + (a / eta) * x0 + torch.sqrt((k ** 2) * (prev / eta) * a) * torch.randn_like(z)
            else:
                z = x0
        return z
