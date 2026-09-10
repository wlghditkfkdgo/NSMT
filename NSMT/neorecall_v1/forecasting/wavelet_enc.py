"""Interpretation-B wavelet encoding: the SCALE axis becomes the spiking simulation axis.

Direct coding wastes the T axis on a repeat -- every simulation step sees the same input.
Here an undecimated a-trous wavelet transform splits the series into J detail bands plus one
approximation, ALL of length L (no decimation), and the J detail bands BECOME the T axis:
at simulation step j the network sees detail band j. The approximation (lowest-frequency)
band is routed to the Neocortex instead, matching its stated role (MASTER 2.3, "window-internal
slow trend") with an analytic rather than a learned signal.

The transform is ADDITIVE by construction -- d_j = c_{j-1} - c_j, so

    x = cA_J + sum_j d_j        (exact)

which makes "each timestep sees a different frequency component of the same signal" literal
rather than approximate, and keeps the T-mean readout interpretable.

T drops from num_patches (24) to J (~4), and T multiplies the spike-op count directly.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

_SMOOTH = {
    'b3': [1 / 16., 4 / 16., 6 / 16., 4 / 16., 1 / 16.],   # B3-spline, classic a-trous kernel
    'haar': [0.5, 0.5],
}


class SWTEncoder(nn.Module):
    """x [B, L, C] -> details [J, B, L, C] (fine -> coarse) + approximation [B, L, C]."""

    def __init__(self, levels=4, name='b3', mode='wavelet', seed=0):
        super().__init__()
        self.levels = int(levels)
        self.mode = mode                        # 'wavelet' | 'rand' (control)
        if mode == 'rand':
            # CONTROL: same band count, same conv cost, no multi-scale structure. Isolates
            # "wavelet decomposition" from "J extra input channels of some filtered signal".
            g = torch.Generator().manual_seed(seed)
            k = torch.randn(5, generator=g)
            k = k / k.abs().sum()               # normalised like a smoothing kernel
            taps = k.tolist()
        else:
            taps = _SMOOTH[name]
        self.register_buffer('h', torch.tensor(taps, dtype=torch.float32).view(1, 1, -1))

    def _smooth(self, x, dil):
        """Circular padding keeps length L exactly; dilation = a-trous (no decimation)."""
        pad = (self.h.shape[-1] - 1) * dil
        lo = pad // 2
        x = F.pad(x, (lo, pad - lo), mode='circular')
        return F.conv1d(x, self.h, dilation=dil)

    def forward(self, x):
        B, L, C = x.shape
        c = x.permute(0, 2, 1).reshape(B * C, 1, L)          # [BC, 1, L]
        details = []
        for j in range(self.levels):
            c_next = self._smooth(c, 2 ** j)
            details.append(c - c_next)                        # detail = what this scale removed
            c = c_next
        det = torch.stack(details, 0).view(self.levels, B, C, L).permute(0, 1, 3, 2)
        appr = c.view(B, C, L).permute(0, 2, 1)
        return det.contiguous(), appr.contiguous()
