"""MODI 2026-10-28: token-order shuffle for the shuffle-retrain diagnostic (--token_shuffle).

Training: every sample gets its own random permutation of the N tokens, drawn from a generator of its own (seed + 1), so the
model's other random streams (initialisation, dropout, data order and crops) stay those of the normal-order run.
Evaluation: one fixed permutation for every window, numpy default_rng(0).permutation(N) -- the b = 1 shuffle of the cause
analysis (rhythm_cause.block_perm(N, 1, 0)), so the eval-time shuffle of a normal-order model and a shuffle-trained model see the
same order. The tokens themselves (patch contents, overlap included) are unchanged; only their positions move.
"""
import numpy as np
import torch


class TokenShuffle:
    def __init__(self, n, seed):
        self.n, self.seed = int(n), int(seed)
        self.gen = None
        self.eval_perm = torch.as_tensor(np.random.default_rng(0).permutation(self.n))

    def perms(self, B, device, training):
        """[B, N] permutations (row b: token order of sample b)."""
        if not training:
            return self.eval_perm.to(device).unsqueeze(0).expand(B, -1)
        if self.gen is None or self.gen.device != torch.device(device):
            self.gen = torch.Generator(device=device); self.gen.manual_seed(self.seed + 1)
        return torch.argsort(torch.rand(B, self.n, generator=self.gen, device=device), dim=1)


def permute_tokens(x, n_dim, perm):
    """x with the batch on dim 1 and the N tokens on dim n_dim (0 or 2); out[.., b, t, ..] = x[.., b, perm[b, t], ..]."""
    if n_dim == 0:
        idx = perm.t().reshape(perm.shape[1], perm.shape[0], *([1] * (x.dim() - 2)))
    elif n_dim == 2:
        idx = perm.reshape(1, *perm.shape, *([1] * (x.dim() - 3)))
    else:
        raise ValueError(n_dim)
    return torch.gather(x, n_dim, idx.expand_as(x).contiguous())
