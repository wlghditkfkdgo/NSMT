"""EMA neocortex memory bank (spike-preserving IAND-style mixing).

Self-contained helper that maintains a cross-window trend prior over the
neocortical embedding ``x_neo``. Kept in a separate module so that the main
:class:`myModel` body in ``ours.py`` only needs additive hooks (one buffer
register and one forward call).

Mixing rule
-----------
``x_neo`` is the binary spike output of a LIF neuron. The bank stores a
running EMA of the *firing rate* per (T, D) location. Naively adding the
continuous bank back into ``x_neo`` mixes continuous noise into the binary
spike stream and breaks the downstream sparse AND-style gating
(see ``Block.forward`` in ``ours.py``).

To keep spike semantics, we

  1. **Binarize the bank** by thresholding its firing-rate buffer
     (positions whose firing rate exceeds ``thresh`` are treated as
     "frequently-firing memory locations").
  2. **Mix only into the OFF positions** of ``x_neo`` (IAND-style),
     preserving every existing spike. Formally, we apply

         x_neo' = x_neo + alpha * (1 - x_neo) * bank_spike

     which means

         - bank_spike = 0  -> x_neo' = x_neo                (no change)
         - x_neo = 1       -> x_neo' = 1                    (existing spike kept)
         - x_neo = 0, bank_spike = 1 -> x_neo' = alpha      (soft nudge)

     The output is bounded in [0, 1] for ``alpha <= 1`` and the
     ``x_neo == 0`` sparsity pattern is preserved everywhere bank_spike
     is also 0. This matches the IAND residual style already used in the
     NSMT data path.

Notation matches ``ours.py`` :
    T : spiking time-step axis, equal to ``num_patches`` (token axis aligned)
    B : batch * channel axis flattened (``B*M`` in the model)
    N : within-token axis after pooling along the patch dim (= 1)
    D : embedding dim
"""

import torch
import torch.nn as nn


class NeoMemoryBank(nn.Module):
    """EMA-accumulated, spike-preserving trend prior over ``x_neo``.

    Args
    ----
    T : int
        Spiking time-step axis (equals ``num_patches`` in NSMT).
    embed_dim : int
        Embedding dim ``D``.
    decay : float, default 0.99
        EMA decay. ``bank <- decay * bank + (1 - decay) * pooled``.
    alpha : float, default 0.1
        Strength of the IAND-style nudge applied to OFF positions of ``x_neo``.
    thresh : float, default 0.10
        Firing-rate threshold above which a bank location is considered a
        "memory-active" position. Defaults to 10%, slightly above typical
        per-layer firing rates observed on NSMT.

    Buffer shape
    ------------
    ``bank`` : ``[T, 1, 1, D]`` so it broadcasts cleanly against
    ``x_neo`` of shape ``[T, B*M, 1, D]``.
    """

    def __init__(self, T: int, embed_dim: int,
                 decay: float = 0.99, alpha: float = 0.1,
                 thresh: float = 0.10):
        super().__init__()
        self.T = int(T)
        self.embed_dim = int(embed_dim)
        self.decay = float(decay)
        self.alpha = float(alpha)
        self.thresh = float(thresh)

        # Pre-allocate so .to(device) and state_dict() Just Work.
        self.register_buffer("bank", torch.zeros(self.T, 1, 1, self.embed_dim))
        self.register_buffer("initialized", torch.zeros(1, dtype=torch.bool))

    @torch.no_grad()
    def update(self, x_neo: torch.Tensor):
        """Update the bank with the current batch's pooled ``x_neo``.

        x_neo : [T, B*M, 1, D] (binary spikes from the LIF output)
        """
        T, BM, N, D = x_neo.shape
        assert T == self.T and N == 1 and D == self.embed_dim, (
            f"unexpected x_neo shape {tuple(x_neo.shape)}, "
            f"expected (T={self.T}, *, 1, D={self.embed_dim})"
        )
        pooled = x_neo.detach().mean(dim=1, keepdim=True)  # [T, 1, 1, D]
        if not bool(self.initialized.item()):
            self.bank.copy_(pooled)
            self.initialized.fill_(True)
        else:
            self.bank.mul_(self.decay).add_(pooled, alpha=1.0 - self.decay)

    def forward(self, x_neo: torch.Tensor) -> torch.Tensor:
        """Inject the bank prior into ``x_neo`` via IAND-style mixing.

        x_neo : [T, B*M, 1, D] -> returns same shape, values in [0, 1].
        """
        T, BM, N, D = x_neo.shape
        assert T == self.T and N == 1 and D == self.embed_dim, (
            f"unexpected x_neo shape {tuple(x_neo.shape)}, "
            f"expected (T={self.T}, *, 1, D={self.embed_dim})"
        )
        if not bool(self.initialized.item()):
            return x_neo

        # bank : [T, 1, 1, D] of firing rates in [0, 1]
        bank_spike = (self.bank > self.thresh).to(x_neo.dtype)  # {0., 1.}

        # IAND-style nudge: only OFF positions of x_neo are touched, by alpha.
        # Result is in [0, 1] for alpha in [0, 1].
        delta = self.alpha * (1.0 - x_neo) * bank_spike  # broadcasts [T,1,1,D]
        x_out = x_neo + delta
        # Defensive clamp; mathematically already in [0, 1] for alpha in [0, 1].
        return x_out.clamp(0.0, 1.0)
