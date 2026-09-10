"""Per-layer firing-rate visualization helper.

Defends the spiking-temporal-alignment claim (T = N) by showing per-layer
firing rates across all ``MultiStepLIFNode`` modules. Output spikes are
binary {0, 1} so ``output.mean()`` is the firing rate of that layer.

Kept as a standalone module so ``test.py`` only needs a single import + call.
"""

import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from spikingjelly.clock_driven import functional
from spikingjelly.clock_driven.neuron import MultiStepLIFNode


@torch.no_grad()
def plot_firing_rate_per_layer(model, loader, device, save_path,
                               max_batches=8, dpi=200):
    """Run a few batches with hooks attached to every spiking neuron and
    save a horizontal bar chart of mean firing rate per layer.

    Args
    ----
    model : nn.Module
        NSMT model in eval mode (caller decides; we keep ``model.train_mode``).
    loader : DataLoader
        Returns batches whose first element is the input tensor ``x``.
    device : torch.device
    save_path : str
        Path prefix; ``.png`` and ``.pdf`` will be appended.
    max_batches : int, default 8
    """
    model.eval()
    layer_names = []      # list[str], registration order
    fr_accum = {}         # name -> running sum of mean spike rate
    count_accum = {}      # name -> number of forward passes seen

    def make_hook(name):
        def hook(_module, _input, output):
            # output: spike tensor in {0, 1}, any shape; mean is firing rate.
            fr_accum[name] = fr_accum.get(name, 0.0) + output.detach().float().mean().item()
            count_accum[name] = count_accum.get(name, 0) + 1
        return hook

    handles = []
    for name, mod in model.named_modules():
        if isinstance(mod, MultiStepLIFNode):
            layer_names.append(name)
            handles.append(mod.register_forward_hook(make_hook(name)))

    if not layer_names:
        print("[firing-rate] no MultiStepLIFNode found - skipping plot.")
        return

    try:
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            x = batch[0].float().to(device)  # [B, L, C]
            functional.reset_net(model)
            _ = model(x)
    finally:
        for h in handles:
            h.remove()

    # Some hooks may never fire (modules registered but skipped in the
    # current ``train_mode`` branch); filter them out.
    used = [n for n in layer_names if count_accum.get(n, 0) > 0]
    if not used:
        print("[firing-rate] no LIF modules fired during the probe - skipping plot.")
        return

    fr_per_layer = [fr_accum[n] / count_accum[n] for n in used]

    short_names = []
    for n in used:
        s = (n.replace('time_block.', 'neo.')
              .replace('encoding_neo.', 'enc_neo.')
              .replace('encoding.', 'enc.')
              .replace('data_block.', 'hippo.'))
        short_names.append(s if len(s) <= 40 else '...' + s[-39:])

    fig_h = max(4.0, 0.18 * len(used))
    fig, ax = plt.subplots(figsize=(8, fig_h))
    y_pos = np.arange(len(used))
    ax.barh(y_pos, fr_per_layer, color="#3b7dd8")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(short_names, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("Firing rate")
    avg_fr = float(np.mean(fr_per_layer))
    ax.axvline(avg_fr, linestyle="--", color="gray", linewidth=0.8,
               label=f"mean = {avg_fr:.3f}")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("Per-layer firing rate (LIF outputs)", fontsize=10)
    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(save_path + ".pdf", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[firing-rate] figure saved to `{save_path}.png` "
          f"(layers={len(used)}, mean fr={avg_fr:.4f})")
