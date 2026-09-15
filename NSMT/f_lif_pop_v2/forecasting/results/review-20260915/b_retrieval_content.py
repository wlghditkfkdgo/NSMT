"""Read-only: is the learned read distribution content-conditioned, or a fixed (t, j) lag kernel?

Loads completed v2 patch checkpoints on CPU and evaluates 96 test windows spread across the
whole test span (the saved diagnostics used only the first 8 consecutive windows)."""
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch

NSMT = '/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT'
TASK = NSMT + '/f_lif_pop_v2/forecasting'
sys.path.insert(0, NSMT)
sys.path.insert(0, TASK)
torch.set_num_threads(8)
from f_lif_pop_v2.forecasting.ours import myModel  # noqa: E402
from data_provider.data_loader import Dataset_ETT_hour  # noqa: E402

KEYS = ['seq_len', 'pred_len', 'patch_size', 'embed_dim', 'num_population', 'head_dim', 'head_mode',
        'heterogeneous', 'retrieval', 'tau_min', 'tau_max', 'threshold', 'memory_strength', 'temperature',
        'input_scale', 'architecture', 'read_mode', 'null_logit_init', 'num_heads']


class Args:
    pass


def analyse(aux):
    W, mass = aux['attention'], aux['real_mass'][..., 0]
    T = W.shape[0]
    tot = res_global = res_neuron = 0.
    rows = lag1 = lag1_uniform = elag = elag_uniform = top_recent = top_far = support = 0.
    empty = cells = 0.
    for t in range(2, T):
        w = W[t, :, :, :t].double()
        keep = mass[t] > 1e-6
        empty += (~keep).sum().item()
        cells += keep.numel()
        w = torch.where(keep[..., None], w / w.sum(-1, keepdim=True).clamp_min(1e-12), torch.zeros_like(w))
        m = keep[..., None].double()
        n = keep.sum().item()
        if n == 0:
            continue
        kernel = (w * m).sum((0, 1)) / n
        kernel_neuron = (w * m).sum(0) / keep.sum(0).clamp_min(1)[:, None]
        tot += (((w - 1. / t) ** 2) * m).sum().item()
        res_global += (((w - kernel) ** 2) * m).sum().item()
        res_neuron += (((w - kernel_neuron) ** 2) * m).sum().item()
        lag = torch.arange(t, 0, -1, dtype=torch.double)  # slot j has lag t-j
        rows += n
        lag1 += (w[..., -1] * keep).sum().item()
        lag1_uniform += n / t
        elag += ((w * lag).sum(-1) * keep).sum().item()
        elag_uniform += n * (t + 1) / 2
        top = w.argmax(-1)
        top_recent += ((top == t - 1) & keep).sum().item()
        top_far += (((t - top) > t / 2) & keep).sum().item()
        support += ((w > 0).sum(-1) * keep).sum().item()
    ev = aux['evidence'][1:].abs().mean() / aux['charge'][1:].abs().mean().clamp_min(1e-8)
    return dict(content_share_vs_global_kernel=res_global / tot, content_share_vs_neuron_kernel=res_neuron / tot,
                lag1_mass=lag1 / rows, lag1_mass_if_uniform=lag1_uniform / rows,
                mean_lag=elag / rows, mean_lag_if_uniform=elag_uniform / rows,
                top1_is_lag1=top_recent / rows, top1_in_far_half=top_far / rows,
                support_size=support / rows, empty_read_frac=empty / cells,
                gate=aux['gate'][1:].mean().item(), evidence_to_charge=ev.item())


results = []
for H in [96, 720]:
    per_run = pd.read_csv(f'{TASK}/results/selective-v2-20260914_patch_p{H}/per_run.csv')
    datasets = {}
    for _, r in per_run[per_run.policy != 'off'].iterrows():
        cfg = torch.load(os.path.join(r.log_path, 'model_state/config.pt'), map_location='cpu')
        model = myModel(**{k: cfg[k] for k in KEYS})
        model.load_state_dict(torch.load(os.path.join(r.log_path, 'model_state/best+model.pt'), map_location='cpu'))
        model.eval()
        if r.data not in datasets:
            a = Args()
            a.seq_len, a.pred_len, a.root_path, a.data_path = 336, H, cfg['root_path'], cfg['data_path']
            ds = Dataset_ETT_hour(a, 'test')
            idx = np.linspace(0, len(ds) - 1, 96).round().astype(int)
            datasets[r.data] = (torch.stack([ds[i][0] for i in idx]), torch.stack([ds[i][0] for i in range(8)]))
        spread, first8 = datasets[r.data]
        with torch.no_grad():
            _, aux = model(spread, return_aux=True)
            _, aux8 = model(first8, return_aux=True)
        row = dict(H=H, data=r.data, seed=r.seed, variant=r.variant, **analyse(aux))
        row['empty_read_frac_first8'] = analyse(aux8)['empty_read_frac']
        results.append(row)
        print(H, r.run_id, 'done', flush=True)

out = pd.DataFrame(results)
out.to_csv(os.path.join(os.path.dirname(__file__), 'b_retrieval_content.csv'), index=False)
cols = ['content_share_vs_global_kernel', 'content_share_vs_neuron_kernel', 'lag1_mass', 'lag1_mass_if_uniform',
        'mean_lag', 'mean_lag_if_uniform', 'top1_is_lag1', 'top1_in_far_half', 'support_size',
        'empty_read_frac', 'empty_read_frac_first8', 'gate', 'evidence_to_charge']
pd.set_option('display.width', 250)
print(out.groupby(['H', 'variant'])[cols].mean().T.to_string(float_format=lambda v: f'{v:.3f}'))
print(out.groupby(['H', 'variant', 'data'])[['content_share_vs_neuron_kernel', 'lag1_mass', 'empty_read_frac', 'empty_read_frac_first8']]
      .mean().to_string(float_format=lambda v: f'{v:.3f}'))
