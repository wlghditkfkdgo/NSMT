"""Read-only: on the trained model's own stored states, how different is the read distribution
produced by the learned Q/K/null from the one produced by the initial Q=K=I, null=-1 scorer?"""
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
from f_lif_pop_v2.forecasting.layers import sparsemax  # noqa: E402
from data_provider.data_loader import Dataset_ETT_hour  # noqa: E402

KEYS = ['seq_len', 'pred_len', 'patch_size', 'embed_dim', 'num_population', 'head_dim', 'head_mode',
        'heterogeneous', 'retrieval', 'tau_min', 'tau_max', 'threshold', 'memory_strength', 'temperature',
        'input_scale', 'architecture', 'read_mode', 'null_logit_init', 'num_heads']


class Args:
    pass


rows = []
for H in [96, 720]:
    per_run = pd.read_csv(f'{TASK}/results/selective-v2-20260914_patch_p{H}/per_run.csv')
    windows = {}
    for _, r in per_run[per_run.policy != 'off'].iterrows():
        cfg = torch.load(os.path.join(r.log_path, 'model_state/config.pt'), map_location='cpu')
        model = myModel(**{k: cfg[k] for k in KEYS})
        model.load_state_dict(torch.load(os.path.join(r.log_path, 'model_state/best+model.pt'), map_location='cpu'))
        model.eval()
        if r.data not in windows:
            a = Args()
            a.seq_len, a.pred_len, a.root_path, a.data_path = 336, H, cfg['root_path'], cfg['data_path']
            ds = Dataset_ETT_hour(a, 'test')
            idx = np.linspace(0, len(ds) - 1, 96).round().astype(int)
            windows[r.data] = torch.stack([ds[i][0] for i in idx])
        lif = model.embedding.lif
        read = sparsemax if lif.read_mode == 'sparse' else (lambda z: z.softmax(-1))
        acc = dict(tv_learned_vs_init=0., tv_learned_vs_uniform=0., top1_agree=0., support_jaccard=0., empty_agree=0.)
        cells = nonempty = 0
        with torch.no_grad():
            _, aux = model(windows[r.data], return_aux=True)
            states, charge = aux['membrane'], aux['charge']
            for t in range(1, states.shape[0]):
                history = states[:t].permute(1, 2, 0, 3)
                c = charge[t]
                learned = -(lif.query(c).unsqueeze(-2) - lif.key(history)).square().mean(-1) / lif.temperature
                initial = -(c.unsqueeze(-2) - history).square().mean(-1) / lif.temperature
                p = read(torch.cat([learned, lif.null_logit.expand_as(learned[..., :1])], -1))
                p0 = read(torch.cat([initial, torch.full_like(initial[..., :1], -1.)], -1))
                uniform = torch.cat([torch.full_like(initial, 1. / t), torch.zeros_like(initial[..., :1])], -1)
                acc['tv_learned_vs_init'] += .5 * (p - p0).abs().sum(-1).sum().item()
                acc['tv_learned_vs_uniform'] += .5 * (p - uniform).abs().sum(-1).sum().item()
                real, real0 = p[..., :-1], p0[..., :-1]
                has, has0 = real.sum(-1) > 1e-6, real0.sum(-1) > 1e-6
                both = has & has0
                acc['top1_agree'] += ((real.argmax(-1) == real0.argmax(-1)) & both).sum().item()
                inter = ((real > 0) & (real0 > 0)).sum(-1).double()
                union = ((real > 0) | (real0 > 0)).sum(-1).double()
                acc['support_jaccard'] += torch.where(union > 0, inter / union.clamp_min(1), torch.ones_like(union)).sum().item()
                acc['empty_agree'] += (has == has0).sum().item()
                cells += has.numel()
                nonempty += both.sum().item()
        row = dict(H=H, data=r.data, seed=r.seed, variant=r.variant)
        row.update({k: v / (nonempty if k == 'top1_agree' else cells) for k, v in acc.items()})
        rows.append(row)
        print(H, r.run_id, 'done', flush=True)

out = pd.DataFrame(rows)
out.to_csv(os.path.join(os.path.dirname(__file__), 'b2_learned_vs_init_read.csv'), index=False)
pd.set_option('display.width', 250)
print(out.groupby(['H', 'variant'])[['tv_learned_vs_init', 'tv_learned_vs_uniform', 'top1_agree', 'support_jaccard', 'empty_agree']]
      .mean().to_string(float_format=lambda v: f'{v:.3f}'))
print(out.groupby(['H', 'variant', 'data'])[['tv_learned_vs_init', 'top1_agree', 'support_jaccard']].mean().to_string(float_format=lambda v: f'{v:.3f}'))
