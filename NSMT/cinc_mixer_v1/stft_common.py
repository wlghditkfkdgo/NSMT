"""Shared settings of the STFT band test (2026-11-16, PROJECT_LOG, exp/f-lif-pop-v3): the SpkMixer of each task exactly as the
runners build it (run_ett_spk.sh / run_mitbih_spk.sh with the tuned LIF values: ETT muxlin + IAND, 1 mixer layer per direction;
MIT-BIH muxlin + ADD, 2 layers), a small fixed batch of real training inputs, and the gain table path of a population model."""
import os

import numpy as np
import torch

from spk_mixer import SpkMixer

HERE = os.path.dirname(os.path.abspath(__file__))
TASKS = {
    'ETTh2': dict(seq_len=96, data_patch_size=8, embed_dim=64, mlp_ratios=0.5, mixer_layers=1, connect_f='IAND', num_classes=5),
    'ETTm2': dict(seq_len=96, data_patch_size=8, embed_dim=64, mlp_ratios=1.0, mixer_layers=1, connect_f='IAND', num_classes=5),
    'MITBIH': dict(seq_len=187, data_patch_size=32, embed_dim=384, mlp_ratios=1.0, mixer_layers=2, connect_f='ADD', num_classes=5),
}
SOMA_INIT = {'pop_lif': 0.5, 'pop_flif': 1.0}                               # as every population run since stage 9
MIT_TRAIN = '/home/yschoi/CLS_spiking_transformer/data/5class_mitbih_train_7_187.csv'
ETT_DIR = os.path.join(HERE, '..', 'forecasting', 'dataset', 'ETT-small')


def table_path(task, mixer, band, taxis):
    return os.path.join(HERE, 'calib', 'stft', f'gains_{task}_{mixer}_muxlin_{TASKS[task]["connect_f"]}_bd{band}_tx{taxis}.json')


def build(task, mixer, band='none', taxis='rep', seed=0, tau_calib=False, gain_table=None, **over):
    """SpkMixer of `task` (classification head; the ETT forecaster replaces it and does not touch the body)."""
    from config import set_random_seed
    set_random_seed(seed)
    kw = dict(TASKS[task], depths=1, mixer=mixer, T_h=4, neo_tau=2.0, neo_alpha=0.5, pop_taus=(4., 8., 16., 32.), bias=False, tau=2.0,
              in_channels=1, pop_soma_init=SOMA_INIT.get(mixer), mixer_comb='muxlin', tau_calib=tau_calib, band=band, taxis=taxis,
              gain_table=gain_table)
    kw.update(over)
    return SpkMixer(**kw)


def batch(task, n=32):
    """real training inputs as the model receives them: ETT [n * 7, 1, 96] (n windows spread over the training months, every
    channel a row, window-normalised as forecast_wrap.Forecaster); MIT-BIH [n, 1, 187] (n beats spread over the training file)."""
    if task == 'MITBIH':
        a = np.loadtxt(MIT_TRAIN, delimiter=',', dtype=np.float32, max_rows=20000)
        idx = np.linspace(0, len(a) - 1, n).astype(int)
        return torch.from_numpy(a[idx, :187]).unsqueeze(1)
    import pandas as pd
    df = pd.read_csv(os.path.join(ETT_DIR, f'{task}.csv'))
    v = df[df.columns[1:]].to_numpy(dtype=np.float32)
    n_train = 12 * 30 * 24 * (4 if task.startswith('ETTm') else 1)        # the training months of the ETT split
    starts = np.linspace(0, n_train - 96, n).astype(int)
    x = torch.from_numpy(np.stack([v[s:s + 96] for s in starts]))          # [n, 96, 7]
    x = x - x.mean(1, keepdim=True)
    x = x / torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
    return x.permute(0, 2, 1).reshape(-1, 1, 96).contiguous()
