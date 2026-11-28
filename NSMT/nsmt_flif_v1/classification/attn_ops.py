"""MODI 2026-11-28 (approved plan 1-snuggly-galaxy.md, step 1 D; REPORT ONLY): the attention additions that syops does not count.

syops counts nn.Linear / BatchNorm only, so the attention products q k^T and A v are missing for every model (Spikformer(DE) included).
With --neo_readout memd the queries are real-valued (no q_linear), so syops alone would favour it. This script loads every run's tested
weights (ETT: checkpoint.pth = best validation; MIT-BIH: best+model.pt = last epoch, as train.py tests it; Sleep: checkpoint.pth = best
validation, the reported 'best' test), switches on layers.SSA_rel_scl._op_counter (q k^T: q != 0 and k = 1; A v: a != 0 and v = 1 -- one
addition each; k and v are 0 / 1 in every model here, checked), runs the first --max_samples test samples in eval mode and reports
additions per sample, their energy at 0.9 pJ (E_AC), and syops + attention energy and its ratio to Spikformer(DE). Judged energy stays
syops (user rule); this is an extra column. Rows = the runs summarize_rd.py uses (same fail-closed loaders).
usage (classification folder): python attn_ops.py --task mitbih|sleep|ett [--max_samples 2048]
"""
import argparse
import json
import os
import sys
from types import SimpleNamespace as NS

import numpy as np
import torch
from spikingjelly.clock_driven import functional

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import summarize_rd as SR
from layers import SSA_rel_scl
from load_model import LOAD_MODEL

E_AC_UJ = 0.9e-6                                                     # 0.9 pJ in µJ


def attach(model):
    ds = []
    for m in model.modules():
        if isinstance(m, SSA_rel_scl):
            m._op_counter = {}; ds.append(m._op_counter)
    if not ds:
        raise SystemExit('STOP: no SSA_rel_scl in the model')
    return ds


def build(task, d):
    cfg = torch.load(os.path.join(d, 'model_state', 'config.pt'), map_location='cpu')
    a = NS(**dict(cfg, device='cuda'))
    if task == 'ett':
        from forecast_wrap import Forecaster
        m = Forecaster(LOAD_MODEL[a.model](a, True), a.embed_dim, a.pred_len, bias=a.bias, head_kind=getattr(a, 'fc_head', 'flat'),
                       head_dim=getattr(a, 'fc_head_dim', 8), seq_len=a.seq_len).cuda()
        w = 'checkpoint.pth'
    else:
        m = LOAD_MODEL[a.model](a, True).cuda()
        w = 'best+model.pt' if task == 'mitbih' else 'checkpoint.pth'
    m.load_state_dict(torch.load(os.path.join(d, 'model_state', w), map_location='cuda'))
    m.eval(); m.train_mode = 'testing'
    return a, m


def loader(task, a):
    if task == 'ett':
        sys.path.append(os.path.join(HERE, '..', 'forecasting'))
        from data_provider.data_factory import data_provider
        from train_forecast import data_args
        return data_provider(data_args(a, a), 'test')[1]
    if task == 'mitbih':
        from dataloader import create_loader
        return create_loader(train=False, batch_size=512, num_classes=a.num_classes, data_root=a.root_path + '/5class_mitbih_test_3_187.csv',
                             data_window_size=187, num_workers=2)[0]
    import rhythm_data as rd
    return torch.utils.data.DataLoader(rd.build(a, a.seed)['test'], batch_size=256, shuffle=False)


def count(task, d, max_samples):
    a, m = build(task, d)
    ds = attach(m)
    n = 0
    with torch.no_grad():
        for batch in loader(task, a):
            x = batch[0].float().cuda()
            functional.reset_net(m); m(x); functional.reset_net(m)
            n += x.shape[0]
            if n >= max_samples:
                break
    if not all(c.get('kv_binary', False) for c in ds) or any(c.get('kv_cat_uncounted') for c in ds):
        raise SystemExit(f'STOP: {d}: keys / values not 0 / 1, or an uncounted kv_cat term')
    adds = sum(c['qk'] + c['av'] for c in ds)
    return dict(adds_per_sample=adds / n, qk_per_sample=sum(c['qk'] for c in ds) / n, av_per_sample=sum(c['av'] for c in ds) / n, n=n)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--task', required=True, choices=['mitbih', 'sleep', 'ett'])
    ap.add_argument('--max_samples', type=int, default=2048); ap.add_argument('--root', default=os.path.dirname(HERE))
    a = ap.parse_args()
    t = {'mitbih': SR.load_mitbih, 'sleep': SR.load_sleep, 'ett': SR.load_ett}[a.task](a.root)
    rows = {}
    for m in ['SF'] + SR.MODELS:
        c = 'spkall_lif' if m == 'SF' else t.cond_of(m)
        per = []
        for p in t.pairs:
            r = t.T[(c,) + p]
            o = count(a.task, r['dir'], a.max_samples)
            o.update(syn=float(r['syn_energy_uJ']), attn_uJ=o['adds_per_sample'] * E_AC_UJ)
            per.append(o)
        rows['Spikformer(DE)' if m == 'SF' else SR.name(m)] = per
    sf = rows['Spikformer(DE)']
    sf_syn, sf_tot = np.mean([o['syn'] for o in sf]), np.mean([o['syn'] + o['attn_uJ'] for o in sf])
    out = [f'# {a.task}: attention additions (report only; judged energy = syops)\n',
           f'First {a.max_samples} test samples per run; E_AC 0.9 pJ. Ratios vs Spikformer(DE).\n',
           '| model | q·k adds / sample | A·v adds / sample | attention µJ | syops µJ | syops + attention µJ | × (syops) | × (syops + attention) |',
           '|---|---|---|---|---|---|---|---|']
    dec = {}
    for nm, per in rows.items():
        qk, av = np.mean([o['qk_per_sample'] for o in per]), np.mean([o['av_per_sample'] for o in per])
        at, sy = np.mean([o['attn_uJ'] for o in per]), np.mean([o['syn'] for o in per])
        dec[nm] = dict(qk=float(qk), av=float(av), attn_uJ=float(at), syn_uJ=float(sy), x_syn=float(sy / sf_syn), x_total=float((sy + at) / sf_tot))
        out.append(f'| {nm} | {qk:,.0f} | {av:,.0f} | {at:.3f} | {sy:.2f} | {sy + at:.2f} | {sy / sf_syn:.2f} | {(sy + at) / sf_tot:.2f} |')
    txt = '\n'.join(out) + '\n'
    print(txt)
    od = os.path.join(a.root, 'classification', 'results', 'flif-v1-rd-summary'); os.makedirs(od, exist_ok=True)
    open(os.path.join(od, f'attn_ops_{a.task}.md'), 'w').write(txt)
    json.dump(dec, open(os.path.join(od, f'attn_ops_{a.task}.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
