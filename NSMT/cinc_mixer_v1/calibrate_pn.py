"""Rate-matched BN gains of the population SpkMixer for the neuron-placement test (2026-11-18, PROJECT_LOG, exp/f-lif-pop-v3; user request 11-18).

Same procedure as calibrate_stft.py / calibrate_len.py (calibrate_spk.calibrate: every population layer in forward order, bisection on one
gain per layer so that its initial firing equals that of the same layer in the reference model), for each task x population neuron
(pop_lif / pop_flif) x scope (all = every neuron layer; mixer = only the directional stacks and the fusion neurons, user choice 11-18) x
repeat pooling (none / mean), on real training inputs of the task (stft_common.batch: 32 ETT windows = 224 rows, 32 MIT-BIH beats; at another
ETT input length check_len.batch; model seed 0, training-mode BN). The reference is the SpkMixer with the single spikeDE neuron of the same
family (lif for pop_lif, flif for pop_flif) with the same pooling and input length (scope does not apply to it). With scope 'mixer' only the
population layers are calibrated; the embedding / MLP neurons are the reference's own neuron and keep gain 1.
The table of pop_lif / all / none at the default length already exists (calib/stft/..._bdnone_txrep.json, runs were trained with it) and is
not remade. Tables: calib/pn/gains_<task>_<mixer>_muxlin_<connect>_sc<scope>_mp<pool>[_s<L>].json; never overwritten; a table whose worst
relative rate error over the layers is >= 0.10 is removed and the script stops (fail closed).
usage: python -B calibrate_pn.py [--tasks ETTh2 ETTm2 MITBIH] [--mixers pop_lif pop_flif] [--scopes all mixer] [--pools none mean] [--len 336]
"""
import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import calibrate_spk as cs                                                 # noqa: E402
import stft_common as sc                                                   # noqa: E402
from spk_mixer import apply_gain_table                                     # noqa: E402

DEFAULT_LEN = {'ETTh2': 96, 'ETTm2': 96, 'MITBIH': 187}


def table_path(task, mixer, scope, pool, L=None):
    L = L or DEFAULT_LEN[task]
    s = '' if L == DEFAULT_LEN[task] else f'_s{L}'
    return os.path.join(HERE, 'calib', 'pn', f'gains_{task}_{mixer}_muxlin_{sc.TASKS[task]["connect_f"]}_sc{scope}_mp{pool}{s}.json')


def runner_table(task, mixer, scope, pool, L=None):
    """the table a run uses: the existing STFT-test table for pop_lif / all / none at the default length, else calib/pn."""
    L = L or DEFAULT_LEN[task]
    if mixer == 'pop_lif' and scope == 'all' and pool == 'none' and L == DEFAULT_LEN[task]:
        return sc.table_path(task, 'pop_lif', 'none', 'rep')
    return table_path(task, mixer, scope, pool, L)


def over(kw, population):
    o = {}
    if kw['pool'] != 'none':
        o['mix_pool'] = kw['pool']
    if population and kw['scope'] != 'all':
        o['neuron_scope'] = kw['scope']
    if kw['L'] != DEFAULT_LEN[kw['task']]:
        o['seq_len'] = kw['L']
    return o


def build(mixer, comb, connect, kw):
    return sc.build(kw['task'], mixer, seed=0, tau_calib=False, **over(kw, mixer in cs.MIXERS)).cuda()


@torch.no_grad()
def rates(m, x):
    out, hs = {}, []
    for name, mod in m.named_modules():
        if isinstance(mod, cs.NeoNeuron):
            hs.append(mod.register_forward_hook(lambda md, i, o, k=cs.layer_kind(name): out.__setitem__(k, float(o.float().mean()))))
    cs.forward(m, x)
    for h in hs:
        h.remove()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', nargs='*', default=['ETTh2', 'ETTm2', 'MITBIH']); ap.add_argument('--mixers', nargs='*', default=['pop_lif', 'pop_flif'])
    ap.add_argument('--scopes', nargs='*', default=['all', 'mixer']); ap.add_argument('--pools', nargs='*', default=['none', 'mean'])
    ap.add_argument('--len', type=int, default=None); ap.add_argument('--n', type=int, default=32)
    a = ap.parse_args()
    os.makedirs(os.path.join(HERE, 'calib', 'pn'), exist_ok=True)
    cs.build = build
    for task in a.tasks:
        L = a.len or DEFAULT_LEN[task]
        if L != DEFAULT_LEN[task]:
            if task == 'MITBIH':
                raise SystemExit('MIT-BIH beats have one length (187)')
            from check_len import batch as len_batch
            x = len_batch(task, L, a.n).cuda()
        else:
            x = sc.batch(task, a.n).cuda()
        connect = sc.TASKS[task]['connect_f']
        for mixer in a.mixers:
            for scope in a.scopes:
                for pool in a.pools:
                    path = runner_table(task, mixer, scope, pool, L)
                    if os.path.exists(path):
                        print(f'{os.path.relpath(path, HERE)}: exists, kept', flush=True)
                        continue
                    kw = dict(task=task, scope=scope, pool=pool, L=L)
                    table, report = cs.calibrate(mixer, 'muxlin', connect, x, kw)
                    json.dump({'gains': table, 'task': task, 'mixer': mixer, 'scope': scope, 'pool': pool, 'seq_len': L, 'rows': int(x.shape[0]),
                               'reference': mixer.split('_', 1)[1], 'report': report}, open(path, 'w'), indent=1)
                    m = build(mixer, 'muxlin', connect, kw)
                    apply_gain_table(m, path)
                    got = rates(m, x)
                    target = cs.ref_rates(mixer.split('_', 1)[1], 'muxlin', connect, x, kw)
                    worst_cal = max(abs(got[cs.layer_kind(n)] - target[cs.layer_kind(n)]) / target[cs.layer_kind(n)] for n, _, _ in cs.calib_layers(m))
                    worst_all = max(abs(got[k] - target[k]) / target[k] for k in target)
                    print(f'{os.path.relpath(path, HERE)}: {len(table)} calibrated layers, worst relative rate error {worst_cal:.3f} '
                          f'(calibrated layers) / {worst_all:.3f} (all layers)', flush=True)
                    for k in target:
                        print(f'    {k}: reference {target[k]:.4f} model {got[k]:.4f}', flush=True)
                    for line in report:
                        print('   ', line, flush=True)
                    if not worst_all < 0.10:
                        os.remove(path)
                        raise SystemExit(f'{path}: rates not matched (worst {worst_all:.3f}); table removed')
                    del m; torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
