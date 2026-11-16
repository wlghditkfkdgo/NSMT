"""Rate-matched BN gains of the population SpkMixer for the STFT band test (2026-11-16, PROJECT_LOG, exp/f-lif-pop-v3).

The population SpkMixer has never been trained on ETT / MIT-BIH and the existing tables (calibrate_spk.py) are for CinC, 2 mixer
layers and the repeat layout. This script runs the SAME procedure (calibrate_spk.calibrate: every population layer in forward order,
bisection on one gain per layer so that its initial firing equals that of the same layer in the LIF SpkMixer of the same task and
spiking-axis layout) for each task x band (none | member) x taxis (rep | token), on real training inputs of the task
(stft_common.batch: 32 ETT windows = 224 rows, 32 MIT-BIH beats; model seed 0, training-mode BN). The reference is always the
band-less LIF model. Tables: calib/stft/gains_<task>_<mixer>_muxlin_<connect>_bd<band>_tx<taxis>.json; never overwritten.
usage: python calibrate_stft.py [--tasks ETTh2 ETTm2 MITBIH] [--mixers pop_lif]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', nargs='*', default=list(sc.TASKS)); ap.add_argument('--mixers', nargs='*', default=['pop_lif'])
    ap.add_argument('--n', type=int, default=32)
    a = ap.parse_args()
    os.makedirs(os.path.join(HERE, 'calib', 'stft'), exist_ok=True)

    def build(mixer, comb, connect, kw):                                    # the reference (lif / flif) is always band-less
        band = kw['band'] if mixer in cs.MIXERS else 'none'
        return sc.build(kw['task'], mixer, band=band, taxis=kw['taxis'], seed=0, tau_calib=False).cuda()
    cs.build = build
    for task in a.tasks:
        x = sc.batch(task, a.n).cuda()
        for mixer in a.mixers:
            for taxis in ('rep', 'token'):
                for band in ('none', 'member'):
                    path = sc.table_path(task, mixer, band, taxis)
                    if os.path.exists(path):
                        print(f'{os.path.basename(path)}: exists, kept', flush=True)
                        continue
                    connect = sc.TASKS[task]['connect_f']
                    table, report = cs.calibrate(mixer, 'muxlin', connect, x, dict(task=task, band=band, taxis=taxis))
                    m = sc.build(task, mixer, band=band, taxis=taxis, seed=0, tau_calib=False).cuda()   # measured rates with the table
                    from spk_mixer import apply_gain_table
                    json.dump({'gains': table, 'task': task, 'mixer': mixer, 'band': band, 'taxis': taxis, 'rows': int(x.shape[0]),
                               'report': report}, open(path, 'w'), indent=1)
                    apply_gain_table(m, path)
                    rates, hs = {}, []
                    for name, mod in m.named_modules():
                        if isinstance(mod, cs.NeoNeuron):
                            hs.append(mod.register_forward_hook(lambda md, i, o, k=cs.layer_kind(name): rates.__setitem__(k, float(o.float().mean()))))
                    cs.forward(m, x)
                    for h in hs: h.remove()
                    target = cs.ref_rates(mixer.split('_', 1)[1], 'muxlin', connect, x, dict(task=task, band=band, taxis=taxis))
                    worst = max(abs(rates[k] - target[k]) / target[k] for k in target)
                    print(f'{os.path.basename(path)}: {len(table)} layers, worst relative rate error {worst:.3f}', flush=True)
                    for line in report:
                        print('   ', line, flush=True)
                    if not worst < 0.10:
                        os.remove(path)
                        raise SystemExit(f'{path}: rates not matched (worst {worst:.3f}); table removed')
                    del m; torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
