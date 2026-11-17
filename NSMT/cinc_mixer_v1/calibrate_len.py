"""Rate-matched BN gains of the population LIF SpkMixer at a longer ETT input (2026-11-17, PROJECT_LOG input-length test, exp/f-lif-pop-v3).

check_len.py d3: with the table made at input 96 the population mixer layers fire 55-63 % more than the LIF model's at input 336
(the slow members keep charging over more tokens), so by the pre-fixed rule (worst relative error >= 0.10) a table is made at the new
length. Same procedure as calibrate_stft.py (calibrate_spk.calibrate: every population layer in forward order, bisection on one gain
per layer so that its initial firing equals that of the same layer in the LIF SpkMixer of the same dataset AND input length), on 32
real training windows of that length (check_len.batch; model seed 0, training-mode BN).
Tables: calib/len/gains_<ds>_pop_lif_muxlin_IAND_s<L>.json; never overwritten.
usage: python -B calibrate_len.py --len 336
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
from check_len import batch, rates                                         # noqa: E402
from spk_mixer import apply_gain_table                                     # noqa: E402


def table_path(ds, L):
    return os.path.join(HERE, 'calib', 'len', f'gains_{ds}_pop_lif_muxlin_{sc.TASKS[ds]["connect_f"]}_s{L}.json')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--len', type=int, required=True); ap.add_argument('--n', type=int, default=32)
    a = ap.parse_args()
    os.makedirs(os.path.join(HERE, 'calib', 'len'), exist_ok=True)
    cs.build = lambda mixer, comb, connect, kw: sc.build(kw['task'], mixer, seed=0, tau_calib=False, seq_len=kw['L']).cuda()
    for ds in ('ETTh2', 'ETTm2'):
        path = table_path(ds, a.len)
        if os.path.exists(path):
            print(f'{os.path.basename(path)}: exists, kept', flush=True)
            continue
        x = batch(ds, a.len, a.n).cuda()
        kw = dict(task=ds, L=a.len)
        table, report = cs.calibrate('pop_lif', 'muxlin', sc.TASKS[ds]['connect_f'], x, kw)
        json.dump({'gains': table, 'task': ds, 'mixer': 'pop_lif', 'seq_len': a.len, 'rows': int(x.shape[0]), 'report': report}, open(path, 'w'), indent=1)
        m = sc.build(ds, 'pop_lif', seed=0, gain_table=path, seq_len=a.len).cuda()
        apply_gain_table(m, path)
        got = rates(m, x)
        target = cs.ref_rates('lif', 'muxlin', sc.TASKS[ds]['connect_f'], x, kw)
        worst = max(abs(got[k] - target[k]) / target[k] for k in target)
        print(f'{os.path.basename(path)}: {len(table)} layers, worst relative rate error {worst:.3f}', flush=True)
        for line in report:
            print('   ', line, flush=True)
        old = json.load(open(sc.table_path(ds, 'pop_lif', 'none', 'rep')))['gains']
        for k in table:
            print(f'    {k}: input 96 ' + '/'.join(f'{g:.3f}' for g in old[k]) + f' -> input {a.len} ' + '/'.join(f'{g:.3f}' for g in table[k]), flush=True)
        if not worst < 0.10:
            os.remove(path)
            raise SystemExit(f'{path}: rates not matched (worst {worst:.3f}); table removed')
        del m; torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
