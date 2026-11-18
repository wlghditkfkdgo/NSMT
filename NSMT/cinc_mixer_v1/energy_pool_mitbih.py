"""MIT-BIH energy under the pooled-integer rule for one suite (2026-11-19, exp/f-lif-pop-v3; PROJECT_LOG mixer-only population pre-registration).
The rule and checks are those of results/check-pool-energy-20261119/energy_recount_pool.py (00:08 entry; a proposal awaiting the user's approval,
reported next to the existing energy, never in its place):
  a Linear input x >= 0 that is integer-valued -> sum(x) ACs (the integer rule of energy_recount_mitbih.py); otherwise, if T_h x is
  integer-valued -> sum(T_h x) ACs (the 1 / T_h of the repeat mean is absorbed by the BatchNorm after the Linear); otherwise MAC.
  E_pool = total_fft_uJ - syn_energy_uJ + (pooled-rule synaptic energy)  (logged RESULT values; the integer-rule energy is total_fft_uJ_corr)
Every final run (val_fold -1) of the suite is reloaded and the test file re-run (energy_recount_mitbih.measure).
Checks (fail closed, exit 1, nothing written): the recomputed macro-F1 equals the logged one (1e-6); the recorder's integer rule equals the
suite's energy_recount.csv syn_int_uJ (relative 1e-6); the pooled rule acts only in mean-pooling runs (tag contains _mpmean) and there only on
Linear layers inside the directional stacks ('mixer' in the module name); in runs without pooling E_pool = total_fft_uJ_corr (relative 1e-6).
usage (from cinc_mixer_v1/): python -B energy_pool_mitbih.py --suite <suite>  ->  results/<suite>/energy_pool.csv
"""
import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch                                                                       # noqa: E402
import energy_recount_mitbih as erm                                                # noqa: E402
from energy import SynOpRecorder, E_AC, E_MAC                                      # noqa: E402
from summarize_rhythm import parse                                                 # noqa: E402

T_H = 4
LAST = []
BAD = []


class PoolRec(SynOpRecorder):
    def __init__(self, model):
        super().__init__(model)
        self.names = {id(m): n for n, m in model.named_modules()}
        self.pl = dict(ac=0., mac=0.)
        self.pooled = set()
        LAST.append(self)

    def _lin_hook(self, module, inp, out):
        super()._lin_hook(module, inp, out)
        x = inp[0].detach()
        rows, fan_out = x.numel() / x.shape[-1], out.shape[-1]
        b = rows * fan_out if module.bias is not None else 0
        if bool(x.min() >= 0) and bool(torch.equal(x, torch.round(x))):
            self.pl['ac'] += float(x.sum()) * fan_out + b
        elif bool(x.min() >= 0) and bool(torch.equal(x * T_H, torch.round(x * T_H))):
            self.pl['ac'] += float((x * T_H).sum()) * fan_out + b
            self.pooled.add(self.names[id(module)])
        else:
            self.pl['mac'] += x.numel() * fan_out + b

    def pool_uJ(self):
        n = max(1, self.n_samples)
        ac, mac = (self.pl['ac'] + self.bn['ac'] + self.neuron_ac) / n, (self.pl['mac'] + self.bn['mac']) / n
        return (E_AC * ac + E_MAC * mac) * 1e6


erm.SynOpRecorder = PoolRec


def fail(m):
    BAD.append(m); print('FAIL ' + m, flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--suite', required=True); ap.add_argument('--device', default='cuda'); a = ap.parse_args()
    S = os.path.join(HERE, 'results', a.suite)
    csvr = {}
    for e in csv.DictReader(open(os.path.join(S, 'energy_recount.csv'))):
        if str(e.get('val_fold', '-1')) == '-1':
            csvr[(e['tag'], str(int(e['seed'])))] = e
    out = []
    for r in parse(os.path.join(S, 'raw.txt')):
        if int(r.get('val_fold', -1)) != -1:
            continue
        tag, seed = r['tag'], str(int(r['seed']))
        e = csvr.get((tag, seed))
        if e is None:
            fail(f'{tag} seed {seed}: no energy_recount.csv row'); continue
        f1, s, _ = erm.measure(r['dir'], a.device); rec = LAST[-1]
        pool = rec.pool_uJ()
        mean_run = '_mpmean' in tag
        if abs(f1 - r['f1']) > 1e-6:
            fail(f"{r['dir']}: f1 {f1} != {r['f1']}")
        if abs(s['syn_energy_int_uJ'] - float(e['syn_int_uJ'])) > 1e-6 * float(e['syn_int_uJ']):
            fail(f"{r['dir']}: integer rule {s['syn_energy_int_uJ']} != csv {e['syn_int_uJ']}")
        if bool(rec.pooled) != mean_run or (mean_run and not all('mixer' in n for n in rec.pooled)):
            fail(f"{r['dir']}: pooled-rule layers {sorted(rec.pooled)} (mean run {mean_run})")
        old = float(e['total_fft_uJ_corr'])
        new = r['total_fft_uJ'] - r['syn_energy_uJ'] + pool
        if not mean_run and abs(new - old) > 1e-6 * old:
            fail(f"{r['dir']}: no pooling but E_pool {new} != {old}")
        out.append(dict(suite=a.suite, tag=tag, seed=int(seed), f1=f"{r['f1']:.6f}", E_old=f'{old:.6f}', E_pool=f'{new:.6f}',
                        pooled_layers=';'.join(sorted(rec.pooled))))
        print(f"{tag} s{seed}: f1 {f1:.6f} E {old:.3f} -> {new:.3f} uJ  pooled {sorted(rec.pooled)}", flush=True)
    if not out:
        fail('no final runs')
    if BAD:
        print(f'{len(BAD)} FAILED'); sys.exit(1)
    with open(os.path.join(S, 'energy_pool.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print(f'{len(out)} runs -> {S}/energy_pool.csv; all checks passed')


if __name__ == '__main__':
    main()
