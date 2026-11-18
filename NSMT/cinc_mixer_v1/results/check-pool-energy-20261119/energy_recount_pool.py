"""Mean-pooling energy recount (2026-11-19, exp/f-lif-pop-v3; PROJECT_LOG; user request 11-18 "mean pooling energy를 다시 측정해줘").
Report only, no training: the trained runs are reloaded and the test set re-run.

Why: with --spk_mixpool mean the two directional stacks read the mean of the T_h = 4 spike trains, values k / 4 (k = number of the 4 repeats
that spiked). syops and the integer rule of energy.SynOpRecorder count such a non-integer input to a Linear as MAC (4.6 pJ). But
W (k / 4) = (1 / 4) (k additions of the weight row), and the 1 / 4 is a constant factor that the BatchNorm right after the Linear absorbs
(inference BN is per-channel affine). So the same layer can run on additions only: sum(k) ACs. This is the integer rule applied to
T_h x input (the sum of the 4 spike trains).

Pooled-integer rule (this script, a proposal for the user): a Linear input x >= 0 that is integer-valued -> sum(x) ACs (= the integer rule);
otherwise, if T_h * x is integer-valued -> sum(T_h * x) ACs; otherwise MAC. BN and neurons as SynOpRecorder.
  ETT  : model.inner (body) as energy_recount_ett.measure; E_pool = total_fft_uJ_corr - body_syn_uJ + body_syn_pool_uJ
         (total_fft_uJ_corr and body_syn_uJ from the suite's energy_recount.csv; the head recount is unchanged)
  MIT-BIH: whole model as energy_recount_mitbih.measure; E_pool = total_fft_uJ - syn_energy_uJ + syn_pool_uJ
Checks (fail closed): recomputed test MSE / macro-F1 = logged; the recorder's syops-rule body energy = the CSV's body_syn_uJ (ETT) or the
integer-rule energy = the CSV's syn_int_uJ (MIT-BIH); the pooled rule is used on mean-pooling runs only, and there only on the first Linear
of each directional stack; for runs without pooling E_pool = the earlier corrected energy.
usage (from cinc_mixer_v1/): CUDA_VISIBLE_DEVICES=<g> python -B results/check-pool-energy-20261119/energy_recount_pool.py
"""
import csv
import os
import sys

C = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, C)
import torch                                                                       # noqa: E402
import energy_recount_ett as ere                                                   # noqa: E402
import energy_recount_mitbih as erm                                                # noqa: E402
from energy import SynOpRecorder, E_AC, E_MAC                                      # noqa: E402
from summarize_rhythm import parse                                                 # noqa: E402

T_H = 4
OUT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(C, 'results')
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


ere.SynOpRecorder = PoolRec
erm.SynOpRecorder = PoolRec


def fail(m):
    BAD.append(m); print('FAIL ' + m, flush=True)


def ett(suite, tags):
    csvr = {(r['ds'], r['pl'], r['tag'], r['seed']): r for r in csv.DictReader(open(os.path.join(RES, suite, 'energy_recount.csv')))}
    out = []
    for r in parse(os.path.join(RES, suite, 'raw.txt')):
        if r['tag'] not in tags(r['ds']):
            continue
        k = (r['ds'], str(int(r['pl'])), r['tag'], str(int(r['seed'])))
        e = csvr[k]
        q = ere.measure(r['dir'], 'cuda'); rec = LAST[-1]
        pool = rec.pool_uJ()
        mean_run = r['tag'].endswith('_mpmean')
        if abs(q['mse'] - r['mse']) > 1e-5 * r['mse']:
            fail(f"{r['dir']}: mse {q['mse']} != {r['mse']}")
        if abs(q['body_syn'] - float(e['body_syn_uJ'])) > 1e-6 * float(e['body_syn_uJ']):
            fail(f"{r['dir']}: body syops-rule {q['body_syn']} != csv {e['body_syn_uJ']}")
        if bool(rec.pooled) != mean_run or (mean_run and not all('mixer' in n for n in rec.pooled)):
            fail(f"{r['dir']}: pooled-rule layers {sorted(rec.pooled)} (mean run {mean_run})")
        old = float(e['total_fft_uJ_corr'])
        new = old - float(e['body_syn_uJ']) + pool
        if not mean_run and abs(new - old) > 1e-6 * old:
            fail(f"{r['dir']}: no pooling but E_pool {new} != {old}")
        out.append(dict(task='ETT', suite=suite, tag=r['tag'], ds=r['ds'], pl=int(r['pl']), seed=int(r['seed']), metric=f"{r['mse']:.6f}",
                        E_old=f'{old:.6f}', E_pool=f'{new:.6f}', pooled_layers=';'.join(sorted(rec.pooled))))
        print(f"ETT {r['tag']} {r['ds']} H{int(r['pl'])} s{int(r['seed'])}: mse {q['mse']:.6f} E {old:.3f} -> {new:.3f} uJ  pooled {sorted(rec.pooled)}", flush=True)
    return out


def mit(suite, tag):
    csvr = {(r['tag'], r['seed'], r['val_fold']): r for r in csv.DictReader(open(os.path.join(RES, suite, 'energy_recount.csv')))}
    out = []
    for r in parse(os.path.join(RES, suite, 'raw.txt')):
        if r.get('tag') != tag or int(r.get('val_fold', -1)) != -1:
            continue
        e = csvr[(tag, str(int(r['seed'])), '-1')]
        f1, s, cf = erm.measure(r['dir'], 'cuda'); rec = LAST[-1]
        pool = rec.pool_uJ()
        mean_run = tag.endswith('_mpmean')
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
        out.append(dict(task='MITBIH', suite=suite, tag=tag, ds='MITBIH', pl='', seed=int(r['seed']), metric=f"{r['f1']:.6f}",
                        E_old=f'{old:.6f}', E_pool=f'{new:.6f}', pooled_layers=';'.join(sorted(rec.pooled))))
        print(f"MIT-BIH {tag} s{int(r['seed'])}: f1 {f1:.6f} E {old:.3f} -> {new:.3f} uJ  pooled {sorted(rec.pooled)}", flush=True)
    return out


if __name__ == '__main__':
    LR = {'ETTh2': '0.0005', 'ETTm2': '0.0001'}
    rows = ett('flif-v1-ett-spkmux-mx3', lambda ds: {f'spkmx_lif_muxlin_iand_lr{LR[ds]}_L1_mpmean'})
    rows += ett('flif-v1-ett-spkmux-tune', lambda ds: {f'spkmx_lif_muxlin_iand_lr{LR[ds]}_L1'})
    rows += mit('flif-v1-mitbih-spkmux-mx3', 'spkmx_lif_muxlin_add_lr0.02_L2_mpmean')
    rows += mit('flif-v1-mitbih-spkmux-tune', 'spkmx_lif_muxlin_add_lr0.02_L2')
    want = 12 + 12 + 3 + 3
    if len(rows) != want:
        fail(f'{len(rows)} runs recounted, want {want}')
    with open(os.path.join(OUT, 'energy_pool.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    if BAD:
        print(f'{len(BAD)} FAILED'); sys.exit(1)
    print(f'{len(rows)} runs -> {OUT}/energy_pool.csv; all checks passed')
