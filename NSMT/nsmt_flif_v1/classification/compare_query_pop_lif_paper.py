"""MODI 2026-11-05 (no training): result table of [query pop LIF NSMT T=4] (nsmt4q_pop_lif) next to the NSMT paper and Spikformer-style
(user request 11-05; docs/MODI_PROJECT_LOG.md 2026-11-05). Everything is recomputed from existing RESULT lines.

Version table: every version of the model that exists, with validation and test, so the choice of the reported version can be read.
Comparison rows, ETT (ETTh2 / ETTm2 x H96 / H720, standard test split, seeds 42 / 7 / 2026):
  query        flif-v1-ett-query (ETTh2) + flif-v1-ett-query-m2 (ETTm2); patience 10, best-validation weights
  NSMT same    flif-v1-s2p10 cond base, *_best fields: the original NSMT trained under the same stopping rule
  NSMT paper   flif-v1-main cond base: the paper's own rule (patience 2, last weights); seed 42 is the paper's Table 13 value
  Spikformer   flif-v1-gen-ett cond spk_ssa: Spikformer-style (T_h 4, no neocortex, same script family and stopping rule as query)
  paper rows   Table 13 / 14 / 5 / 6 of papers/NSMT.pdf, as printed (3 decimals, seed 42). The paper has no plain Spikformer row:
               SpikF is the Spiking Fourier Network [43] and iSpikformer [24] is the inverted variant.
Rules (the ones used since 10-27): ETT r = 100 (A / B - 1) per cell and seed; R = mean; better: R <= -1 % and >= 8/12 pairs lower;
worse: the mirror; per cell +-1 % and 2/3 seeds. MIT-BIH: mean >= +0.5 %p and 2/3 seeds. A missing, duplicated or non-finite value raises.
Energy (uJ): synapses + neurons (total_fft). The flat head of the T_h-mean forecaster is counted as spike additions (energy_recount.csv, the
standard since 10-28), which is how syops already counts the per-step head of the original NSMT and the H6 layers that read spikes.
syn = synapses only, the quantity of the paper's Tables 5 and 6.
usage (classification folder): python compare_query_pop_lif_paper.py [--out FILE]
"""
import argparse
import csv
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FC = os.path.join(HERE, '..', 'forecasting', 'results')
CL = os.path.join(HERE, 'results')
SEEDS = (42, 7, 2026)
CELLS = (('ETTh2', 96), ('ETTh2', 720), ('ETTm2', 96), ('ETTm2', 720))
COLS = ' | '.join(f'{d} H{p}' for d, p in CELLS)
# papers/NSMT.pdf Table 13 (NSMT) and Table 14 (the others): (MSE, MAE) as printed
PAPER = {'NSMT': ((0.288, 0.334), (0.409, 0.429), (0.175, 0.251), (0.399, 0.393)),
         'SpikF': ((0.290, 0.336), (0.416, 0.436), (0.175, 0.251), (0.405, 0.397)),
         'SpikeRNN': ((0.295, 0.335), (0.428, 0.438), (0.176, 0.251), (0.409, 0.396)),
         'iSpikformer': ((0.308, 0.349), (0.438, 0.439), (0.178, 0.256), (0.406, 0.398))}
PAPER_H720_SYN = {'ETTh2': 44.90, 'ETTm2': 46.72}           # Table 6, NSMT (SpikF 1050.47 / 1050.10)
PAPER_MITBIH = (98.45, 91.91, 19.05)                        # Table 5: accuracy, F1, energy


def results(path, **sel):
    out = []
    for ln in open(path):
        if not ln.startswith('RESULT'):
            continue
        d = dict(t.split('=', 1) for t in ln.split()[1:] if '=' in t)
        if all(d.get(k) == str(v) for k, v in sel.items()):
            out.append(d)
    return out


def num(d, key):
    v = float(d[key])
    if not math.isfinite(v):
        raise RuntimeError(f'non-finite {key}')
    return v


def by_seed(rows, what):
    got = {int(r['seed']): r for r in rows}
    if len(rows) != 3 or set(got) != set(SEEDS):
        raise RuntimeError(f'{what}: expected one row per seed {SEEDS}, got {sorted(int(r["seed"]) for r in rows)}')
    return got


def recounts(*suites):
    out = {}
    for s in suites:
        for r in csv.DictReader(open(os.path.join(FC, s, 'energy_recount.csv'))):
            k = (r['ds'], int(r['pl']), r['cond'], int(r['seed']))
            if k in out:
                raise RuntimeError(f'duplicate recount {k}')
            out[k] = float(r['total_fft_uJ_corr'])
    return out


def row(d, mse, mae, syn, tot, par):
    return dict(mse=num(d, mse), mae=num(d, mae), syn=num(d, syn), tot=num(d, tot), par=int(d[par]))


def ett_recounted(cond, suites):
    """Dual-path / Spikformer-style forecaster runs: the flat head's MAC cost replaced by the spike-driven recount."""
    rc = recounts(*suites); out = {}
    for ds, pl in CELLS:
        rows = [r for s in suites for r in results(os.path.join(FC, s, 'raw.txt'), ds=ds, pl=pl, cond=cond)]
        for sd, d in by_seed(rows, f'{cond} {ds} {pl}').items():
            x = row(d, 'mse', 'mae', 'syn_energy_uJ', 'total_fft_uJ', 'params'); x['val'] = num(d, 'val_loss_best')
            corr = rc[(ds, pl, cond, sd)]; x['syn'] -= x['tot'] - corr; x['tot'] = corr
            out[ds, pl, sd] = x
    return out


def ett_h6():
    out = {}
    for ds, pl in CELLS:
        rows = results(os.path.join(FC, 'flif-v1-ett-h6', 'raw.txt'), ds=ds, pl=pl, cond='nsmt4q_pop_lif_h6')
        for sd, d in by_seed(rows, f'h6 {ds} {pl}').items():
            x = row(d, 'mse', 'mae', 'syn_energy_uJ', 'total_fft_uJ', 'params'); x['val'] = num(d, 'val_loss_best')
            out[ds, pl, sd] = x
    return out


def ett_base(suite, keys):
    out = {}
    for ds, pl in CELLS:
        rows = results(os.path.join(FC, suite, 'raw.txt'), cond='base', dataset=f'{ds}_p{pl}')
        for sd, d in by_seed(rows, f'{suite} base {ds} {pl}').items():
            out[ds, pl, sd] = row(d, *keys, 'params')
    return out


def cell_vals(t, ds, pl, key):
    return [t[ds, pl, s][key] for s in SEEDS]


def ms(v, f='{:.4f}'):
    return f.format(np.mean(v)) + ' ± ' + f.format(np.std(v, ddof=1))


def compare(a, b):
    """r per cell and seed, the 12-pair verdict, and the mean per-cell energy ratio."""
    per = {c: [100 * (a[c[0], c[1], s]['mse'] / b[c[0], c[1], s]['mse'] - 1) for s in SEEDS] for c in CELLS}
    allr = [x for c in CELLS for x in per[c]]; R = float(np.mean(allr)); low = sum(x < 0 for x in allr); high = sum(x > 0 for x in allr)
    v = 'better' if R <= -1 and low >= 8 else 'worse' if R >= 1 and high >= 8 else 'no clear difference'
    cv = {}
    for c in CELLS:
        m = float(np.mean(per[c])); lo = sum(x < 0 for x in per[c]); hi = sum(x > 0 for x in per[c])
        cv[c] = (m, 'better' if m <= -1 and lo >= 2 else 'worse' if m >= 1 and hi >= 2 else 'no clear difference')
    en = float(np.mean([np.mean(cell_vals(a, *c, 'tot')) / np.mean(cell_vals(b, *c, 'tot')) for c in CELLS]))
    return dict(per=per, R=R, low=low, v=v, cv=cv, en=en)


def cmp_row(name, c):
    cells = ' | '.join(f"{c['cv'][x][0]:+.2f} ({c['cv'][x][1]})" for x in CELLS)
    return f"| {name} | {cells} | {c['R']:+.2f} | {c['low']}/12 | {c['v']} | {c['en']:.2f} |"


def mitbih(path, cond, **sel):
    rows = results(path, cond=cond, **sel)
    return {s: dict(f1=100 * num(d, 'f1'), acc=100 * num(d, 'overall_acc'), syn=num(d, 'syn_energy_uJ'), tot=num(d, 'total_fft_uJ'),
                    par=int(d['params']), f1s=[100 * num(d, f'f1_{i}') for i in range(5)])
            for s, d in by_seed(rows, f'{cond} {sel}').items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'flif-v1-query-pop-lif-vs-paper.md'))
    a = ap.parse_args()
    q = ett_recounted('nsmt4q_pop_lif', ('flif-v1-ett-query', 'flif-v1-ett-query-m2'))
    h6 = ett_h6()
    sp = ett_recounted('spk_ssa', ('flif-v1-gen-ett',))
    same = ett_base('flif-v1-s2p10', ('mse_best', 'mae_best', 'syn_energy_best', 'total_fft_uJ_best'))
    prule = ett_base('flif-v1-main', ('mse', 'mae', 'syn_energy', 'total_fft_uJ'))

    # fail-closed checks: the reproduction equals the printed paper values, and the earlier summaries are reproduced
    for i, c in enumerate(CELLS):
        x = prule[c[0], c[1], 42]
        if abs(x['mse'] - PAPER['NSMT'][i][0]) > 0.00051 or abs(x['mae'] - PAPER['NSMT'][i][1]) > 0.00051:
            raise RuntimeError(f'NSMT seed 42 does not reproduce Table 13 at {c}')
    for ds, v in PAPER_H720_SYN.items():
        if abs(prule[ds, 720, 42]['syn'] - v) > 0.0051:
            raise RuntimeError(f'NSMT seed 42 does not reproduce Table 6 at {ds}')
    c_sp, c_same, c_rule, c_h6 = compare(q, sp), compare(q, same), compare(q, prule), compare(h6, q)
    if round(c_sp['R'], 2) != -0.68 or c_sp['low'] != 9 or round(c_h6['R'], 2) != -0.25 or c_h6['low'] != 7:
        raise RuntimeError(f"earlier ETT verdicts not reproduced: {c_sp['R']} {c_sp['low']} {c_h6['R']} {c_h6['low']}")

    L = ['# [query pop LIF NSMT T=4] next to the NSMT paper and Spikformer-style', '',
         'See the module docstring for sources and rules. Checks passed: seed 42 of the NSMT reproduction equals Table 13 (4 cells) and Table 6 '
         '(H720), and the earlier ETT verdicts (query vs Spikformer-style, H6 vs flat) are reproduced.', '']

    # versions
    L += ['## Versions of the model that exist (ETT)', '',
          '| version | cells | ' + COLS + ' | mean validation L1 vs default |', '|---|---|---|---|---|---|---|']
    val = lambda t, c: float(np.mean(cell_vals(t, *c, 'val')))
    tune = results(os.path.join(FC, 'flif-v1-ett-tune', 'raw.txt'), cond='nsmt4q_pop_lif')
    L.append('| default: flat head, lr 5e-4, soma reset subtract (test MSE) | 4 | ' +
             ' | '.join(f"{np.mean(cell_vals(q, *c, 'mse')):.4f}" for c in CELLS) + ' | 1.0000 |')
    ratio = float(np.mean([val(h6, c) / val(q, c) for c in CELLS]))
    L.append('| H6: linear shortcut + small head | 4 | ' + ' | '.join(f"{np.mean(cell_vals(h6, *c, 'mse')):.4f}" for c in CELLS) + f' | {ratio:.4f} |')
    two = CELLS[:2]
    for label, sel in (('lr 2.5e-4', dict(lr='0.00025', reset='spikede')), ('lr 1e-3', dict(lr='0.001', reset='spikede')),
                       ('lr 2e-3', dict(lr='0.002', reset='spikede')), ('reset zero', dict(lr='0.0005', reset='zero'))):
        mse, rel = [], []
        for ds, pl in two:
            rr = by_seed([r for r in tune if r['ds'] == ds and r['pl'] == str(pl) and all(r[k] == v for k, v in sel.items())], f'{label} {ds} {pl}')
            mse.append(float(np.mean([num(r, 'mse') for r in rr.values()])))
            rel.append(float(np.mean([num(r, 'val_loss_best') for r in rr.values()])) / val(q, (ds, pl)))
        L.append(f'| {label} | ETTh2 only | {mse[0]:.4f} | {mse[1]:.4f} | not run | not run | {np.mean(rel):.4f} |')
    sel_lr = json.load(open(os.path.join(FC, 'flif-v1-ett-tune', 'selected_lr.json')))['selected']['nsmt4q_pop_lif']
    sel_re = json.load(open(os.path.join(FC, 'flif-v1-ett-tune', 'selected_reset.json')))['selected']['nsmt4q_pop_lif']
    L += ['', f'Pre-registered selections on validation (ETTh2 two cells): lr {sel_lr}, reset {sel_re}. '
          'Per-cell validation L1 of default / H6: ' + '; '.join(f'{c[0]} H{c[1]} {val(q, c):.6f} / {val(h6, c):.6f}' for c in CELLS) + '.', '']

    # ETT results
    L += ['## ETT test MSE / MAE (mean ± sd of seeds 42, 7, 2026 unless noted; lower is better)', '',
          '| row | stopping rule | ' + COLS + ' |', '|---|---|---|---|---|---|']
    for name, rule, t in (('**query pop LIF (default)**', 'patience 10, best-validation weights', q), ('query pop LIF, H6 head', 'patience 10, best-validation weights', h6),
                          ('NSMT reproduction, same rule', 'patience 10, best-validation weights', same), ('NSMT reproduction, paper rule', 'patience 2, last weights', prule),
                          ('Spikformer-style', 'patience 10, best-validation weights', sp)):
        L.append(f'| {name} MSE | {rule} | ' + ' | '.join(ms(cell_vals(t, *c, 'mse')) for c in CELLS) + ' |')
        L.append(f'| {name} MAE | | ' + ' | '.join(ms(cell_vals(t, *c, 'mae')) for c in CELLS) + ' |')
    L.append('| query pop LIF (default), seed 42 only, MSE / MAE | patience 10, best-validation weights | ' +
             ' | '.join(f"{q[c[0], c[1], 42]['mse']:.4f} / {q[c[0], c[1], 42]['mae']:.4f}" for c in CELLS) + ' |')
    for k, lab in (('NSMT', 'NSMT paper Table 13'), ('iSpikformer', 'iSpikformer, paper Table 14'), ('SpikeRNN', 'SpikeRNN, paper Table 14'),
                   ('SpikF', 'SpikF (Spiking Fourier Network, not Spikformer), paper Table 14')):
        L.append(f'| {lab}, MSE / MAE | seed 42, as printed | ' + ' | '.join(f'{m:.3f} / {e:.3f}' for m, e in PAPER[k]) + ' |')

    # comparisons
    L += ['', '## query pop LIF (default) vs the others: r % of test MSE (negative = query lower); per cell mean and verdict', '',
          '| vs | ' + COLS + ' | R % | pairs lower | verdict | energy A/B |', '|---|---|---|---|---|---|---|---|---|']
    L.append(cmp_row('NSMT reproduction, same rule', c_same))
    L.append(cmp_row('NSMT reproduction, paper rule (different rule)', c_rule))
    L.append(cmp_row('Spikformer-style', c_sp))
    L.append(cmp_row('(H6 vs query default)', c_h6))
    c6s, c6p = compare(h6, same), compare(h6, sp)
    L.append(cmp_row('(H6 vs NSMT reproduction, same rule)', c6s))
    L.append(cmp_row('(H6 vs Spikformer-style)', c6p))
    L += ['', 'Against the printed Table 13 values (r % of MSE; rounding of the printed value is up to 0.3 % at H96 of ETTm2):', '',
          '| | ' + COLS + ' | mean |', '|---|---|---|---|---|---|']
    pm = [100 * (float(np.mean(cell_vals(q, *c, 'mse'))) / PAPER['NSMT'][i][0] - 1) for i, c in enumerate(CELLS)]
    ps = [100 * (q[c[0], c[1], 42]['mse'] / PAPER['NSMT'][i][0] - 1) for i, c in enumerate(CELLS)]
    L.append('| 3-seed mean vs Table 13 | ' + ' | '.join(f'{x:+.2f}' for x in pm) + f' | {np.mean(pm):+.2f} |')
    L.append('| seed 42 vs Table 13 (same seed) | ' + ' | '.join(f'{x:+.2f}' for x in ps) + f' | {np.mean(ps):+.2f} |')

    # energy
    L += ['', '## ETT energy (uJ, mean of 3 seeds) and parameters', '',
          '| row | quantity | ' + COLS + ' |', '|---|---|---|---|---|---|']
    for name, t in (('query pop LIF (default)', q), ('query pop LIF, H6 head', h6), ('NSMT reproduction, same rule', same),
                    ('NSMT reproduction, paper rule', prule), ('Spikformer-style', sp)):
        L.append(f'| {name} | synapses + neurons | ' + ' | '.join(f"{np.mean(cell_vals(t, *c, 'tot')):.2f}" for c in CELLS) + ' |')
        L.append(f'| {name} | synapses only | ' + ' | '.join(f"{np.mean(cell_vals(t, *c, 'syn')):.2f}" for c in CELLS) + ' |')
        L.append(f'| {name} | parameters | ' + ' | '.join(f"{t[c[0], c[1], 42]['par']:,}" for c in CELLS) + ' |')
    L.append('| NSMT paper Table 6 | synapses only | not printed | 44.90 | not printed | 46.72 |')

    # MIT-BIH
    mq = mitbih(os.path.join(CL, 'flif-v1-mitbih-query', 'raw.txt'), 'nsmt4q_pop_lif', lr='0.01', val='0')
    mh = mitbih(os.path.join(CL, 'flif-v1-mitbih-h2', 'raw.txt'), 'nsmt4q_pop_lif_hattn')
    mb = mitbih(os.path.join(CL, 'flif-v1-cls', 'raw.txt'), 'base')
    ms_ = mitbih(os.path.join(CL, 'flif-v1-gen-mitbih', 'raw.txt'), 'spk_ssa')
    tq = {lr: mitbih(os.path.join(CL, 'flif-v1-mitbih-query-tune', 'raw.txt'), 'nsmt4q_pop_lif', lr=lr, val='1') for lr in ('0.01', '0.02')}
    if round(np.mean([mq[s]['f1'] for s in SEEDS]), 2) != 92.17 or round(np.mean([ms_[s]['f1'] for s in SEEDS]), 2) != 92.22:
        raise RuntimeError('earlier MIT-BIH means not reproduced')
    L += ['', '## MIT-BIH test (seeds 42, 7, 2026, last-epoch weights, 30 epochs, lr 0.01)', '',
          '| row | accuracy % | macro-F1 % | F1 N / S / V / F / Q | synapses uJ | synapses + neurons uJ | parameters |', '|---|---|---|---|---|---|---|']
    for name, t in (('**query pop LIF (default)**', mq), ('query pop LIF, H2 head', mh), ('NSMT reproduction', mb), ('Spikformer-style', ms_)):
        fs = np.mean([t[s]['f1s'] for s in SEEDS], axis=0)
        L.append(f"| {name} | {ms([t[s]['acc'] for s in SEEDS], '{:.2f}')} | {ms([t[s]['f1'] for s in SEEDS], '{:.2f}')} | " + ' / '.join(f'{x:.2f}' for x in fs) +
                 f" | {np.mean([t[s]['syn'] for s in SEEDS]):.2f} | {np.mean([t[s]['tot'] for s in SEEDS]):.2f} | {t[42]['par']:,} |")
    L.append(f'| NSMT paper Table 5 (as printed) | {PAPER_MITBIH[0]:.2f} | {PAPER_MITBIH[1]:.2f} | not printed | {PAPER_MITBIH[2]:.2f} | not printed | not printed |')
    L += ['', '| query pop LIF (default) vs | Δ macro-F1 %p (42 / 7 / 2026) | mean | verdict | energy A/B |', '|---|---|---|---|---|']
    for name, t in (('NSMT reproduction', mb), ('Spikformer-style', ms_)):
        d = [mq[s]['f1'] - t[s]['f1'] for s in SEEDS]; m = float(np.mean(d))
        v = 'better' if m >= 0.5 and sum(x > 0 for x in d) >= 2 else 'worse' if m <= -0.5 and sum(x < 0 for x in d) >= 2 else 'no clear difference'
        L.append(f"| {name} | " + ' / '.join(f'{x:+.2f}' for x in d) + f" | {m:+.2f} | {v} | {np.mean([mq[s]['tot'] for s in SEEDS]) / np.mean([t[s]['tot'] for s in SEEDS]):.2f} |")
    L.append(f"| NSMT paper Table 5 (one printed value) | | {np.mean([mq[s]['f1'] for s in SEEDS]) - PAPER_MITBIH[1]:+.2f} | not judged | synapses {np.mean([mq[s]['syn'] for s in SEEDS]) / PAPER_MITBIH[2]:.2f} |")
    t01, t02 = tq['0.01'], tq['0.02']
    L += ['', f"lr on MIT-BIH (80 / 20 split of the training file, validation picks lr 0.02): test {np.mean([t02[s]['f1'] for s in SEEDS]):.2f} vs {np.mean([t01[s]['f1'] for s in SEEDS]):.2f} at lr 0.01 in the same split; "
          'lr 0.02 was not run on the full training file.', '']
    L += ['Not run for this model: ETTh1, ETTm1, the other forecasting datasets, anomaly detection, UEA classification, and the paper\'s stopping rule (patience 2, last weights). '
          'CinC is not in the paper; results/flif-v1-followup-summary.md has this model (90 epochs) at 67.73 vs Spikformer-style (30 epochs) 65.74 over 5 folds.', '']
    with open(a.out, 'w') as f:
        f.write('\n'.join(L))
    print('\n'.join(L))


if __name__ == '__main__':
    main()
