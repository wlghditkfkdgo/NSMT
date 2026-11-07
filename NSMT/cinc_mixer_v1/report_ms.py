"""Independent recomputation and report-only tables of the membrane shortcut test (2026-11-08, exp/f-lif-pop-v3; PROJECT_LOG membrane shortcut pre-registration). No model, no checkpoint, no verdict changes.
Own arithmetic and own file reading (no import from summarize_ms.py or summarize_l2.py), so a summarizer bug shows up as a mismatch. Fail closed: any verdict row of a summary.md that differs stops with exit 2 and writes nothing.
Report only, chosen after the results were seen: first-epoch selection and last-epoch ETT MSE (does the shortcut move the best epoch), layer firing rates from each run's log/neuron_energy.csv (why Spikformer-style + MS fell on MIT-BIH).
usage: python -B report_ms.py   ->   results/flif-v1-ett-spkmux-ms/report_extra.md
"""
import csv
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results')
ETT, MIT = os.path.join(R, 'flif-v1-ett-spkmux-ms'), os.path.join(R, 'flif-v1-mitbih-spkmux-ms')
TUNE, BASE, MT = os.path.join(R, 'flif-v1-ett-spkmux-tune'), os.path.join(R, 'flif-v1-ett-spkmux'), os.path.join(R, 'flif-v1-mitbih-spkmux-tune')
CELLS = (('ETTh2', '96'), ('ETTh2', '720'), ('ETTm2', '96'), ('ETTm2', '720'))
SEEDS = ('42', '7', '2026')
LRM = {'ETTh2': '0.0005', 'ETTm2': '0.0001'}
LRS = {'ETTh2': '0.0005', 'ETTm2': '0.00025'}
LYS = {'ETTh2': '1', 'ETTm2': '2'}
AC = 0.9e-6                                                               # µJ


def stop(msg):
    print('STOP:', msg)
    sys.exit(2)


def avg(v):
    v = list(v)
    if not v or not all(math.isfinite(x) for x in v):
        stop(f'bad values {v}')
    return sum(v) / len(v)


def lines(p):
    out = []
    for l in open(p):
        if l.startswith('SKIP'):
            stop(f'SKIP in {p}')
        if l.startswith('RESULT'):
            out.append(dict(t.split('=', 1) for t in l.split()[1:] if '=' in t))
    return out


def ett_rows():
    add = 2 * 4 * 24 * 64 * 7 * AC
    M = {}
    en = {(e['tag'], e['ds'], e['pl'], e['seed']): float(e['total_fft_uJ_corr']) for e in csv.DictReader(open(os.path.join(ETT, 'energy_recount.csv')))}
    for r in lines(os.path.join(ETT, 'raw.txt')):
        name = 'SpkMixer + MS' if r['cond'].startswith('spkmx') else 'Spikformer-style + MS'
        M[name, r['ds'], r['pl'], r['seed']] = dict(r, E=en[r['tag'], r['ds'], r['pl'], r['seed']] + add)
    en = {(e['tag'], e['ds'], e['pl'], e['seed']): float(e['total_fft_uJ_corr']) for e in csv.DictReader(open(os.path.join(TUNE, 'energy_recount.csv')))}
    for r in lines(os.path.join(TUNE, 'raw.txt')):
        k = (r['tag'], r['ds'], r['pl'], r['seed'])
        if r['tag'] == f"spkmx_lif_muxlin_iand_lr{LRM[r['ds']]}_L1":
            M['SpkMixer', r['ds'], r['pl'], r['seed']] = dict(r, E=en[k])
        if r['ds'] == 'ETTm2' and r['tag'] == 'spk_ssa_lr0.00025_L2':
            M['Spikformer-style', r['ds'], r['pl'], r['seed']] = dict(r, E=en[k])
    en = {(e['ds'], e['pl'], e['seed']): float(e['total_fft_uJ_corr']) for e in csv.DictReader(open(os.path.join(BASE, 'baseline_spk_ssa_energy.csv')))}
    for r in lines(os.path.join(BASE, 'baseline_spk_ssa_raw.txt')):
        if r['ds'] == 'ETTh2':
            M['Spikformer-style', r['ds'], r['pl'], r['seed']] = dict(r, E=en[r['ds'], r['pl'], r['seed']])
    for n in ('SpkMixer', 'SpkMixer + MS', 'Spikformer-style', 'Spikformer-style + MS'):
        for ds, pl in CELLS:
            for s in SEEDS:
                if (n, ds, pl, s) not in M:
                    stop(f'missing {n} {ds} {pl} {s}')
    return M


def mit_rows():
    add = 2 * 4 * 11 * 384 * AC
    M = {}
    en = {(e['tag'], e['seed']): float(e['total_fft_uJ_corr']) for e in csv.DictReader(open(os.path.join(MIT, 'energy_recount.csv')))}
    for r in lines(os.path.join(MIT, 'raw.txt')):
        name = 'SpkMixer + MS' if r['cond'].startswith('spkmx') else 'Spikformer-style + MS'
        M[name, r['seed']] = dict(r, E=en[r['tag'], r['seed']] + add)
    en = {(e['tag'], e['seed'], e['val_fold']): float(e['total_fft_uJ_corr']) for e in csv.DictReader(open(os.path.join(MT, 'energy_recount.csv')))}
    for r in lines(os.path.join(MT, 'raw.txt')):
        if r['val_fold'] != '-1':
            continue
        if r['tag'] == 'spkmx_lif_muxlin_add_lr0.02_L2':
            M['SpkMixer', r['seed']] = dict(r, E=en[r['tag'], r['seed'], '-1'])
        if r['tag'] == 'spk_ssa_lr0.005_L2':
            M['Spikformer-style', r['seed']] = dict(r, E=en[r['tag'], r['seed'], '-1'])
    for n in ('SpkMixer', 'SpkMixer + MS', 'Spikformer-style', 'Spikformer-style + MS'):
        for s in SEEDS:
            if (n, s) not in M:
                stop(f'missing MIT-BIH {n} {s}')
    return M


Q = (('M1', 'SpkMixer + MS', 'SpkMixer'), ('M2', 'Spikformer-style + MS', 'Spikformer-style'), ('M3', 'SpkMixer + MS', 'Spikformer-style + MS'), ('M4', 'SpkMixer + MS', 'Spikformer-style'))


def verify():
    E, M = ett_rows(), mit_rows()
    n = 0
    txt = open(os.path.join(ETT, 'summary.md')).read()
    for q, a, b in Q:
        pairs, ratio = {}, []
        for ds, pl in CELLS:
            pairs[ds, pl] = [100 * (float(E[a, ds, pl, s]['mse']) / float(E[b, ds, pl, s]['mse']) - 1) for s in SEEDS]
            ratio.append(avg(E[a, ds, pl, s]['E'] for s in SEEDS) / avg(E[b, ds, pl, s]['E'] for s in SEEDS))
        all_p = [x for v in pairs.values() for x in v]
        Rv, lo, hi = avg(all_p), sum(x < 0 for x in all_p), sum(x > 0 for x in all_p)
        v = 'better' if Rv <= -1 and lo >= 8 else 'worse' if Rv >= 1 and hi >= 8 else 'no clear difference'
        m = re.search(rf'^\| {q}[^|]*\| ([+-][\d.]+) \| (\d+)/12 \| ([a-z ]+) \| ([\d.]+) \|', txt, re.M)
        if m is None or abs(float(m[1]) - Rv) > .0051 or int(m[2]) != lo or m[3].strip() != v or abs(float(m[4]) - avg(ratio)) > .0051:
            stop(f'ETT {q}: summary {m and m.groups()} vs own {Rv:.3f} {lo} {v} {avg(ratio):.3f}')
        n += 1
    txt = open(os.path.join(MIT, 'summary.md')).read()
    for q, a, b in Q:
        d = [100 * (float(M[a, s]['f1']) - float(M[b, s]['f1'])) for s in SEEDS]
        dm, hi, lo = avg(d), sum(x > 0 for x in d), sum(x < 0 for x in d)
        v = 'better' if dm >= .5 and hi >= 2 else 'worse' if dm <= -.5 and lo >= 2 else 'no clear difference'
        ratio = avg(M[a, s]['E'] for s in SEEDS) / avg(M[b, s]['E'] for s in SEEDS)
        m = re.search(rf'^\| {q}[^|]*\| ([+-][\d.]+) \| (\d)/3 \| [^|]*\| ([a-z ]+) \| ([\d.]+) \|', txt, re.M)
        if m is None or abs(float(m[1]) - dm) > .0051 or int(m[2]) != hi or m[3].strip() != v or abs(float(m[4]) - ratio) > .0051:
            stop(f'MIT-BIH {q}: summary {m and m.groups()} vs own {dm:.3f} {hi} {v} {ratio:.3f}')
        n += 1
    return E, M, n


def layers(r):
    p = os.path.join(r.get('dir', ''), 'log', 'neuron_energy.csv')
    if not os.path.exists(p):
        return None
    return {re.sub(r'\.neuron$', '', x['module']): float(x['rho']) for x in csv.DictReader(open(p)) if x['module'] != 'TOTAL'}


def main():
    E, M, n = verify()
    L = ['# Report-only tables of the membrane shortcut test', '', f'Generated by report_ms.py. All {n} comparison rows (M1-M4 on ETT and MIT-BIH: value, count, verdict, energy ratio) were recomputed independently and match the summaries. The tables below were chosen after the results were seen.', '']
    L += ['## ETT: selected epoch and last-epoch weights', '', '| model | runs selecting epoch 0 | mean selected epoch | test MSE, best-validation weights | test MSE, last-epoch weights | last / best |', '|---|--:|--:|--:|--:|--:|']
    for name in ('SpkMixer', 'SpkMixer + MS', 'Spikformer-style', 'Spikformer-style + MS'):
        rr = [E[name, ds, pl, s] for ds, pl in CELLS for s in SEEDS]
        be = [int(x['best_epoch_val_epochs'].split(':')[0]) for x in rr]
        rows = [(float(x['mse']), float(x['last_mse'])) for x in rr if 'last_mse' in x]
        if len(rows) == 12:
            L.append(f"| {name} | {sum(b == 0 for b in be)}/12 | {avg(be):.1f} | {avg(a for a, _ in rows):.4f} | {avg(b for _, b in rows):.4f} | {avg(b / a for a, b in rows):.3f} |")
        else:
            L.append(f"| {name} | {sum(b == 0 for b in be)}/12 | {avg(be):.1f} | {avg(float(x['mse']) for x in rr):.4f} | not recorded on all runs | - |")
    L += ['', '## ETT: rho_hippo, shortcut / control, by cell', '', '| model | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 |', '|---|--:|--:|--:|--:|']
    for a, b in (('SpkMixer + MS', 'SpkMixer'), ('Spikformer-style + MS', 'Spikformer-style')):
        L.append(f'| {a} | ' + ' | '.join(f"{avg(float(E[a, ds, pl, s]['rho_hippo']) for s in SEEDS) / avg(float(E[b, ds, pl, s]['rho_hippo']) for s in SEEDS):.2f}" for ds, pl in CELLS) + ' |')
    L += ['', '## MIT-BIH: layer firing rates (mean of 3 seeds)', '']
    Ls = (('Spikformer-style', 'data_block.0', ('encoding.emb_lif', 'ssa.q_lif', 'ssa.k_lif', 'ssa.v_lif', 'ssa.attn_lif', 'ssa.proj_linear.lif', 'ms_mix.lif', 'mlp.fc1.lif', 'mlp.fc2.lif')),
          ('SpkMixer', 'blocks.0', ('encoding.emb_lif', 'mixer.Mem.1.lif', 'mixer_rev.Mem.1.lif', 'sd_neuron', 'fuse.neuron', 'ms_mix.lif', 'fc1.lif', 'fc2.lif')))
    for base, pre, names in Ls:
        L += [f'| layer | {base} | {base} + MS |', '|---|--:|--:|']
        for nm in names:
            key = nm if nm.startswith('encoding') else f'{pre}.{nm}'
            vals = []
            for name in (base, base + ' + MS'):
                v = [layers(M[name, s]) for s in SEEDS]
                vals.append('-' if any(x is None or key not in x for x in v) else f'{avg(x[key] for x in v):.4f}')
            L.append(f'| {nm} | {vals[0]} | {vals[1]} |')
        L.append('')
    L += ['## MIT-BIH: F1 per class, change with the shortcut, %p (mean of 3 seeds)', '', '| model | N | S | V | F | Q |', '|---|--:|--:|--:|--:|--:|']
    for a, b in (('SpkMixer + MS', 'SpkMixer'), ('Spikformer-style + MS', 'Spikformer-style')):
        L.append(f'| {a} | ' + ' | '.join(f"{avg(100 * (float(M[a, s][f'f1_{i}']) - float(M[b, s][f'f1_{i}'])) for s in SEEDS):+.2f}" for i in range(5)) + ' |')
    p = os.path.join(ETT, 'report_extra.md')
    with open(p, 'w') as f:
        f.write('\n'.join(L) + '\n')
    print('recomputation matches;', n, 'rows; wrote', p)
    print('\n'.join(L))


if __name__ == '__main__':
    main()
